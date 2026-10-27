import json
from types import SimpleNamespace

import pytest

from bialy.campaign.containers import Container
from bialy.campaign.journal import CampaignFault


class Docker:
    def __init__(self):
        self.state, self.calls = None, []
    def __call__(self, argv, **kwargs):
        self.calls.append(argv)
        if argv[1:3] == ['container', 'inspect']:
            return SimpleNamespace(returncode=0 if self.state else 1, stdout=json.dumps([self.state]) if self.state else '')
        if argv[1] == 'create':
            label = next(value for value in argv if value.startswith('bialy.campaign='))
            self.state = {'Id': 'campaign-container', 'Config': {'Labels': {'bialy.campaign': label.split('=', 1)[1]}},
                          'State': {'Status': 'created', 'ExitCode': 127, 'OOMKilled': False}}
        elif argv[1] == 'start':
            self.state['State']['Status'] = 'running'
        elif argv[1] == 'wait':
            self.state['State']['Status'] = 'exited'
        elif argv[1] == 'rm':
            self.state = None
        return SimpleNamespace(returncode=0, stdout='')


def test_failed_task_is_retained_and_never_replayed_on_restart(tmp_path):
    docker = Docker()
    container = Container(tmp_path, run=docker)
    result = container.execute('campaign-container', ['docker', 'run', '--rm', '--name', 'campaign-container', 'image'], {'task': 'task'})
    assert result['exit_status'] == 127
    assert container.reconcile() == result
    assert len([call for call in docker.calls if call[1] == 'start']) == 1
    container.release()
    assert container.reconcile() == result
    assert (tmp_path / 'container-released.json').exists()


def test_running_campaign_task_is_reattached_without_restart(tmp_path):
    docker = Docker()
    container = Container(tmp_path, run=docker)
    container.execute('campaign-container', ['docker', 'run', '--name', 'campaign-container', 'image'], {'task': 'task'})
    (tmp_path / 'container-terminal.json').unlink()
    docker.state['State']['Status'] = 'running'
    container.reconcile()
    assert len([call for call in docker.calls if call[1] == 'start']) == 1


def test_foreign_container_is_not_adopted_or_deleted(tmp_path):
    docker = Docker()
    container = Container(tmp_path, run=docker)
    container.execute('campaign-container', ['docker', 'run', '--name', 'campaign-container', 'image'], {'task': 'task'})
    docker.state['Id'] = 'foreign'
    with pytest.raises(CampaignFault, match='binding_mismatch'):
        container.release()
    assert not any(call[1] == 'rm' for call in docker.calls)


def test_unknown_outcome_never_creates_a_replacement_task(tmp_path):
    docker = Docker()
    container = Container(tmp_path, run=docker)
    container.execute('campaign-container', ['docker', 'run', '--name', 'campaign-container', 'image'], {'task': 'task'})
    (tmp_path / 'container-terminal.json').unlink()
    docker.state = None
    with pytest.raises(CampaignFault, match='outcome_unknown'):
        container.reconcile()
    assert len([call for call in docker.calls if call[1] == 'create']) == 1


def test_lost_create_response_reconciles_the_campaign_container(tmp_path):
    import subprocess
    docker = Docker()
    def disconnected(argv, **kwargs):
        result = docker(argv, **kwargs)
        if argv[1] == 'create':
            raise subprocess.CalledProcessError(1, argv)
        return result
    result = Container(tmp_path, run=disconnected).execute(
        'campaign-container', ['docker', 'run', '--name', 'campaign-container', 'image'], {'task': 'task'})
    assert result['exit_status'] == 127
    assert sum(call[1] == 'create' for call in docker.calls) == 1
    assert sum(call[1] == 'start' for call in docker.calls) == 1
