import json

import pytest

from bialy.student_host import rental
from bialy.student_host.jobs import JobError, tree
from bialy.student_host.resources import write
from bialy.student_host.specification import identity


def test_restart_observes_existing_worker_without_launching_again(tmp_path, monkeypatch):
    config = {'registry': str(tmp_path / 'registry')}
    config_path = tmp_path / 'configuration.json'
    write(config_path, config)
    splits = tmp_path / 'splits'
    splits.mkdir()
    (splits / 'train.jsonl').write_text('fixture')
    directory = tmp_path / 'rental'
    directory.mkdir()
    write(directory / 'binding.json', {'configuration': identity(config), 'student': 'e4b', 'splits': tree(splits)})
    write(directory / 'resource.json', {'token': 'owned'})
    write(directory / 'outcome.json', {'status': 'running', 'token': 'owned'})
    write(directory / 'remote.json', {'directory': '/scratch/runs/owned'})
    actions = []
    class Resources:
        def __init__(self, *args):
            pass
        def recover(self, token, seconds):
            actions.append(('recover', token))
            return {'server': {'public_ip': {'address': '192.0.2.1'}}}
        def release(self, token):
            actions.append(('release', token))
    class Remote:
        def __init__(self, address):
            pass
        def command(self, args, **kwargs):
            if args == ['true']:
                return ''
            assert args[:2] == ['python3', '-c']
            return json.dumps({'execution': {'status': 'failed', 'failure': {'code': 'out_of_memory'}}, 'heartbeat_age': 0})
        def receive(self, source, destination):
            actions.append(('receive', source))
    monkeypatch.setattr(rental, 'provider', lambda config: None)
    monkeypatch.setattr(rental, 'Resources', Resources)
    monkeypatch.setattr(rental, 'Remote', Remote)
    monkeypatch.setattr(rental, 'collect_checkpoint', lambda *args: [])
    with pytest.raises(JobError, match='did not complete'):
        rental.train(config_path, 'e4b', splits, 'unused', tmp_path, resume=directory, sleep=lambda _: None)
    assert actions == [('recover', 'owned'), ('receive', '/scratch/runs/owned/'), ('release', 'owned')]
    outcome = json.loads((directory / 'outcome.json').read_text())
    assert outcome['status'] == 'failed' and outcome['cleanup'] == 'confirmed'
