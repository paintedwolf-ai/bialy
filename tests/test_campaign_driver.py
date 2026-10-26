import json
from types import SimpleNamespace

import pytest
import yaml

from bialy.campaign.driver import EpisodeDriver
from bialy.campaign.journal import CampaignFault, Journal
from bialy.campaign.lineage import FAMILIES, PARTITIONS, Lineage


def test_episode_request_uses_wire_identity_and_the_configured_execution_deadline(tmp_path, monkeypatch):
    journal = Journal(tmp_path / 'campaign', {})
    lineage = Lineage(journal)
    tasks = [{'id': partition + family, 'family': family, 'partition': partition, 'source_class': 'synthetic',
              'workspace': 'python', 'project_path': '/work/python', 'oracle_directory': '.', 'prompt': 'Implement the task.'}
             for partition in PARTITIONS for family in FAMILIES]
    lineage.roots(tasks)
    lineage.freeze()
    factory = SimpleNamespace(workspace=lambda name: SimpleNamespace(name=name, kind='stack'), fleet={'prompt_timeout': '45m'})
    run = SimpleNamespace(factory=factory, checkout=tmp_path, setting=lambda key: {'epoch': 'epoch', 'students': {}, 'host_commit': 'commit'})
    monkeypatch.setattr('bialy.campaign.driver.inspect', lambda *args: {})
    provider = yaml.safe_dump({'providers': [{'id': 'provider'}]})
    monkeypatch.setattr('bialy.campaign.driver.fleet.pilot_env', lambda pilot: {})
    monkeypatch.setattr('bialy.campaign.driver.fleet.sidecar_files',
                        lambda *args: (provider, '{}', '', {'teacher': 'provider'}, {'teacher': 'wire-model'}))
    recorded = []
    def container_args(factory, run_name, shard, mounts, image):
        recorded.append(json.loads((shard / 'workspace.json').read_text()))
        recorded.append(json.loads((shard / 'tasks.jsonl').read_text()))
        return 'container', tmp_path / 'work', ['docker', 'run', '--name', 'container', image]
    monkeypatch.setattr('bialy.campaign.driver.fleet.container_args', container_args)
    monkeypatch.setattr('bialy.campaign.driver.Container', lambda *args: SimpleNamespace(execute=lambda *args: None))
    driver = EpisodeDriver(journal, lineage, run, None, tmp_path)
    monkeypatch.setattr(driver, 'settle', lambda inputs, attempt: {'completed': True})
    inputs = {'task': tasks[0], 'actor': {'id': 'teacher', 'kind': 'teacher'}, 'selector': {}, 'host': {'image': 'sha256:pinned'}}
    assert driver.execute('trial', inputs, 'attempt') == {'completed': True}
    assert recorded[0]['prompt_timeout'] == '45m'
    assert recorded[1]['provider_id'] == 'provider' and recorded[1]['model'] == 'wire-model'
    assert recorded[1]['meta']['model'] == 'teacher'
    with pytest.raises(CampaignFault, match='trial_task_changed'):
        driver.validate_task({**tasks[0], 'prompt': 'Unregistered prompt'})
    journal.close()
