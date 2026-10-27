import json
from types import SimpleNamespace

import pytest

from bialy.campaign import oracle
from bialy.campaign.journal import CampaignFault, digest
from bialy.provenance import file_sha256


def files(tmp_path):
    project = tmp_path / 'project'
    project.mkdir()
    (project / 'answer.py').write_text('answer = 42\n')
    def run(*args, **kwargs):
        return SimpleNamespace(stdout=b'answer.py\0')
    return project, run


def test_snapshot_restart_preserves_verified_bytes_and_refuses_changed_source(tmp_path):
    project, run = files(tmp_path)
    destination = tmp_path / 'evidence' / 'source'
    inventory = oracle.snapshot(project, destination, run=run)
    (destination.parent / 'source-inventory.json').unlink()
    assert oracle.snapshot(project, destination, run=run) == inventory
    (project / 'answer.py').write_text('answer = 41\n')
    with pytest.raises(CampaignFault, match='snapshot_changed'):
        oracle.snapshot(project, destination, run=run)
    assert (destination / 'answer.py').read_text() == 'answer = 42\n'


def test_snapshot_rejects_symlinks_and_parent_traversal(tmp_path):
    project, run = files(tmp_path)
    (project / 'link.py').symlink_to(project / 'answer.py')
    for index, name in enumerate(('link.py', '../outside', '/absolute')):
        with pytest.raises(CampaignFault, match='artifact_'):
            oracle.snapshot(project, tmp_path / str(index), [name], run)


def test_oracle_keeps_failed_checks_bound_to_source_and_pinned_grader(tmp_path, monkeypatch):
    project, run = files(tmp_path)
    source = tmp_path / 'evidence' / 'source'
    inventory = oracle.snapshot(project, source, run=run)
    bank = tmp_path / 'bank'
    bank.mkdir()
    (bank / 'check.py').write_text('assert False\n')
    task = {'oracle_files': {'check.py': file_sha256(bank / 'check.py')},
            'checks': [{'id': 'outcome', 'argv': ['python', '/oracle/check.py']}],
            'required_checks': ['outcome', 'host-closeout']}
    calls = []
    class Container:
        def __init__(self, *args, **kwargs):
            pass
        def execute(self, name, arguments, binding):
            calls.append((arguments, binding))
            return {'exit_status': 1, 'oom_killed': False}
        def release(self):
            calls.append('released')
    monkeypatch.setattr(oracle, 'Container', Container)
    def process(argv, **kwargs):
        kwargs['stdout'].write(b'AssertionError\n')
    destination = tmp_path / 'grading'
    result = oracle.grade(task, source, inventory, bank, source.parent, 'sha256:pinned', destination, process)
    assert result['checks'][0]['passed'] is False
    assert result['checks'][0]['source_sha256'] == digest(inventory)
    assert json.loads((destination / 'oracle.json').read_text()) == result
    assert calls[-1] == 'released'
    assert '--network' in calls[0][0] and 'none' in calls[0][0]
    (bank / 'check.py').write_text('assert True\n')
    with pytest.raises(CampaignFault, match='oracle_changed'):
        oracle.grade(task, source, inventory, bank, source.parent, 'sha256:pinned', destination, process)
