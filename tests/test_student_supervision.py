import json

import pytest

from bialy.student_host import jobs, supervision
from bialy.student_host.resources import write


def test_duplicate_worker_does_not_overwrite_active_execution(tmp_path, monkeypatch):
    state = {'status': 'running', 'attempts': 1}
    write(tmp_path / 'execution.json', state)
    def busy(directory):
        raise jobs.JobBusy('owned')
    monkeypatch.setattr(jobs, 'train', busy)
    with pytest.raises(jobs.JobBusy):
        supervision.train(tmp_path)
    assert json.loads((tmp_path / 'execution.json').read_text()) == state


def test_worker_failure_publishes_a_terminal_outcome(tmp_path, monkeypatch):
    def failed(directory):
        raise jobs.JobError('invalid artifact')
    monkeypatch.setattr(jobs, 'train', failed)
    with pytest.raises(jobs.JobError):
        supervision.train(tmp_path)
    state = json.loads((tmp_path / 'execution.json').read_text())
    assert state['status'] == 'failed'
    assert state['failure']['type'] == 'JobError'
