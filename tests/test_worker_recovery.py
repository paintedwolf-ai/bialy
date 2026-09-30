"""Worker inputs come from bound job fields, not rendered instructions."""
import importlib.util
import sqlite3
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location('worker_recovery', Path(__file__).parents[1] / 'train-host/recover_worker_requests.py')
recovery = importlib.util.module_from_spec(spec)
spec.loader.exec_module(recovery)


def archive(path, brief='Build a container', leg='', child='child'):
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as conn:
        conn.executescript('''
            CREATE TABLE messages(id, worker_job_id, session_id, role);
            CREATE TABLE worker_jobs(id, child_session_id, parent_session_id, prompt, brief, leg_id);
        ''')
        conn.execute('INSERT INTO messages VALUES (?, ?, ?, ?)', ('opening', 'job', child, 'user'))
        conn.execute('INSERT INTO worker_jobs VALUES (?, ?, ?, ?, ?, ?)',
                     ('job', child, 'parent', 'Rendered assignment or leg prompt', brief, leg))


def row():
    return {'host': 'worker', 'partial': False, 'opening_message_id': 'opening',
            'session': 'child', 'root_session': 'parent', 'state': {'user': 'preamble'}}


def test_recovery_uses_brief_for_direct_jobs_and_prompt_for_legs(tmp_path):
    path = tmp_path / 'store.db'
    archive(path)
    original = row()
    fixed, report = recovery.recover([original], tmp_path)
    assert fixed[0]['state']['user'] == 'Build a container'
    assert original['state']['user'] == 'preamble'
    assert report['recovered'] == 1
    with sqlite3.connect(path) as conn:
        conn.execute("UPDATE worker_jobs SET leg_id = 'leg'")
    fixed, _ = recovery.recover([original], tmp_path)
    assert fixed[0]['state']['user'] == 'Rendered assignment or leg prompt'


def test_recovery_refuses_wrong_child_wal_and_conflicting_archives(tmp_path):
    path = tmp_path / 'first/store.db'
    archive(path, child='other')
    assert recovery.recover([row()], tmp_path)[0] == []
    archive(tmp_path / 'second/store.db')
    wal = tmp_path / 'second/store.db-wal'
    wal.touch()
    fixed, report = recovery.recover([row()], tmp_path)
    assert fixed == []
    assert report['errors'][0]['error'].startswith('WAL present')
    wal.unlink()
    archive(tmp_path / 'third/store.db', brief='Different assignment')
    with pytest.raises(ValueError, match='conflicting'):
        recovery.recover([row()], tmp_path)


merge_spec = importlib.util.spec_from_file_location('worker_merge', Path(__file__).parents[1] / 'train-host/merge_worker_requests.py')
worker_merge = importlib.util.module_from_spec(merge_spec)
merge_spec.loader.exec_module(worker_merge)


def test_merge_preserves_split_and_refuses_unrelated_label_changes():
    import copy

    original = row() | {'receipt': 1, 'meta': {'split': 'historical-value'},
                        'labels': {'requests': [], 'guides': {'guide': None}, 'skill_scores': {'x': 0}}}
    fixed = copy.deepcopy(original)
    fixed['state']['user'] = 'actual task'
    fixed['meta'] = {'split': 'wrong', 'worker_request_recovery': {'job': 'job'}}
    fixed['labels']['skill_scores']['x'] = 4
    fixed['judge'] = {'model': 'judge'}
    result = worker_merge.merge(original, fixed)
    assert result['state']['user'] == 'actual task'
    assert result['meta']['split'] == 'historical-value'
    assert result['labels']['guides'] == original['labels']['guides']
    assert original['labels']['skill_scores']['x'] == 0
    fixed['labels']['requests'] = [{'need': 'changed'}]
    with pytest.raises(ValueError, match='unrelated labels'):
        worker_merge.merge(original, fixed)
