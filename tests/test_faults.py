import dataclasses
import json
import sqlite3
from types import SimpleNamespace

import pytest

from bialy import faults, fleet


def store(path):
    with sqlite3.connect(path) as db:
        db.executescript("""
        CREATE TABLE turns(status TEXT);
        INSERT INTO turns VALUES('failed'),('complete');
        CREATE TABLE prompt_submissions(status TEXT,error_code TEXT,completed_at TEXT);
        INSERT INTO prompt_submissions VALUES('failed','provider_request_rejected','2026-10-22T00:00:00Z');
        CREATE TABLE code_scans(status TEXT);
        INSERT INTO code_scans VALUES('complete');
        CREATE TABLE turn_load_receipts(abstained INT,reason TEXT);
        INSERT INTO turn_load_receipts VALUES(1,'engine unavailable');
        CREATE TABLE llm_calls(estimated_nano_usd INT);
        INSERT INTO llm_calls VALUES(1000000000);
        """)


def test_live_store_statuses_and_first_code_are_read_without_error_prose(tmp_path):
    store(tmp_path / "store.db")
    report = faults.snapshot(tmp_path)
    assert report["failed_turns"] == 1 and report["turns"] == 2
    assert report["codes"] == {"provider_request_rejected": {"count": 1, "first_at": "2026-10-22T00:00:00Z"}}
    assert report["scanner_completed"] == 1 and report["engine_unavailable"] == 1
    assert report["spend_usd"] == 1


def test_live_docker_query_is_one_readonly_transaction(tmp_path, monkeypatch):
    path = tmp_path / "store.db"
    store(path)
    def execute(args, **kw):
        assert args[:5] == ["docker", "exec", "owned", "sqlite3", "-readonly"]
        assert "-cmd" in args and kw["timeout"] == 10
        statements = args[-1].split(";")
        with sqlite3.connect(path) as db:
            db.execute(statements[0])
            result = db.execute(statements[1]).fetchone()[0]
        return SimpleNamespace(returncode=0, stdout=result)
    monkeypatch.setattr(faults.subprocess, "run", execute)
    report = faults.snapshot(tmp_path, "owned")
    assert report["failed_turns"] == 1
    assert report["codes"]["provider_request_rejected"]["count"] == 1


def test_capture_triage_distinguishes_payloads_commits_exclusions_and_summary_protocol(tmp_path):
    captures = tmp_path / "training-captures"
    captures.mkdir()
    records = [
        {"schema": "pw-training-turn/1", "session_id": "s", "message_id": "m", "settled": True,
         "student_request": {"model": "student"}, "response": {"content": "ok"}},
        {"schema": "pw-training-fact/1", "kind": "turn_committed", "session_id": "s", "message_id": "m"},
        {"schema": "pw-training-turn/1", "exclusion_reason": "host_modified_response", "modified_fields": ["content"]},
        {"schema": "pw-training-health/1", "purpose": "compaction", "success": False, "finish_reason": "length"},
        {"schema": "pw-training-health/1", "purpose": "compaction", "success": True},
    ]
    for n, record in enumerate(records):
        directory = captures / "health" if record["schema"] == "pw-training-health/1" else captures
        directory.mkdir(exist_ok=True)
        (directory / ("%s.json" % n)).write_text(json.dumps(record))
    report = faults.capture_snapshot(tmp_path)
    assert report["captures"] == 2 and report["payloads_kept"] == report["committed_payloads"] == 1
    assert report["exclusion_reasons"] == {"host_modified_response": 1}
    assert report["modified_fields"] == {"content": 1}
    assert report["summary_calls"] == 2 and report["summary_length"] == report["summary_completed"] == 1


@pytest.mark.parametrize(("report", "settings", "code"), [
    ({"failed_turns": 1}, {}, "turn_failed"),
    ({"scanner_failures": 1}, {}, "scanner_failed"),
    ({"engine_unavailable": 1}, {"require_engine": True}, "engine_unavailable"),
    ({"engine_unavailable": 1}, {}, None),
    ({"summary_length": 2}, {}, "summary_truncated"),
    ({"summary_length": 1}, {}, None),
    ({"turns": 3}, {"require_captures": True}, "no_kept_captures"),
    ({"turns": 2}, {"require_captures": True}, None),
    ({"turns": 3, "payloads_kept": 1}, {"require_captures": True}, None),
    ({"task_fault": {"code": "driver_exit"}}, {}, "driver_exit"),
])
def test_fail_fast_uses_protocol_facts(report, settings, code):
    assert faults.reason(report, settings) == code


def test_first_failed_turn_stops_only_this_runs_active_containers_and_retains_evidence(factory, tmp_path, monkeypatch):
    factory = dataclasses.replace(factory, root=tmp_path)
    directory = tmp_path / "fleet-on"
    for name in ("a", "b", "c"):
        shard = directory / "shards" / name
        shard.mkdir(parents=True)
        (shard / "workspace.json").write_text('{"prompt_timeout":"45m"}')
        (shard / "tasks.jsonl").write_text('{"id":"one"}\n')
    started, stopped, exported = [], [], []
    class Container:
        def __init__(self, args, **kw):
            self.name = args[0]
            started.append(self.name)
            (directory / "shards" / self.name / "ready.json").write_text('{}')
        def poll(self):
            return None
        def wait(self):
            return 0
    def command(*args, **kw):
        if args[:2] == ("docker", "stop"):
            stopped.append(args[-1])
        return SimpleNamespace(returncode=0, stdout="")
    monkeypatch.setattr(fleet, "running_containers", lambda _: {})
    monkeypatch.setattr(fleet, "container_args", lambda f, r, s, m, i: (s.name, None, [s.name]))
    monkeypatch.setattr(fleet.subprocess, "Popen", Container)
    monkeypatch.setattr(fleet.time, "sleep", lambda _: None)
    monkeypatch.setattr(fleet, "sh", command)
    monkeypatch.setattr(fleet, "release_work", lambda _: None)
    monkeypatch.setattr(fleet, "export_live", lambda name, *_: exported.append(name))
    monkeypatch.setattr(faults, "snapshot", lambda shard, container: {"shard": shard.name, "failed_turns": 1})
    with pytest.raises(faults.RunFault, match="turn_failed"):
        fleet.run(factory, directory, runners=2)
    assert started == stopped == exported == ["a", "b"]
    assert json.loads((directory / "fault.json").read_text())["code"] == "turn_failed"
    with pytest.raises(faults.RunFault, match="turn_failed"):
        fleet.run(factory, directory, runners=2)
    assert started == ["a", "b"]


def test_run_timeout_overrides_archetype_budget_and_limits_whole_task(factory, tmp_path):
    factory = dataclasses.replace(factory, run=dict(factory.run, task_timeout="45m", tasks_per_runner=1))
    (tmp_path / "tasks").mkdir()
    task = {"id": "one", "prompt": "p", "model": "qwen3.6-35b-a3b", "follow_ups": [{"prompt": "again"}], "meta": {"prompt_timeout": "300m"}}
    (tmp_path / "tasks/flask.jsonl").write_text((json.dumps(task) + "\n") * 2)
    assert fleet.plan_shards(factory, tmp_path / "tasks", tmp_path / "run", {}) == 2
    for shard in (tmp_path / "run/shards").iterdir():
        assert json.loads((shard / "workspace.json").read_text())["task_timeout"] == "45m"
        assert fleet.shard_deadline(factory, shard, 10) == 10 + 45 * 60 + 120


def test_triage_keeps_same_named_shards_in_different_passes_separate(tmp_path, monkeypatch):
    off = tmp_path / "fleet-off/shards/s"
    on = tmp_path / "fleet-on/shards/s"
    off.mkdir(parents=True)
    on.mkdir(parents=True)
    monkeypatch.setattr(faults, "snapshot", lambda shard, container: (str(shard), container))
    assert faults.run_snapshot(tmp_path, {str(on): "active-on"}) == [(str(off), None), (str(on), "active-on")]
