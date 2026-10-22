import dataclasses
import json
import subprocess

import pytest

from bialy import custom_model_run as cm
from bialy import provenance
from bialy import run as runmod


@pytest.fixture
def custom_run(factory, tmp_path, monkeypatch):
    f = dataclasses.replace(factory, root=tmp_path)
    settings = runmod.settings_for(f, {"lycaon_checkout": str(tmp_path / "host"), "engine_on.enabled": False, "custom_model.enabled": True,
                                      "custom_model.diagnostic_only": True, "custom_model.epoch": "legacy-review-1",
                                      "custom_model.host_commit": "a" * 40})
    run = runmod.Run(factory=f, name="custom", settings=settings)
    monkeypatch.setattr(cm.provenance, "factory_source", lambda: {"tree_sha256": "b" * 64})
    monkeypatch.setattr(cm.subprocess, "run", lambda args, **kw: subprocess.CompletedProcess(
        args, 0, "a" * 40 + "\n" if "rev-parse" in args else "", ""))
    return run


def source_inputs(run):
    identity = cm.binding(run)
    for name, value in (("corpus.json", {"catalog_revision": "catalog"}), ("build/bin/BUILD.json", {"lycaon_commit": "a" * 40})):
        cm.write_json(run.path(name), value)
    for name in ("lycaon", "lycaon-debug"):
        run.bin_dir.joinpath(name).write_bytes(name.encode())
    for name, value in (("fleet-off/shards/a/store.db", "store"), ("fleet-off/shards/a/manifest.jsonl", "{}\n"),
                        ("tasks/tasks.jsonl", '{"id":"task"}\n'), ("judged-off.jsonl", "{}\n")):
        path = run.path(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value)
    cm.write_json(run.path("fleet-off/provenance.json"), {"engine": provenance.engine(run.factory, run.bin_dir)})
    run.save_state({"stages": {name: {"status": "done"} for name in ("drive", "collect", "judge")}})
    return identity, cm.input_manifest(run, identity)


def test_the_binding_names_the_student_and_its_serving_is_validated(custom_run):
    identity = cm.binding(custom_run)
    assert (identity["student_provider"], identity["student_kind"], identity["student_model"], identity["student_context_window"]) == (
        "student", "openai-compatible", "gemma-4-e4b-it", 32768)
    cfg = dict(custom_run.settings["custom_model"])
    for bad in ({"kind": "ollama"}, {"context_window": 2048}, {"model": ""}):
        cfg["student_serving"] = dict(custom_run.settings["custom_model"]["student_serving"], **bad)
        with pytest.raises(cm.DatasetError, match="student_serving"):
            cm.validate_settings(cfg)
    cfg["student_serving"] = {"provider": "student"}
    with pytest.raises(cm.DatasetError, match="student_serving"):
        cm.validate_settings(cfg)


def test_training_run_is_blocked_before_any_stages(custom_run):
    custom_run.settings["custom_model"]["diagnostic_only"] = False
    called = []
    with pytest.raises(runmod.RunError, match="pinned host lacks"):
        runmod.execute(custom_run, funcs={"check": lambda run: called.append(True)}, until="check")
    assert not called and not custom_run.state_path.exists()


def test_collection_epoch_is_explicit_and_cannot_change_on_resume(custom_run):
    cfg = custom_run.settings["custom_model"]
    cfg["epoch"] = None
    with pytest.raises(cm.DatasetError, match="epoch must name"):
        cm.binding(custom_run)
    cfg["epoch"] = "legacy-review-1"
    first = cm.binding(custom_run)
    assert cm.binding(custom_run) == first
    cfg["tool_surface"] = "command-v1"
    with pytest.raises(cm.DatasetError, match="new run name and epoch"):
        cm.binding(custom_run)


def test_old_unbound_runs_are_not_adopted(custom_run):
    custom_run.save_state({"stages": {"build": {"status": "done"}}})
    with pytest.raises(cm.DatasetError, match="recollect in a new pinned run"):
        cm.binding(custom_run)


def test_dirty_or_wrong_host_checkout_is_refused(custom_run, monkeypatch):
    monkeypatch.setattr(cm.subprocess, "run", lambda args, **kw: subprocess.CompletedProcess(args, 0, "b" * 40, ""))
    with pytest.raises(cm.DatasetError, match="does not match"):
        cm.binding(custom_run)
    monkeypatch.setattr(cm.subprocess, "run", lambda args, **kw: subprocess.CompletedProcess(
        args, 0, "a" * 40 if "rev-parse" in args else " M source.go", ""))
    with pytest.raises(cm.DatasetError, match="clean pinned"):
        cm.binding(custom_run)


def test_input_identity_includes_wal_and_refuses_mismatched_fleet(custom_run):
    identity, before = source_inputs(custom_run)
    custom_run.path("fleet-off/shards/a/store.db-wal").write_bytes(b"new rows")
    after = cm.input_manifest(custom_run, identity)
    assert cm.digest(before) != cm.digest(after)
    assert "fleet-off/shards/a/store.db-wal" in after["files"]
    custom_run.bin_dir.joinpath("lycaon").write_bytes(b"wrong binary")
    with pytest.raises(cm.DatasetError, match="fleet provenance"):
        cm.input_manifest(custom_run, identity)


def test_only_validated_rows_bound_to_this_epoch_reach_qualified_review(custom_run, tmp_path, monkeypatch):
    identity, inputs = source_inputs(custom_run)
    capture = {"epoch": identity["epoch"], "tool_surface": identity["tool_surface"], "host_revision": identity["host_commit"],
               "catalog_sha256": inputs["catalog_sha256"]}
    rows = [{"id": "candidate"}, {"id": "foreign", "request_capture": dict(capture, epoch="future")},
            {"id": "validated", "request_capture": capture}]
    path = tmp_path / "candidates.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    monkeypatch.setattr(cm.dpo, "training_eligible", lambda row: row["id"] != "candidate", raising=False)
    out = tmp_path / "qualified-review.jsonl"
    counts = cm.qualify_inputs([path], out, identity, inputs)
    assert counts == {"candidates": 3, "unqualified": 1, "foreign_epoch": 1, "qualified": 1}
    assert [json.loads(line)["id"] for line in out.read_text().splitlines()] == ["validated"]


def test_reextract_uses_changed_store_and_never_creates_training_splits(custom_run, monkeypatch):
    source_inputs(custom_run)
    exports = []

    def export(passes, tasks, out):
        exports.append(out)
        out.write_bytes(b"diagnostic archive")

    def extract(transcripts, **kw):
        kw["out_sft"].write_text('{"id":"candidate"}\n')
        kw["out_dpo"].write_text("")
        assert kw["drop_retired_command_rejections"] is False
        return {"sft_rows": 1, "dpo_rows": 0}

    monkeypatch.setattr(cm.dpo, "export_transcripts", export)
    monkeypatch.setattr(cm.dpo, "extract", extract)
    monkeypatch.setattr(cm.dpo, "training_eligible", lambda row: False, raising=False)
    first = cm.collect(custom_run)
    second = cm.collect(custom_run)
    assert first["directory"] != second["directory"] and len(exports) == 2
    assert first["status"] == "not_ready" and first["training_consumable"] is False
    assert first["qualification"]["sft"]["unqualified"] == 1
    assert not list(custom_run.path("custom_model").rglob("train.jsonl"))
    custom_run.path("fleet-off/shards/a/store.db").write_bytes(b"new store")
    third = cm.collect(custom_run)
    assert third["directory"] != first["directory"]
    assert json.loads(custom_run.path("custom_model/latest.json").read_text())["directory"] == third["directory"]


def test_not_ready_stops_run_before_release_and_reports(custom_run, monkeypatch):
    monkeypatch.setattr(cm, "binding", lambda run: {})
    called = []
    funcs = {name: (lambda run, name=name: called.append(name) or {}) for name in runmod.STAGES}
    funcs["custom_model"] = lambda run: {"status": "not_ready", "reason": "no qualified rows"}
    state = runmod.execute(custom_run, funcs=funcs)
    assert state["stages"]["custom_model"]["status"] == "not_ready"
    assert "release" not in called
    assert "no qualified rows" in custom_run.path("REPORT.md").read_text()


def test_changed_inputs_during_processing_never_receive_report_or_latest(custom_run, monkeypatch):
    source_inputs(custom_run)
    monkeypatch.setattr(cm.dpo, "export_transcripts", lambda passes, tasks, out: out.write_bytes(b"archive"))

    def extract(transcripts, **kw):
        kw["out_sft"].write_text("")
        kw["out_dpo"].write_text("")
        custom_run.path("judged-off.jsonl").write_text('{"changed":true}\n')
        return {}

    monkeypatch.setattr(cm.dpo, "extract", extract)
    with pytest.raises(cm.DatasetError, match="changed during processing"):
        cm.collect(custom_run)
    assert not custom_run.path("custom_model/latest.json").exists()
    assert not list(custom_run.path("custom_model").rglob("report.json"))


def test_capture_files_change_input_identity(custom_run):
    identity, before = source_inputs(custom_run)
    capture = custom_run.path("fleet-off/shards/a/training-captures/call.json")
    cm.write_json(capture, {"schema": "pw-training-turn/1"})
    after = cm.input_manifest(custom_run, identity)
    assert cm.digest(before) != cm.digest(after)
    assert "fleet-off/shards/a/training-captures/call.json" in after["files"]


def test_non_diagnostic_requires_exporter_capability(custom_run):
    custom_run.settings["custom_model"]["diagnostic_only"] = False
    exporter = custom_run.checkout / "scripts/bialy/custom_model/collect.py"
    exporter.parent.mkdir(parents=True)
    exporter.write_text("# fixture capability")
    identity = cm.binding(custom_run)
    assert identity["epoch"] == "legacy-review-1"
