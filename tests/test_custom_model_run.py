import dataclasses
import subprocess

import pytest

from bialy import custom_model_run as cm
from bialy import run as runmod
from bialy.student_host.specification import students


@pytest.fixture
def custom_run(factory, tmp_path, monkeypatch):
    f = dataclasses.replace(factory, root=tmp_path)
    settings = runmod.settings_for(f, {"lycaon_checkout": str(tmp_path / "host"), "engine_on.enabled": False, "custom_model.enabled": True,
                                      "custom_model.epoch": "legacy-review-1",
                                      "custom_model.host_commit": "a" * 40})
    run = runmod.Run(factory=f, name="custom", settings=settings)
    monkeypatch.setattr(cm.provenance, "factory_source", lambda: {"tree_sha256": "b" * 64})
    monkeypatch.setattr(cm.subprocess, "run", lambda args, **kw: subprocess.CompletedProcess(
        args, 0, "a" * 40 + "\n" if "rev-parse" in args else "", ""))
    monkeypatch.setattr(cm.host_bundle, "contract", lambda image, destination, commit: ({}, "c" * 64))
    collector = run.checkout / "scripts/bialy/custom_model/collect.py"
    collector.parent.mkdir(parents=True)
    collector.write_text("# fixture collector")
    return run


def test_the_binding_names_the_student_and_its_serving_is_validated(custom_run):
    identity = cm.binding(custom_run)
    assert set(identity['students']) == set(students())
    assert all(item['context_window'] == 65536 for item in identity['students'].values())
    cfg = dict(custom_run.settings['custom_model'])
    for bad in ({'kind': 'ollama'}, {'context_window': 32768}, {'model': ''}):
        cfg['students'] = {**identity['students'], 'e4b': dict(identity['students']['e4b'], **bad)}
        with pytest.raises(cm.DatasetError, match='students'):
            cm.validate_settings(cfg)
    cfg['students'] = {'e4b': identity['students']['e4b']}
    with pytest.raises(cm.DatasetError, match='students'):
        cm.validate_settings(cfg)


def test_a_host_without_the_collector_is_refused_before_any_stage(custom_run):
    (custom_run.checkout / "scripts/bialy/custom_model/collect.py").unlink()
    called = []
    with pytest.raises(runmod.RunError, match="pinned host lacks"):
        runmod.execute(custom_run, funcs={"check": lambda run: called.append(True)}, until="check")
    assert not called and not custom_run.state_path.exists()


def test_collection_epoch_is_explicit_and_cannot_change_on_resume(custom_run, monkeypatch):
    cfg = custom_run.settings["custom_model"]
    cfg["epoch"] = None
    with pytest.raises(cm.DatasetError, match="epoch must name"):
        cm.binding(custom_run)
    cfg["epoch"] = "legacy-review-1"
    first = cm.binding(custom_run)
    assert cm.binding(custom_run) == first
    monkeypatch.setattr(cm.host_bundle, "contract", lambda image, destination, commit: ({}, "d" * 64))
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









def test_the_binding_names_the_hosts_contract(custom_run):
    identity = cm.binding(custom_run)
    assert identity["epoch"] == "legacy-review-1" and identity["contract_sha256"] == "c" * 64




def test_a_host_image_that_cannot_state_its_contract_blocks_the_binding(custom_run, monkeypatch):
    def refuse(image, destination, commit):
        raise cm.host_bundle.HostEvidenceError("host_contract_revision_mismatch")
    monkeypatch.setattr(cm.host_bundle, "contract", refuse)
    with pytest.raises(cm.DatasetError, match="host_contract_revision_mismatch"):
        cm.binding(custom_run)
