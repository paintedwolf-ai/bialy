import json
from pathlib import Path

import pytest

from bialy import config
from bialy import run as runmod


def make_run(factory, tmp_path, **settings):
    import dataclasses
    f = dataclasses.replace(factory, root=tmp_path)
    return runmod.Run(factory=f, name="t", settings=runmod.settings_for(f, settings))


def stub_funcs(record, fail_at=None, skip=()):
    def make(name):
        def stage(run):
            record.append(name)
            if name == fail_at:
                raise RuntimeError("boom")
            if name in skip:
                return {"skipped": "not here"}
            return {"did": name}
        return stage
    return {name: make(name) for name in runmod.STAGES}


def test_run_settings_defaults_and_unknown_keys():
    settings = config.run_settings({"task_cap": 2, "train": {"device": "cpu"}, "engine_on": False})
    assert settings["task_cap"] == 2 and settings["train"]["device"] == "cpu"
    assert settings["train"]["recipes"] == ["B5", "B7G", "E4", "code-rank"]
    assert settings["engine_on"] == {"enabled": False, "deadline_ms": 60000}
    assert config.run_settings({"engine_on": {"deadline_ms": 5}})["engine_on"] == {"enabled": True, "deadline_ms": 5}
    assert settings["lycaon_checkout"].startswith("/")
    for bad in ({"bogus": 1}, {"train": {"bogus": 1}}, {"engine_on": {"bogus": 1}}, {"skillreq": {"bogus": 1}}):
        with pytest.raises(config.ConfigError):
            config.run_settings(bad)


def test_scanner_setting_is_pinned_or_none(factory, tmp_path):
    assert make_run(factory, tmp_path).scanner is True
    assert make_run(factory, tmp_path, scanner="none").scanner is False
    with pytest.raises(runmod.RunError, match="run.scanner"):
        runmod.run_scanner("maybe")
    with pytest.raises(runmod.RunError, match="scanner_candidate"):
        runmod.run_scanner("candidate")
    run = make_run(factory, tmp_path, scanner="candidate", scanner_candidate=str(tmp_path / "artifact"))
    assert run.scanner is True and run.scanner_candidate == str(tmp_path / "artifact")


def test_flag_overrides_reach_nested_settings(factory):
    settings = runmod.settings_for(factory, {"train.epochs": 2, "engine_on.enabled": False, "runners": None})
    assert settings["train"]["epochs"] == 2 and settings["engine_on"]["enabled"] is False
    assert settings["runners"] == factory.run["runners"]


def test_every_stage_has_a_function_and_the_engine_on_stages_are_between_the_passes():
    assert set(runmod.STAGE_FUNCS) == set(runmod.STAGES)
    order = [runmod.STAGES.index(s) for s in runmod.ENGINE_ON_STAGES]
    assert order == sorted(order) and runmod.STAGES.index("judge") < order[0] < runmod.STAGES.index("split")


def test_execute_records_every_stage_and_resumes(factory, tmp_path):
    run = make_run(factory, tmp_path)
    record = []
    state = runmod.execute(run, funcs=stub_funcs(record))
    assert record == list(runmod.STAGES)
    assert all(state["stages"][name]["status"] == "done" for name in runmod.STAGES)
    assert json.loads(run.state_path.read_text())["stages"]["tasks"]["outputs"] == {"did": "tasks"}
    assert runmod.finished(run)
    record.clear()
    runmod.execute(run, funcs=stub_funcs(record))
    assert record == []
    runmod.execute(run, start="judge", funcs=stub_funcs(record))
    assert record == list(runmod.STAGES[runmod.STAGES.index("judge"):])


def test_execute_stops_at_a_failure_and_after_until(factory, tmp_path):
    run = make_run(factory, tmp_path)
    record = []
    with pytest.raises(runmod.RunError, match="stage judge failed: boom"):
        runmod.execute(run, funcs=stub_funcs(record, fail_at="judge"))
    state = run.load_state()
    assert state["stages"]["judge"]["status"] == "failed" and "split" not in state["stages"]
    assert "boom" in run.path("REPORT.md").read_text()
    assert not runmod.finished(run)
    record.clear()
    runmod.execute(run, until="split_pilot", funcs=stub_funcs(record))
    assert record == ["judge", "judge_skillreq", "split_pilot"]
    assert [a for _, a in runmod.plan(run, until="split_pilot")] == ["skip"] * (runmod.STAGES.index("split_pilot") + 1)


def test_redo_reruns_one_stage_and_keeps_the_rest(factory, tmp_path):
    run = make_run(factory, tmp_path)
    record = []
    runmod.execute(run, funcs=stub_funcs(record))
    record.clear()
    runmod.execute(run, funcs=stub_funcs(record), redo=["image"])
    assert record == ["image"] and runmod.finished(run)


def test_engine_on_stages_skip_when_the_second_pass_is_off(factory, tmp_path):
    run = make_run(factory, tmp_path, engine_on={"enabled": False, "deadline_ms": 1})
    for name in runmod.ENGINE_ON_STAGES:
        if name != "pilot":
            assert runmod.STAGE_FUNCS[name](run) == {"skipped": "run.engine_on is off"}
    assert runmod.stage_pilot(run)["skipped"]


def test_report_lists_publish_commands_without_pushing(factory, tmp_path):
    run = make_run(factory, tmp_path)
    funcs = stub_funcs([])
    funcs["release"] = lambda r: {"release": str(r.path("dist", "dataset-x")), "version": "x"}
    funcs["release_heads"] = lambda r: {"release": str(r.path("dist", "heads-x")), "version": "x"}
    funcs["report"] = runmod.stage_report
    run.path("usage.jsonl").parent.mkdir(parents=True)
    run.path("usage.jsonl").write_text(json.dumps({"model": "accounts/fireworks/models/glm-5p3-flash", "usage": {"prompt_tokens": 10, "completion_tokens": 5}}) + "\n")
    state = runmod.execute(run, funcs=funcs)
    text = run.path("REPORT.md").read_text()
    assert "bialy publish dataset --release" in text and "bialy publish heads --release" in text and "--push" in text
    assert "| accounts/fireworks/models/glm-5p3-flash | 1 | 10 | 5 |" in text
    assert "sidecars" in text
    assert state["stages"]["report"]["outputs"]["published"] == {}


def test_spend_ceiling_stops_the_run(factory, tmp_path):
    import dataclasses
    priced = [dataclasses.replace(m, hosted=dict(m.hosted, price_per_million={"input": 1000000, "output": 0})) if m.hosted else m for m in factory.models]
    f = dataclasses.replace(factory, root=tmp_path, models=priced)
    run = runmod.Run(factory=f, name="t", settings=runmod.settings_for(f, {"spend_ceiling_usd": 1}))
    run.path("usage.jsonl").parent.mkdir(parents=True)
    run.path("usage.jsonl").write_text(json.dumps({"model": "accounts/fireworks/models/glm-5p3-flash", "usage": {"prompt_tokens": 5, "completion_tokens": 0}}) + "\n")
    with pytest.raises(runmod.RunError, match="exceeds run.spend_ceiling_usd"):
        runmod.execute(run, funcs=stub_funcs([]))


def test_check_refuses_a_ceiling_without_prices_and_a_missing_key(factory, tmp_path, monkeypatch):
    monkeypatch.delenv("FIREWORKS_API_KEY", raising=False)
    run = make_run(factory, tmp_path, spend_ceiling_usd=5, generators=["glm-5.3-flash"], lycaon_checkout=str(tmp_path))
    with pytest.raises(runmod.RunError) as exc:
        runmod.stage_check(run)
    message = str(exc.value)
    assert "needs FIREWORKS_API_KEY" in message and "price_per_million" in message and "lacks lycaon/cmd/lycaon" in message


def test_scaled_factory_caps_tasks_and_narrows_generators(factory, tmp_path):
    run = make_run(factory, tmp_path, task_cap=1, generators=["glm-5.3-flash"], repos=["flask"])
    scaled = runmod.scaled_factory(run)
    assert all(a.per_repo == 1 for a in scaled.archetypes)
    assert [m.id for m in scaled.generators()] == ["glm-5.3-flash"] and [r.name for r in scaled.repos] == ["flask"]
    assert scaled.judge_for("glm-5.3-flash").family != "glm"
    assert runmod.writer_model(run).id in {m.id for m in factory.models if "writer" in m.roles}
    # A model named as neither generator nor judge keeps its other roles.
    assert "writer" in scaled.model("glm-5.3-flash").roles and "qwen3.6-35b-a3b" not in {m.id for m in scaled.models}
    run = make_run(factory, tmp_path, generators=["glm-5.3-flash"], judges=["inkling"])
    scaled = runmod.scaled_factory(run)
    assert [m.id for m in scaled.models if "judge" in m.roles] == ["inkling"]


def test_an_early_stop_still_leaves_a_current_report(factory, tmp_path):
    run = make_run(factory, tmp_path)
    funcs = stub_funcs([])
    funcs["release"] = lambda r: {"release": str(r.path("dist", "dataset-x")), "version": "x"}
    run.push = True
    runmod.execute(run, until="release", funcs=funcs)
    text = run.path("REPORT.md").read_text()
    assert "| release | done |" in text and "bialy publish dataset" in text and "pushed" not in text


def test_run_paths_follow_the_host_platform(factory, tmp_path, monkeypatch):
    from bialy import build
    run = make_run(factory, tmp_path)
    monkeypatch.setattr(build, "host_is_runner_platform", lambda: True)
    assert run.host_bin_dir == run.bin_dir
    monkeypatch.setattr(build, "host_is_runner_platform", lambda: False)
    assert run.host_bin_dir == run.path("build", "host-bin")
    assert run.image("on") == "%s:t-on" % factory.fleet["image"]


def test_an_engine_on_pass_without_the_engine_is_refused(factory, tmp_path, monkeypatch):
    run = make_run(factory, tmp_path)
    rows = run.path("fleet-on", "rows.jsonl")
    rows.parent.mkdir(parents=True)
    rows.write_text(json.dumps({"engine": {"state": "abstained", "reason": "engine unavailable"}}) + "\n")
    monkeypatch.setattr(runmod.fleet, "collect", lambda run_dir, pass_name, tasks_dir: (rows, {"rows": 1}))
    with pytest.raises(runmod.RunError, match="engine was unavailable"):
        runmod.collect_pass(run, "on")
    rows.write_text(json.dumps({"engine": {"state": "decided"}}) + "\n" + json.dumps({"engine": {"state": "abstained", "reason": "deadline"}}) + "\n")
    assert runmod.collect_pass(run, "on")["stats"]["engine_states"] == {"decided": 1, "abstained/deadline": 1}


def test_a_stale_judge_manifest_is_cleared_and_a_matching_one_resumes(factory, tmp_path, monkeypatch):
    run = make_run(factory, tmp_path)
    run.dir.mkdir(parents=True)
    run.path("corpus.json").write_text(json.dumps({"catalog_revision": "rev", "skills": [], "tools": []}))
    rows = run.path("rows.jsonl")
    rows.write_text(json.dumps({"session": "s", "receipt": 1}) + "\n")
    out = run.path("judged.jsonl")
    Path(str(out) + ".manifest.json").write_text(json.dumps({"signature": {"inputs_sha256": "old"}}))
    Path(str(out) + ".partial.jsonl").write_text("stale")
    seen = {}

    def fake_run(scaled, c, rows_in, rows_out, workers, units):
        seen["partial_exists"] = Path(str(rows_out) + ".partial.jsonl").exists()
        Path(rows_out).write_text("")
        return {"rows": 1}
    monkeypatch.setattr(runmod.judge, "run", fake_run)
    monkeypatch.setattr(runmod.provenance, "record", lambda *a, **k: {})
    runmod.judge_rows(run, rows, out)
    assert seen["partial_exists"] is False
    scaled = runmod.scaled_factory(run)
    sig = runmod.judge.signature(scaled, json.loads(run.path("corpus.json").read_text()), str(rows), runmod.judge.UNITS, scaled.judge["second_judge_fraction"])
    Path(str(out) + ".manifest.json").write_text(json.dumps({"signature": sig}))
    Path(str(out) + ".partial.jsonl").write_text("resume")
    runmod.judge_rows(run, rows, out)
    assert seen["partial_exists"] is True
