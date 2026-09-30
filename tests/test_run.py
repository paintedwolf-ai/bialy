import json

import pytest

from bialy import config, run as runmod


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
    settings = config.run_settings({"task_cap": 2, "train": {"device": "cpu"}})
    assert settings["task_cap"] == 2 and settings["train"]["device"] == "cpu" and settings["train"]["recipes"] == ["B5", "B7G"]
    assert settings["lycaon_bin"].endswith("/bin")
    with pytest.raises(config.ConfigError):
        config.run_settings({"bogus": 1})
    with pytest.raises(config.ConfigError):
        config.run_settings({"train": {"bogus": 1}})


def test_execute_records_every_stage_and_resumes(factory, tmp_path):
    run = make_run(factory, tmp_path)
    record = []
    state = runmod.execute(run, funcs=stub_funcs(record))
    assert record == list(runmod.STAGES)
    assert all(state["stages"][name]["status"] == "done" for name in runmod.STAGES)
    assert json.loads(run.state_path.read_text())["stages"]["tasks"]["outputs"] == {"did": "tasks"}
    # A second invocation runs nothing; --from reruns from a stage on.
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
    # After the fix, the run resumes at the failed stage.
    record.clear()
    runmod.execute(run, until="split", funcs=stub_funcs(record))
    assert record == ["judge", "split"]
    assert [a for _, a in runmod.plan(run, until="split")] == ["skip"] * 11


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


def test_scaled_factory_caps_tasks_and_narrows_generators(factory, tmp_path):
    run = make_run(factory, tmp_path, task_cap=1, generators=["glm-5.3-flash"], repos=["flask"])
    scaled = runmod.scaled_factory(run)
    assert all(a.per_repo == 1 for a in scaled.archetypes)
    assert [m.id for m in scaled.generators()] == ["glm-5.3-flash"] and [r.name for r in scaled.repos] == ["flask"]
    assert scaled.judge_for("glm-5.3-flash").family != "glm"


def test_an_early_stop_still_leaves_a_current_report(factory, tmp_path):
    run = make_run(factory, tmp_path)
    funcs = stub_funcs([])
    funcs["release"] = lambda r: {"release": str(r.path("dist", "dataset-x")), "version": "x"}
    run.push = True
    runmod.execute(run, until="release", funcs=funcs)
    text = run.path("REPORT.md").read_text()
    assert "| release | done |" in text and "bialy publish dataset" in text and "pushed" not in text
