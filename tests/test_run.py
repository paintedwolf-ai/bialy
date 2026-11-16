import json
import subprocess
from pathlib import Path

import pytest

from bialy import build, config, custom_model_run, heads, host_scripts
from bialy import run as runmod
from bialy.campaign import selector
from bialy.campaign.journal import CampaignFault


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


def reported(run):
    return run.load_state()["stages"].get("report", {}).get("status") == "done"


def test_run_settings_defaults_and_unknown_keys():
    settings = config.run_settings({"task_cap": 2, "train": {"device": "cpu"}, "engine_on": {"enabled": False}})
    assert settings["task_cap"] == 2 and settings["train"]["device"] == "cpu"
    assert settings["train"]["recipes"] == ["B5", "E4", "code-rank"]
    assert settings["engine_on"] == {"enabled": False, "deadline_ms": 60000, "heads": None}
    assert config.run_settings({"engine_on": {"deadline_ms": 5}})["engine_on"] == {"enabled": True, "deadline_ms": 5, "heads": None}
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
    # A pilot policy is fitted like the final one: it needs the site dumps and the engine that calibrates it.
    assert runmod.STAGES.index("coderank") < runmod.STAGES.index("engine") < runmod.STAGES.index("train_pilot")
    assert runmod.STAGES.index("train") < runmod.STAGES.index("evaluate") < runmod.STAGES.index("release_heads")


def test_execute_records_every_stage_and_resumes(factory, tmp_path):
    run = make_run(factory, tmp_path)
    record = []
    state = runmod.execute(run, funcs=stub_funcs(record))
    assert record == list(runmod.STAGES)
    assert all(state["stages"][name]["status"] == "done" for name in runmod.STAGES)
    assert json.loads(run.state_path.read_text())["stages"]["tasks"]["outputs"] == {"did": "tasks"}
    assert reported(run)
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
    assert not reported(run)
    record.clear()
    runmod.execute(run, until="split_pilot", funcs=stub_funcs(record))
    assert record == ["judge", "judge_skillreq", "coderank", "engine", "split_pilot"]
    assert [a for _, a in runmod.plan(run, until="split_pilot")] == ["skip"] * (runmod.STAGES.index("split_pilot") + 1)


def test_redo_reruns_one_stage_and_keeps_the_rest(factory, tmp_path):
    run = make_run(factory, tmp_path)
    record = []
    runmod.execute(run, funcs=stub_funcs(record))
    record.clear()
    runmod.execute(run, funcs=stub_funcs(record), redo=["image"])
    assert record == ["image"] and reported(run)


def test_a_collection_only_run_needs_no_judge_or_writer(factory, tmp_path):
    run = make_run(factory, tmp_path, train={"recipes": []}, judge={"enabled": False})
    assert runmod.stage_skillreq(run)["skipped"] and runmod.stage_judge_skillreq(run)["skipped"] and runmod.stage_coderank(run)["skipped"]
    rows = run.path("fleet-off", "rows.jsonl")
    rows.parent.mkdir(parents=True)
    rows.write_text('{"receipt": 1}\n')
    out = runmod.stage_judge(run)
    assert out["skipped"] and run.path("judged-off.jsonl").read_text() == '{"receipt": 1}\n'
    assert runmod.judged_files(run) == [run.path("judged-off.jsonl")]
    # Follow-up overrides reach every archetype of the scaled factory.
    shaped_run = make_run(factory, tmp_path, retention={"follow_up_rate": 1.0, "follow_ups": 4})
    shaped = runmod.scaled_factory(shaped_run).archetypes
    assert {a.follow_up_rate for a in shaped} == {1.0} and {a.max_follow_ups for a in shaped} == {4}
    assert {a.follow_up_rate for a in runmod.scaled_factory(run).archetypes} == {a.follow_up_rate for a in factory.archetypes}


def test_compare_matches_requests_across_passes(factory, tmp_path):
    run = make_run(factory, tmp_path)
    off = run.path("fleet-off", "requests.jsonl")
    on = run.path("fleet-on", "requests.jsonl")
    off.parent.mkdir(parents=True)
    on.parent.mkdir(parents=True)

    def row(task, index, tokens, recalls, dropped):
        return {"schema": "pw-request-row/1", "host": "coordinator", "surface": "implement_investigate", "partial": False, "request_index": index,
                "calls": 4, "prompt_tokens": tokens, "recall_calls": recalls, "repeated_calls": 0, "meta": {"task_id": task},
                "retention": {"recorded": True, "asked": dropped is not None, "watermark": False, "tokens": 1, "candidates": 3, "dropped": dropped or 0}}
    off.write_text("".join(json.dumps(r) + "\n" for r in (row("t1", 0, 10000, 0, None), row("t1", 1, 20000, 0, None), row("t2", 0, 5000, 0, None))))
    on.write_text("".join(json.dumps(r) + "\n" for r in (row("t1", 0, 9000, 0, 1), row("t1", 1, 15000, 1, 2), row("t3", 0, 1, 0, 0))))
    out = runmod.stage_compare(run)
    c = out["comparison"]
    assert c["requests"] == 2 and c["off_only"] == 1 and c["on_only"] == 1 and c["released_results"] == 3 and c["released_requests"] == 2
    assert c["measures"]["prompt_tokens"] == {"off": 15000.0, "on": 12000.0, "delta": -3000.0, "delta_median": -3000, "wants": "down"}
    assert c["measures"]["recall_calls"]["delta"] == 0.5
    assert runmod.stage_compare(make_run(factory, tmp_path / "other", engine_on={"enabled": False, "deadline_ms": 1}))["skipped"]


def test_engine_on_stages_skip_when_the_second_pass_is_off(factory, tmp_path):
    run = make_run(factory, tmp_path, engine_on={"enabled": False, "deadline_ms": 1})
    for name in runmod.ENGINE_ON_STAGES:
        if name != "pilot":
            assert runmod.STAGE_FUNCS[name](run) == {"skipped": "run.engine_on is off"}
    assert runmod.stage_pilot(run)["skipped"]


def test_released_heads_drive_the_engine_on_pass_only_under_the_release_that_pins_them(factory, tmp_path):
    released = tmp_path / "heads"
    released.mkdir()
    checkout = tmp_path / "co"
    release = checkout / "lycaon/config" / build.RELEASE
    release.parent.mkdir(parents=True)
    (released / "turn-load.safetensors").write_bytes(b"shipped")
    release.write_text(json.dumps({"release": "r1", "heads": {"turn-load": {"sha256": runmod.provenance.file_sha256(released / "turn-load.safetensors")}}}))
    run = make_run(factory, tmp_path, engine_on={"enabled": True, "deadline_ms": 1, "heads": str(released)}, lycaon_checkout=str(checkout))
    assert run.engine_on_only
    for name in ("plan", "drive", "collect", "judge", "split_pilot", "train_pilot"):
        assert runmod.STAGE_FUNCS[name](run) == {"skipped": "run.engine_on.heads drives the engine-on pass only"}
    assert runmod.released_heads(run) == {"turn-load": released / "turn-load.safetensors"}
    # Heads the checkout does not pin would be served another release's option vocabulary.
    (released / "turn-load.safetensors").write_bytes(b"retrained")
    with pytest.raises(runmod.RunError, match="decision release r1 pins"):
        runmod.released_heads(run)
    assert not make_run(factory, tmp_path).engine_on_only


class Selector:
    """selector.fit, qualify and holdout, faked: what a run's stages asked of them."""

    def __init__(self, monkeypatch, unqualified=None):
        self.calls, self.unqualified = [], unqualified
        monkeypatch.setattr(selector, "Workbench", lambda run: "bench")
        monkeypatch.setattr(selector, "fit", self.fit)
        monkeypatch.setattr(selector, "qualify", self.qualify)
        monkeypatch.setattr(selector, "holdout", self.holdout)

    def fit(self, run, split, directory, workbench, config, label):
        self.calls.append(("fit", split.name, directory.name, label))
        (directory / "pilot" / "heads").mkdir(parents=True)
        (directory / "pilot" / "heads" / "turn-load.safetensors").write_bytes(b"h")
        return {"policy": {"identity": "p-" + label}, "heads": {"turn-load": "h"}, "pilot": str(directory / "pilot")}

    def qualify(self, run, split, directory, trained, workbench, config):
        self.calls.append(("qualify", split.name, directory.name))
        if self.unqualified:
            raise CampaignFault("selector_unqualified", details={"failures": [self.unqualified]})
        return {"policy": trained["policy"]["identity"], "measures": {}, "parity": {"passed": True}}

    def holdout(self, run, split, directory, trained, workbench):
        self.calls.append(("holdout", split.name, directory.name))
        return {"policy": trained["policy"]["identity"]}


def trained_run(factory, tmp_path, stage="train", **settings):
    run = make_run(factory, tmp_path, **settings)
    state = run.load_state()
    state["stages"][stage] = {"status": "done", "at": "now", "outputs": runmod.STAGE_FUNCS[stage](run)}
    run.save_state(state)
    return run


def test_both_policies_come_from_the_selector_over_their_own_split(factory, tmp_path, monkeypatch):
    fake = Selector(monkeypatch)
    run = make_run(factory, tmp_path)
    assert runmod.stage_train_pilot(run)["selector"]["policy"] == {"identity": "p-pilot"}
    assert runmod.stage_train(run)["selector"]["policy"] == {"identity": "p-final"}
    assert fake.calls == [("fit", "split-pilot", "pilot-policy", "pilot"), ("fit", "split", "selector", "final")]
    collecting = make_run(factory, tmp_path / "r", train={"recipes": []})
    assert runmod.stage_train(collecting)["skipped"] and runmod.stage_evaluate(collecting)["skipped"]
    with pytest.raises(runmod.RunError, match="no policy to serve"):
        runmod.stage_pilot(collecting)


def test_evaluate_qualifies_then_compares_and_an_unqualified_policy_fails_the_stage(factory, tmp_path, monkeypatch):
    fake = Selector(monkeypatch)
    run = trained_run(factory, tmp_path)
    out = runmod.stage_evaluate(run)
    assert out["policy"] == "p-final" and out["holdout"] == {"policy": "p-final"}
    assert fake.calls[1:] == [("qualify", "split", "selector"), ("holdout", "split", "selector")]
    failing = Selector(monkeypatch, unqualified={"head": "code-rank", "metric": "mrr_gain", "value": -0.02, "at_least": 0.01})
    with pytest.raises(runmod.RunError, match="selector_unqualified.*mrr_gain"):
        runmod.stage_evaluate(run)
    assert failing.calls == [("qualify", "split", "selector")]


def test_no_heads_release_is_cut_from_an_unqualified_policy(factory, tmp_path, monkeypatch):
    Selector(monkeypatch)
    run = trained_run(factory, tmp_path)
    with pytest.raises(runmod.RunError, match="selector policy"):
        runmod.stage_release_heads(run)
    assert not run.path("dist").exists()
    built = []
    monkeypatch.setattr(heads, "build", lambda directory, version, dataset, commit, out: built.append((directory.name, version, commit)))
    monkeypatch.setattr(runmod.build, "checkout_commit", lambda checkout: "c0")
    assert runmod.stage_release_heads(run)["version"] == "t" and built == [("selector", "t", "c0")]
    assert runmod.stage_release_heads(make_run(factory, tmp_path / "none"))["skipped"]


def test_the_engine_on_pass_mounts_the_pilot_policy(factory, tmp_path, monkeypatch):
    Selector(monkeypatch)
    run = trained_run(factory, tmp_path, stage="train_pilot")
    monkeypatch.setattr(runmod.subprocess, "run", lambda args, **kw: subprocess.CompletedProcess(args, 0, "", ""))
    out = runmod.stage_pilot(run)
    assert out["pilot"] == str(run.path("pilot-policy", "pilot")) and out["policy"] == "p-pilot"
    state = run.load_state()
    state["stages"]["pilot"] = {"status": "done", "outputs": out}
    run.save_state(state)
    assert runmod.pilot_mount(run) == run.path("pilot-policy", "pilot")


def test_check_refuses_part_of_a_policy_and_a_checkout_that_dropped_a_flag(factory, tmp_path, monkeypatch):
    monkeypatch.setenv("FIREWORKS_API_KEY", "k")
    contract = tmp_path / "co" / host_scripts.CONTRACT
    contract.parent.mkdir(parents=True)
    recorded = json.loads((Path(__file__).parent / "fixtures" / "factory-contract.json").read_text())
    recorded["scripts"]["replay_eval.py"].remove("--tool-truth")
    contract.write_text(json.dumps(recorded))
    run = make_run(factory, tmp_path, train={"recipes": ["B5", "E4"]}, lycaon_checkout=str(tmp_path / "co"))
    with pytest.raises(runmod.RunError) as exc:
        runmod.stage_check(run)
    assert "a selector policy is whole or absent" in str(exc.value) and "replay_eval.py --tool-truth" in str(exc.value)


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


def test_check_refuses_a_ceiling_without_prices_and_a_missing_key(factory, tmp_path, monkeypatch):
    import dataclasses

    monkeypatch.delenv("FIREWORKS_API_KEY", raising=False)
    unpriced = dataclasses.replace(factory, models=[
        dataclasses.replace(m, hosted={k: v for k, v in m.hosted.items() if k != "price_per_million"}) if m.hosted else m for m in factory.models])
    run = make_run(unpriced, tmp_path, spend_ceiling_usd=5, generators=["glm-5.3-flash"], lycaon_checkout=str(tmp_path))
    with pytest.raises(runmod.RunError) as exc:
        runmod.stage_check(run)
    message = str(exc.value)
    assert "needs FIREWORKS_API_KEY" in message and "price_per_million" in message and "lacks lycaon/cmd/lycaon" in message


def test_scaled_factory_caps_tasks_and_narrows_generators(factory, tmp_path):
    run = make_run(factory, tmp_path, task_cap=1, generators=["glm-5.3-flash"], workspaces=["flask", "react-vite"])
    scaled = runmod.scaled_factory(run)
    assert all(a.per_workspace == 1 for a in scaled.archetypes)
    assert [m.id for m in scaled.generators()] == ["glm-5.3-flash"]
    assert [w.name for w in scaled.workspaces()] == ["flask", "react-vite"]
    assert scaled.reviewer().id == run.settings["judges"][0]
    assert runmod.writer_model(run).id in {m.id for m in factory.models if "writer" in m.roles}
    # A model named as neither generator nor judge keeps its other roles.
    assert "writer" in scaled.model("glm-5.3-flash").roles and "qwen3.6-35b-a3b" not in {m.id for m in scaled.models}
    run = make_run(factory, tmp_path, generators=["glm-5.3-flash"], judges=["deepseek-v4.1-flash"])
    scaled = runmod.scaled_factory(run)
    assert [m.id for m in scaled.models if "judge" in m.roles] == ["deepseek-v4.1-flash"]
    # A share scales every archetype and keeps at least one task of each; a cap then bounds it.
    shared = runmod.scaled_factory(make_run(factory, tmp_path, task_share=0.19))
    by_id = {a.id: a for a in factory.archetypes}
    assert all(a.per_workspace == max(1, round(by_id[a.id].per_workspace * 0.19)) for a in shared.archetypes)
    assert sum(a.per_workspace for a in shared.archetypes) < sum(a.per_workspace for a in factory.archetypes)
    assert all(a.per_workspace == 1 for a in runmod.scaled_factory(make_run(factory, tmp_path, task_share=0.5, task_cap=1)).archetypes)
    with pytest.raises(runmod.RunError, match="task_share"):
        runmod.scaled_factory(make_run(factory, tmp_path, task_share=1.5))
    # Archetype selection keeps only the named kinds of request.
    picked = runmod.scaled_factory(make_run(factory, tmp_path, archetypes=["repository_question", "new_project"]))
    assert sorted(a.id for a in picked.archetypes) == ["new_project", "repository_question"]
    with pytest.raises(runmod.RunError, match="unknown archetypes"):
        runmod.scaled_factory(make_run(factory, tmp_path, archetypes=["no_such_archetype"]))


def test_a_collection_declares_the_students_window_and_identity(factory, tmp_path):
    run = runmod.Run(factory=factory, name="t", settings=runmod.settings_for(factory, {"custom_model.enabled": True}))
    assert run.collection_window == 65536
    assert make_run(factory, tmp_path).collection_window is None
    students = run.settings["custom_model"]["students"]
    name = next(iter(students))
    students[name] = dict(students[name], context_window=32768)
    with pytest.raises(custom_model_run.DatasetError, match="one context window"):
        _ = run.collection_window


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


def test_a_section_setting_takes_its_keys():
    with pytest.raises(config.ConfigError, match="run.engine_on is a section"):
        config.run_settings({"engine_on": False})


def test_check_validates_routing_before_provider_stages(factory, tmp_path):
    run = make_run(factory, tmp_path, generators=["glm-5.3-flash", "missing"],
                   generator_weights={"glm-5.3-flash": float("nan")},
                   archetype_generators={"misspelled": ["glm-5.3-flash"]})
    with pytest.raises(runmod.RunError) as exc:
        runmod.stage_check(run)
    message = str(exc.value)
    assert "unavailable generators: missing" in message
    assert "unknown archetypes: misspelled" in message
    assert "finite nonnegative" in message


def test_a_run_scales_and_filters_scenarios_like_archetypes(factory, tmp_path):
    import dataclasses

    from bialy.config import Scenario

    def written(sid, workspace, archetype):
        return Scenario(id=sid, workspace=workspace, archetype=archetype,
                        variants=tuple({"prompt": "variant %d of scenario %s, typed another way" % (i, sid), "follow_ups": []} for i in range(12)))
    f = dataclasses.replace(factory, scenario_variants=10, scenarios=[
        written("flask-bug", "flask", "bugfix"), written("flask-where", "flask", "locate"), written("gin-bug", "gin", "bugfix")])
    capped = runmod.scaled_factory(make_run(f, tmp_path, task_cap=2, workspaces=["flask"], archetypes=["bugfix", "locate"]))
    assert [s.id for s in capped.scenarios] == ["flask-bug", "flask-where"] and capped.scenario_variants == 2
    shared = runmod.scaled_factory(make_run(f, tmp_path, task_share=0.25, archetypes=["bugfix"]))
    assert [s.id for s in shared.scenarios] == ["flask-bug", "gin-bug"] and shared.scenario_variants == 2
    assert runmod.scaled_factory(make_run(f, tmp_path)).scenario_variants == 10
