import json
import subprocess
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


def reported(run):
    return run.load_state()["stages"].get("report", {}).get("status") == "done"


def test_run_settings_defaults_and_unknown_keys():
    settings = config.run_settings({"task_cap": 2, "train": {"device": "cpu"}, "engine_on": {"enabled": False}})
    assert settings["task_cap"] == 2 and settings["train"]["device"] == "cpu"
    assert settings["train"]["recipes"] == ["B5", "B7G", "E4", "code-rank", "R1"]
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
    assert record == ["judge", "judge_skillreq", "split_pilot"]
    assert [a for _, a in runmod.plan(run, until="split_pilot")] == ["skip"] * (runmod.STAGES.index("split_pilot") + 1)


def test_redo_reruns_one_stage_and_keeps_the_rest(factory, tmp_path):
    run = make_run(factory, tmp_path)
    record = []
    runmod.execute(run, funcs=stub_funcs(record))
    record.clear()
    runmod.execute(run, funcs=stub_funcs(record), redo=["image"])
    assert record == ["image"] and reported(run)


def test_a_retention_only_run_needs_no_judge_or_writer(factory, tmp_path):
    run = make_run(factory, tmp_path, train={"recipes": ["R1"]}, judge={"enabled": False})
    assert runmod.stage_skillreq(run)["skipped"] and runmod.stage_judge_skillreq(run)["skipped"]
    rows = run.path("fleet-off", "rows.jsonl")
    rows.parent.mkdir(parents=True)
    rows.write_text('{"receipt": 1}\n')
    out = runmod.stage_judge(run)
    assert out["skipped"] and run.path("judged-off.jsonl").read_text() == '{"receipt": 1}\n'
    assert runmod.judged_files(run) == [run.path("judged-off.jsonl")]
    # Follow-up overrides reach every archetype of the scaled factory.
    pilot = make_run(factory, tmp_path, retention={"follow_up_rate": 1.0, "follow_ups": 4})
    shaped = runmod.scaled_factory(pilot).archetypes
    assert {a.follow_up_rate for a in shaped} == {1.0} and {a.max_follow_ups for a in shaped} == {4}
    assert {a.follow_up_rate for a in runmod.scaled_factory(run).archetypes} == {a.follow_up_rate for a in factory.archetypes}


def replay_report(objective, released=5, needed=0, low=50):
    """A retention_eval.py --fit --holdout report as the selection reads it."""
    tally = {"candidates": 500, "released": released, "needed_released": needed, "objective_tokens": objective,
             "objective_per_request": objective // 50, "objective_share": 0.2}
    return {"overall": {"rows": 60, "auc": 0.8, "needed_rate": 0.2, "abstained": 0, "engine": tally, "structural": {"none": dict(tally, objective_tokens=0)}},
            "fitted": {"drop_below": 0.2, "confidence_floor": 0.7, "tally": tally},
            "holdout": {"rows": 50, "auc": 0.78, "needed_rate": 0.19, "abstained": 0, "engine": tally, "policy": {"drop_below": 0.2, "confidence_floor": 0.7},
                        "structural": {"none": dict(tally, objective_tokens=0), "repeated": dict(tally, objective_tokens=-400), "stale": dict(tally, objective_tokens=100)},
                        "bootstrap": {"engine": {"low": low, "median": 200, "high": 400, "draws": 500}}},
            "recommendation": {"facts": [], "fragment": "retention:\n  drop_below: 0.2\n  confidence_floor: 0.7\n"}}


def test_evaluate_selects_the_searched_retention_head_and_reads_the_verdict(factory, tmp_path, monkeypatch):
    run = make_run(factory, tmp_path, train={"recipes": ["R1"]})
    search = run.path("heads", "search")
    search.mkdir(parents=True)
    for name in ("lr0.001-pw6-needed", "lr0.0005-pw6-needed"):
        (search / ("retention-%s.safetensors" % name)).write_bytes(name.encode())
    monkeypatch.setattr(runmod, "model_dir", lambda run: tmp_path)
    monkeypatch.setattr(runmod.build, "launcher", lambda engine, model, heads, out, device=None: Path(out))
    for split in ("val", "holdout"):
        rows = run.path("split", "retention", "%s.jsonl" % split)
        rows.parent.mkdir(parents=True, exist_ok=True)
        rows.write_text(json.dumps({"schema": "pw-retention-row/1", "partial": False}) + "\n")
    objectives = {"lr0.001-pw6-needed": 9000, "lr0.0005-pw6-needed": 12000}
    calls = []

    def fake_run(args, cwd=None, capture_output=None, text=None, env=None):
        calls.append(args)
        name = Path(args[args.index("--json") + 1]).stem
        Path(args[args.index("--json") + 1]).write_text(json.dumps(replay_report(objectives[name])))
        return subprocess.CompletedProcess(args, 0, "", "")
    monkeypatch.setattr(runmod.subprocess, "run", fake_run)
    out = runmod.stage_evaluate(run)
    assert len(calls) == 2 and all("--fit" in a and "--holdout" in a for a in calls)
    retained = out["retention"]
    assert retained["chosen"] == "lr0.0005-pw6-needed" and run.path("heads", "retention.safetensors").read_bytes() == b"lr0.0005-pw6-needed"
    assert retained["fitted"] == {"drop_below": 0.2, "confidence_floor": 0.7} and retained["facts"] == []
    assert retained["viability"]["viable"] and retained["viability"]["reasons"] == []
    selection = json.loads(Path(retained["selection"]).read_text())
    assert selection["chosen"] == "lr0.0005-pw6-needed" and set(selection["search"]) == set(objectives)
    assert "drop_below: 0.2" in Path(retained["fragment"]).read_text()
    # The turn replay has no heads to score, and the report carries the section.
    assert out["evals"]["skipped"]
    state = run.load_state()
    state["stages"]["evaluate"] = {"status": "done", "at": "now", "outputs": out}
    run.save_state(state)
    text = Path(runmod.stage_report(run, publish=False)["report"]).read_text()
    assert "## Retention head" in text and "**Viable.**" in text and "| Search point |" in text and "lr0.001-pw6-needed" in text


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
    assert scaled.judge_for("glm-5.3-flash").family != "glm"
    assert runmod.writer_model(run).id in {m.id for m in factory.models if "writer" in m.roles}
    # A model named as neither generator nor judge keeps its other roles.
    assert "writer" in scaled.model("glm-5.3-flash").roles and "qwen3.6-35b-a3b" not in {m.id for m in scaled.models}
    run = make_run(factory, tmp_path, generators=["glm-5.3-flash"], judges=["deepseek-v4.1-flash"])
    scaled = runmod.scaled_factory(run)
    assert [m.id for m in scaled.models if "judge" in m.roles] == ["deepseek-v4.1-flash"]


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


def test_evaluation_replays_a_seeded_sample_in_file_order(tmp_path):
    rows = tmp_path / "holdout.jsonl"
    rows.write_text("".join(json.dumps({"n": i}) + "\n" for i in range(50)))
    out = tmp_path / "holdout.sample.jsonl"
    picked, total = runmod.sample_rows(rows, out, 10, "7:evaluate:holdout")
    kept = [json.loads(line)["n"] for line in out.read_text().splitlines()]
    assert picked == out and total == 50 and len(kept) == 10 and kept == sorted(kept)
    assert runmod.sample_rows(rows, tmp_path / "again.jsonl", 10, "7:evaluate:holdout")[0].read_text() == out.read_text()
    # A split no larger than the sample, or no limit, replays as it is.
    assert runmod.sample_rows(rows, tmp_path / "x.jsonl", 50, "s") == (rows, 50)
    assert runmod.sample_rows(rows, tmp_path / "x.jsonl", 0, "s") == (rows, 50)


def test_code_rank_sites_are_evaluated_side_by_side(factory, tmp_path, monkeypatch):
    import threading

    run = make_run(factory, tmp_path, workspaces=["zod", "sinatra"])
    for repo in ("zod", "sinatra"):
        for kind, name in (("units", repo + ".jsonl"), ("pairs", "docs-%s.jsonl" % repo)):
            path = run.path("coderank", kind, name)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("{}\n")
    running, peak, lock = [0], [0], threading.Lock()
    release = threading.Barrier(runmod.RERANK_WORKERS, timeout=5)

    def eval_site(args, **kw):
        with lock:
            running[0] += 1
            peak[0] = max(peak[0], running[0])
        release.wait()
        with lock:
            running[0] -= 1
        return subprocess.CompletedProcess(args, 0, "", "")

    import subprocess
    monkeypatch.setattr(runmod, "model_dir", lambda run: tmp_path / "model")
    monkeypatch.setattr(runmod.subprocess, "run", eval_site)
    reports = runmod.rerank_evals(run, {"code-rank": "head"})
    assert sorted(reports) == sorted("%s-%s" % (s, r) for s in runmod.coderank.SITES for r in ("zod", "sinatra"))
    assert peak[0] == runmod.RERANK_WORKERS


def test_a_section_setting_takes_its_keys():
    with pytest.raises(config.ConfigError, match="run.engine_on is a section"):
        config.run_settings({"engine_on": False})
