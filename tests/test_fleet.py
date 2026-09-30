import json

from bialy import fleet


def write_tasks(path, repo, n):
    path.mkdir(parents=True, exist_ok=True)
    rows = [{"id": "%s%d" % (repo, i), "prompt": "p%d" % i, "model": "qwen3.6-35b-a3b", "meta": {"repo": repo}} for i in range(n)]
    (path / ("%s.jsonl" % repo)).write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_shards_spread_a_model_over_its_replicas(factory):
    seen = set()
    for lane in range(len(factory.model("gemma-4-26b-a4b-it").ports()) * 2):
        providers, policy, env, chosen = fleet.sidecar_files(factory, {"LYCAON_DECIDE_DISABLED": "1"}, lane)
        seen.add(chosen["gemma-4-26b-a4b-it"])
        assert chosen["gemma-4-26b-a4b-it"] in providers and chosen["gemma-4-26b-a4b-it"] in policy
        assert env.strip() == "LYCAON_DECIDE_DISABLED=1"
    assert len(seen) == len(factory.model("gemma-4-26b-a4b-it").ports())


def test_plan_writes_self_contained_shards(factory, tmp_path):
    write_tasks(tmp_path / "tasks", "flask", 20)
    planned = fleet.plan_shards(factory, tmp_path / "tasks", tmp_path / "run", {"LYCAON_DECIDE_DISABLED": "1"})
    shards = sorted((tmp_path / "run" / "shards").iterdir())
    per = factory.fleet["tasks_per_shard"]
    assert planned == len(shards) == -(-20 // per)
    first = shards[0]
    for name in ("tasks.jsonl", "providers.local.yaml", "model-policy.yaml", "sidecar.env", "repo.json"):
        assert (first / name).exists()
    task = json.loads((first / "tasks.jsonl").read_text().splitlines()[0])
    assert task["provider_id"] in (first / "providers.local.yaml").read_text()
    assert json.loads((first / "repo.json").read_text())["commit"] == factory.repo("flask").commit


def test_replan_rewrites_only_unstarted_shards(factory, tmp_path, monkeypatch):
    monkeypatch.setattr(fleet, "running_containers", lambda run_dir: {})
    write_tasks(tmp_path / "tasks", "flask", 24)
    run = tmp_path / "run"
    fleet.plan_shards(factory, tmp_path / "tasks", run, {})
    started = sorted((run / "shards").iterdir())[0]
    (started / "runner.log").write_text("started")
    before = (started / "providers.local.yaml").read_text()
    removed, planned = fleet.replan(factory, tmp_path / "tasks", run, {"LYCAON_DECIDE_DISABLED": "1"})
    assert removed == planned == len(list((run / "shards").iterdir())) - 1
    assert (started / "providers.local.yaml").read_text() == before and (started / "sidecar.env").read_text().strip() == ""


def test_engine_on_pass_raises_only_turn_decision_deadlines(tmp_path):
    source = tmp_path / "decisions.yaml"
    source.write_text("turn:\n  deadline_ms: 2500\nrerank:\n  summarize:\n    deadline_ms: 300\nrequest:\n  deadline_ms: 2500\n")
    fleet.raise_decision_deadlines(source, tmp_path / "engine", 30000)
    text = (tmp_path / "engine" / fleet.DECISIONS).read_text()
    assert text.count("deadline_ms: 30000") == 2 and "deadline_ms: 300\n" in text


def test_collect_joins_rows_to_their_task_and_lists_what_it_drove(tmp_path):
    write_tasks(tmp_path / "tasks", "flask", 1)
    shard = tmp_path / "run" / "shards" / "flask-000"
    shard.mkdir(parents=True)
    (shard / "manifest.jsonl").write_text(json.dumps({"task_id": "flask0", "root_session": "s1", "status": "settled", "meta": {"repo": "flask"}}) + "\n")
    (shard / "rows.jsonl").write_text(json.dumps({"root_session": "s1", "receipt": 1}) + "\n")
    out, stats = fleet.collect(tmp_path / "run", "engine-off", tmp_path / "tasks")
    row = json.loads(out.read_text())
    assert stats == {"rows": 1, "shards": 1, "tasks": 1, "settled": 1}
    assert row["meta"] == {"repo": "flask", "task_status": "settled", "run": "run", "pass_name": "engine-off"}
    driven = json.loads((tmp_path / "run" / "tasks-driven.jsonl").read_text())
    assert driven["prompt"] == "p0" and driven["outcome"]["status"] == "settled" and driven["outcome"]["shard"] == "flask-000"
