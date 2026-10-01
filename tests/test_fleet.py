import json

import pytest

from bialy import fleet


def write_tasks(path, repo, n):
    path.mkdir(parents=True, exist_ok=True)
    rows = [{"id": "%s%d" % (repo, i), "prompt": "p%d" % i, "model": "qwen3.6-35b-a3b", "meta": {"repo": repo}} for i in range(n)]
    (path / ("%s.jsonl" % repo)).write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_shards_spread_a_model_over_its_replicas(factory):
    seen = set()
    for lane in range(len(factory.model("gemma-4-26b-a4b-it").ports()) * 2):
        providers, policy, env, chosen, _ = fleet.sidecar_files(factory, {"LYCAON_DECIDE_DISABLED": "1"}, lane)
        seen.add(chosen["gemma-4-26b-a4b-it"])
        assert chosen["gemma-4-26b-a4b-it"] in providers and chosen["gemma-4-26b-a4b-it"] in policy
        assert "LYCAON_DECIDE_DISABLED=1" in env and "BIALY_HOSTED_PROVIDERS=fireworks-glm-5-3-flash:FIREWORKS_API_KEY" in env
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


def test_replan_rewrites_every_shard_without_rows(factory, tmp_path, monkeypatch):
    monkeypatch.setattr(fleet, "running_containers", lambda run_dir: {})
    write_tasks(tmp_path / "tasks", "flask", 24)
    run = tmp_path / "run"
    fleet.plan_shards(factory, tmp_path / "tasks", run, {})
    started, failed = sorted((run / "shards").iterdir())[:2]
    (started / "rows.jsonl").write_text("row")
    (failed / "runner.log").write_text("driver exited 1")
    before = (started / "providers.local.yaml").read_text()
    removed, planned = fleet.replan(factory, tmp_path / "tasks", run, {"LYCAON_DECIDE_DISABLED": "1"})
    assert removed == planned == len(list((run / "shards").iterdir())) - 1
    assert (started / "providers.local.yaml").read_text() == before and "LYCAON_DECIDE_DISABLED" not in (started / "sidecar.env").read_text()
    # A fresh plan discards finished shards too, and the pass's collected files.
    (run / "rows.jsonl").write_text("stale")
    removed, planned = fleet.replan(factory, tmp_path / "tasks", run, {"LYCAON_DECIDE_DISABLED": "1"}, fresh=True)
    assert removed == planned == len(list((run / "shards").iterdir())) and not (run / "rows.jsonl").exists()
    assert not (started / "rows.jsonl").exists()


def test_engine_on_pass_raises_only_turn_decision_deadlines(tmp_path):
    config = tmp_path / "config"
    (config / "packs/painted-wolf/platform/host").mkdir(parents=True)
    (config / "packs/painted-wolf/platform/host/decisions.yaml").write_text(
        "turn:\n  deadline_ms: 2500\nrerank:\n  summarize:\n    deadline_ms: 300\nrequest:\n  deadline_ms: 2500\n")
    (config / "packs/painted-wolf/skills.yaml").write_text("skills: []\n")
    fleet.raise_decision_deadlines(config, tmp_path / "engine", 30000)
    # The whole tree travels, since the config root replaces the embedded configuration.
    assert (tmp_path / "engine/overlay/config/packs/painted-wolf/skills.yaml").exists()
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
    assert row["model"] == json.loads((tmp_path / "tasks" / "flask.jsonl").read_text().splitlines()[0])["model"]
    driven = json.loads((tmp_path / "run" / "tasks-driven.jsonl").read_text())
    assert driven["prompt"] == "p0" and driven["outcome"]["status"] == "settled" and driven["outcome"]["shard"] == "flask-000"


def test_shards_are_cut_per_budget_sized_by_time_and_the_longest_start_first(factory, tmp_path):
    tasks = tmp_path / "tasks"
    tasks.mkdir()
    per = factory.fleet["tasks_per_shard"]
    rows = [{"id": "q%d" % i, "prompt": "p", "model": "qwen3.6-35b-a3b", "meta": {"repo": "flask", "prompt_timeout": "60m"}} for i in range(per)]
    rows += [{"id": "w%d" % i, "prompt": "p", "model": "qwen3.6-35b-a3b", "meta": {"repo": "flask", "prompt_timeout": "180m"}} for i in range(per)]
    (tasks / "flask.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    fleet.plan_shards(factory, tasks, tmp_path / "run", {})
    shards = sorted((tmp_path / "run" / "shards").iterdir())
    budgets = [json.loads((s / "repo.json").read_text())["prompt_timeout"] for s in shards]
    sizes = [len((s / "tasks.jsonl").read_text().splitlines()) for s in shards]
    kinds = [{json.loads(line)["id"][0] for line in (s / "tasks.jsonl").read_text().splitlines()} for s in shards]
    long_size = max(1, min(per, factory.fleet["shard_minutes"] // 180))
    assert all(k == {"w"} for k, b in zip(kinds, budgets, strict=True) if b == "180m")
    assert {size for size, b in zip(sizes, budgets, strict=True) if b == "180m"} <= set(range(1, long_size + 1))
    assert budgets.count("60m") == 1 and sizes[budgets.index("60m")] == per
    order = sorted(shards, key=lambda s: (-fleet.shard_minutes(factory, s), s.name))
    assert fleet.shard_minutes(factory, order[0]) == 180


def test_an_engine_payload_missing_a_tool_is_refused(tmp_path):
    for rel in fleet.ENGINE_PAYLOAD + fleet.SCANNER_PAYLOAD:
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).touch()
    fleet.check_engine(tmp_path)
    (tmp_path / "browser/chrome-headless-shell").unlink()
    try:
        fleet.check_engine(tmp_path)
    except ValueError as err:
        assert "browser/chrome-headless-shell" in str(err)
    else:
        raise AssertionError("a payload without the browser was accepted")


def test_hosted_generator_shard_files_name_the_provider_and_keep_the_key_out(factory, monkeypatch, tmp_path):
    providers, policy, env, chosen, wire_model = fleet.sidecar_files(factory, {"LYCAON_DECIDE_DISABLED": "1"}, 0)
    assert chosen["glm-5.3-flash"] == "fireworks-glm-5-3-flash" and wire_model["glm-5.3-flash"] == "accounts/fireworks/models/glm-5p3-flash"
    assert "api_key_env: FIREWORKS_API_KEY" in providers and "accounts/fireworks/models/glm-5p3-flash" in policy
    assert "FIREWORKS_API_KEY=" not in providers and "FIREWORKS_API_KEY=" not in env
    assert "BIALY_HOSTED_PROVIDERS=fireworks-glm-5-3-flash:FIREWORKS_API_KEY" in env
    monkeypatch.setenv("FIREWORKS_API_KEY", "k")
    import dataclasses

    factory = dataclasses.replace(factory, root=tmp_path)
    shard = tmp_path / "shards" / "x"
    shard.mkdir(parents=True)
    (shard / "repo.json").write_text(json.dumps({"repo": "flask", "commit": "c", "prompt_timeout": "5m"}))
    monkeypatch.setattr(fleet, "mount_cache", lambda factory, work: "/cache")
    monkeypatch.setattr(fleet, "release_work", lambda work: None)
    _, _, args = fleet.container_args(factory, "run", shard, [])
    assert "FIREWORKS_API_KEY" in args and "k" not in args


def test_the_scanner_can_be_declined_but_nothing_else(tmp_path):
    for rel in fleet.ENGINE_PAYLOAD:
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text("x")
    with pytest.raises(ValueError, match="opengrep/opengrep"):
        fleet.check_engine(tmp_path)
    fleet.check_engine(tmp_path, scanner=False)


def test_pilot_env_names_the_heads_the_pilot_carries(tmp_path):
    assert fleet.pilot_env(None) == {"LYCAON_DECIDE_DISABLED": "1"}
    (tmp_path / "heads").mkdir()
    (tmp_path / "heads/turn-load.safetensors").write_bytes(b"h")
    (tmp_path / "heads/guide-load.safetensors").write_bytes(b"h")
    env = fleet.pilot_env(tmp_path)
    assert env["LYCAON_DECIDE_BINARY"] == "/opt/decide/bialy" and env["LYCAON_DECIDE_MODEL_DIR"] == "/opt/decide/model"
    assert env["LYCAON_DECIDE_HEADS"] == "guide-load=/opt/decide/heads/guide-load.safetensors,turn-load=/opt/decide/heads/turn-load.safetensors"
    with pytest.raises(ValueError):
        fleet.pilot_env(tmp_path / "empty")


def test_a_shard_is_overdue_well_past_the_sum_of_its_budgets(factory, tmp_path):
    shard = tmp_path / "s"
    shard.mkdir()
    (shard / "tasks.jsonl").write_text('{"id":1}\n{"id":2}\n')
    (shard / "repo.json").write_text(json.dumps({"repo": "flask", "commit": "c", "prompt_timeout": "20m"}))
    assert fleet.shard_deadline(factory, shard, 1000) == 1000 + 60 * (1.5 * 2 * 20 + 30)
