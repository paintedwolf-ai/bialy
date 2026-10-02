import json

import pytest

from bialy import fleet


def write_tasks(path, repo, n):
    path.mkdir(parents=True, exist_ok=True)
    rows = [{"id": "%s%d" % (repo, i), "prompt": "p%d" % i, "model": "qwen3.6-35b-a3b", "meta": {"workspace": repo}} for i in range(n)]
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
    for name in ("tasks.jsonl", "providers.local.yaml", "model-policy.yaml", "sidecar.env", "workspace.json"):
        assert (first / name).exists()
    task = json.loads((first / "tasks.jsonl").read_text().splitlines()[0])
    assert task["provider_id"] in (first / "providers.local.yaml").read_text()
    assert json.loads((first / "workspace.json").read_text()) == {
        "workspace": "flask", "kind": "repository", "commit": factory.repo("flask").commit, "prompt_timeout": "60m"}


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
    (shard / "manifest.jsonl").write_text(json.dumps({"task_id": "flask0", "root_session": "s1", "status": "settled", "meta": {"workspace": "flask"}}) + "\n")
    (shard / "rows.jsonl").write_text(json.dumps({"root_session": "s1", "receipt": 1}) + "\n")
    out, stats = fleet.collect(tmp_path / "run", "engine-off", tmp_path / "tasks")
    row = json.loads(out.read_text())
    assert stats == {"rows": 1, "shards": 1, "tasks": 1, "settled": 1}
    assert row["meta"] == {"workspace": "flask", "task_status": "settled", "run": "run", "pass_name": "engine-off"}
    assert row["model"] == json.loads((tmp_path / "tasks" / "flask.jsonl").read_text().splitlines()[0])["model"]
    driven = json.loads((tmp_path / "run" / "tasks-driven.jsonl").read_text())
    assert driven["prompt"] == "p0" and driven["outcome"]["status"] == "settled" and driven["outcome"]["shard"] == "flask-000"


def test_shards_are_cut_per_budget_sized_by_time_and_the_longest_start_first(factory, tmp_path):
    tasks = tmp_path / "tasks"
    tasks.mkdir()
    per = factory.fleet["tasks_per_shard"]
    rows = [{"id": "q%d" % i, "prompt": "p", "model": "qwen3.6-35b-a3b", "meta": {"workspace": "flask", "prompt_timeout": "60m"}} for i in range(per)]
    rows += [{"id": "w%d" % i, "prompt": "p", "model": "qwen3.6-35b-a3b", "meta": {"workspace": "flask", "prompt_timeout": "180m"}} for i in range(per)]
    (tasks / "flask.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    fleet.plan_shards(factory, tasks, tmp_path / "run", {})
    shards = sorted((tmp_path / "run" / "shards").iterdir())
    budgets = [json.loads((s / "workspace.json").read_text())["prompt_timeout"] for s in shards]
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
    (shard / "workspace.json").write_text(json.dumps({"workspace": "flask", "kind": "repository", "commit": "c", "prompt_timeout": "5m"}))
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
    (shard / "workspace.json").write_text(json.dumps({"workspace": "flask", "kind": "repository", "commit": "c", "prompt_timeout": "20m"}))
    assert fleet.shard_deadline(factory, shard, 1000) == 1000 + 60 * (1.5 * 2 * 20 + 30)


class Refusal(Exception):
    status_code = 402


def fake_chat(refuse):
    class Completions:
        def create(self, **kw):
            if refuse:
                raise Refusal("insufficient credit")

    class Client:
        chat = type("Chat", (), {"completions": Completions()})()

        def with_options(self, **kw):
            return self

    class Chat:
        def __init__(self, base_url, model_id, api_key="unused", hosted=False):
            self.client, self.model = Client(), model_id

    return Chat


def test_an_account_refusal_names_the_generator(factory, monkeypatch):
    monkeypatch.setattr(fleet.llm, "Chat", fake_chat(refuse=False))
    assert fleet.account_refusal(factory) is None
    monkeypatch.setattr(fleet.llm, "Chat", fake_chat(refuse=True))
    assert fleet.account_refusal(factory) == "glm-5.3-flash: HTTP 402"


def test_a_refused_account_stops_the_fleet_and_keeps_lost_shards_for_a_resume(factory, monkeypatch, tmp_path):
    import dataclasses

    factory = dataclasses.replace(factory, root=tmp_path)
    run_dir = tmp_path / "runs" / "r"
    # a drives cleanly; b loses a task to the refused account; c must never start.
    outcomes = {"a": ["settled", "settled"], "b": ["settled", "failed"], "c": ["settled", "settled"]}
    for name in outcomes:
        shard = run_dir / "shards" / name
        shard.mkdir(parents=True)
        (shard / "tasks.jsonl").write_text('{"id":1}\n{"id":2}\n')
        (shard / "workspace.json").write_text(json.dumps({"workspace": "flask", "kind": "repository", "commit": "c", "prompt_timeout": "5m"}))
    started = []

    class Container:
        def __init__(self, args, **kw):
            shard = run_dir / "shards" / args[0]
            started.append(shard.name)
            (shard / "manifest.jsonl").write_text("".join(json.dumps({"status": s}) + "\n" for s in outcomes[shard.name]))
            (shard / "rows.jsonl").write_text("{}\n")

        def poll(self):
            return 0

    monkeypatch.setattr(fleet, "running_containers", lambda run_dir: {})
    monkeypatch.setattr(fleet, "container_args", lambda factory, run_name, shard, mounts, image: (shard.name, None, [shard.name]))
    monkeypatch.setattr(fleet, "release_work", lambda work: None)
    monkeypatch.setattr(fleet.subprocess, "Popen", Container)
    monkeypatch.setattr(fleet.time, "sleep", lambda s: None)
    monkeypatch.setattr(fleet, "account_refusal", lambda factory: "glm-5.3-flash: HTTP 402")
    with pytest.raises(fleet.AccountRefused, match="HTTP 402.*2 shards wait"):
        fleet.run(factory, run_dir, runners=1)
    assert started == ["a", "b"]
    assert (run_dir / "shards" / "a" / "rows.jsonl").exists()
    assert not (run_dir / "shards" / "b" / "rows.jsonl").exists()


def test_a_task_failing_for_its_own_reasons_keeps_the_fleet_going(factory, monkeypatch, tmp_path):
    shard = tmp_path / "s"
    shard.mkdir()
    (shard / "rows.jsonl").write_text("{}\n")
    (shard / "manifest.jsonl").write_text('{"status": "settled"}\n{"status": "timeout"}\n')
    assert not fleet.lost_tasks(shard)
    (shard / "manifest.jsonl").write_text('{"status": "settled"}\n{"status": "failed"}\n')
    assert fleet.lost_tasks(shard)
    monkeypatch.setattr(fleet.llm, "Chat", fake_chat(refuse=False))
    assert fleet.account_refusal(factory) is None


def test_a_stack_shard_starts_in_an_empty_workspace(factory, monkeypatch, tmp_path):
    import dataclasses

    write_tasks(tmp_path / "tasks", "react-vite", 3)
    fleet.plan_shards(factory, tmp_path / "tasks", tmp_path / "run", {}, workspaces=["react-vite"])
    [shard] = sorted((tmp_path / "run" / "shards").iterdir())
    assert shard.name == "react-vite-000"
    assert json.loads((shard / "workspace.json").read_text()) == {"workspace": "react-vite", "kind": "stack", "prompt_timeout": "60m"}
    monkeypatch.setattr(fleet, "mount_cache", lambda factory, work: "/cache")
    monkeypatch.setattr(fleet, "release_work", lambda work: None)
    _, _, args = fleet.container_args(dataclasses.replace(factory, root=tmp_path), "run", shard, [])
    assert "WORKSPACE=react-vite" in args and "WORKSPACE_KIND=stack" in args
    assert not any(a.startswith("COMMIT=") for a in args)


def test_the_deadline_gives_every_prompt_its_budget(factory, tmp_path):
    shard = tmp_path / "s"
    shard.mkdir()
    (shard / "workspace.json").write_text(json.dumps({"workspace": "react-vite", "kind": "stack", "prompt_timeout": "150m"}))
    (shard / "tasks.jsonl").write_text(json.dumps({"id": "a", "follow_ups": [{"prompt": "x"}, {"prompt": "y"}]}) + "\n")
    # One task with two follow-ups is three prompts of 150 minutes each.
    assert fleet.shard_deadline(factory, shard, 0) == 60 * (1.5 * 3 * 150 + 30)


def overdue_shard(tmp_path, exported_roots):
    """A shard of three tasks whose runner finished the first and was cut off in the second,
    and the rows its live store held when it was stopped."""
    shard = tmp_path / "s"
    shard.mkdir()
    (shard / "tasks.jsonl").write_text("".join(json.dumps({"id": t}) + "\n" for t in ("t1", "t2", "t3")))
    (shard / "manifest.jsonl").write_text(json.dumps({"task_id": "t1", "root_session": "s1", "status": "settled"}) + "\n")
    exported = shard / "overdue-rows.part"
    exported.write_text("".join(json.dumps({"root_session": r, "receipt": n}) + "\n" for n, r in enumerate(exported_roots)))
    return shard, exported


def test_an_overdue_shard_keeps_its_finished_sessions_and_the_turns_it_cut_off(tmp_path):
    shard, exported = overdue_shard(tmp_path, ["s1", "s1", "s2"])
    assert fleet.keep_overdue(shard, exported)
    manifest = [json.loads(line) for line in (shard / "manifest.jsonl").read_text().splitlines()]
    assert manifest[1:] == [{"task_id": "t2", "root_session": "s2", "status": "overdue"},
                            {"task_id": "t3", "root_session": None, "status": "overdue"}]
    assert len((shard / "rows.jsonl").read_text().splitlines()) == 3 and not exported.exists()


def test_sessions_no_task_can_claim_are_dropped(tmp_path):
    shard, exported = overdue_shard(tmp_path, ["s1", "s2", "s9"])
    assert fleet.keep_overdue(shard, exported)
    assert [json.loads(line)["root_session"] for line in (shard / "rows.jsonl").read_text().splitlines()] == ["s1"]
    manifest = [json.loads(line) for line in (shard / "manifest.jsonl").read_text().splitlines()]
    assert [m["root_session"] for m in manifest[1:]] == [None, None]


def test_an_overdue_shard_that_drove_nothing_runs_again(tmp_path):
    shard, exported = overdue_shard(tmp_path, [])
    (shard / "manifest.jsonl").write_text("")
    assert not fleet.keep_overdue(shard, exported) and not (shard / "rows.jsonl").exists()


def test_the_fleet_stops_an_overdue_shard_and_counts_what_it_kept(factory, monkeypatch, tmp_path):
    import dataclasses

    factory = dataclasses.replace(factory, root=tmp_path)
    run_dir = tmp_path / "runs" / "r"
    shard = run_dir / "shards" / "react-vite-000"
    shard.mkdir(parents=True)
    (shard / "tasks.jsonl").write_text(json.dumps({"id": "t1"}) + "\n" + json.dumps({"id": "t2"}) + "\n")
    (shard / "workspace.json").write_text(json.dumps({"workspace": "react-vite", "kind": "stack", "prompt_timeout": "5m"}))
    commands = []

    class Hung:
        def __init__(self, args, **kw):
            (shard / "manifest.jsonl").write_text(json.dumps({"task_id": "t1", "root_session": "s1", "status": "settled"}) + "\n")

        def poll(self):
            return None

        def wait(self):
            return 137

    def export(container, shard, out_name, roots=None):
        (shard / out_name).write_text(json.dumps({"root_session": "s1"}) + "\n" + json.dumps({"root_session": "s2"}) + "\n")
        return True

    monkeypatch.setattr(fleet, "running_containers", lambda run_dir: {})
    monkeypatch.setattr(fleet, "container_args", lambda factory, run_name, shard, mounts, image: (shard.name, None, [shard.name]))
    monkeypatch.setattr(fleet, "release_work", lambda work: None)
    monkeypatch.setattr(fleet.subprocess, "Popen", Hung)
    monkeypatch.setattr(fleet.time, "sleep", lambda s: None)
    monkeypatch.setattr(fleet, "shard_deadline", lambda factory, shard, started: 0)
    monkeypatch.setattr(fleet, "export_live", export)
    monkeypatch.setattr(fleet, "sh", lambda *args, **kw: commands.append(args))
    assert fleet.run(factory, run_dir, runners=1) == (1, 0)
    assert ("docker", "kill", "react-vite-000") in commands
    ledger = [json.loads(line) for line in (run_dir / "ledger.jsonl").read_text().splitlines()]
    assert {"event": "overdue", "container": "react-vite-000", "shard": "react-vite-000", "kept": True}.items() <= ledger[-2].items()
    manifest = [json.loads(line) for line in (shard / "manifest.jsonl").read_text().splitlines()]
    assert manifest[-1] == {"task_id": "t2", "root_session": "s2", "status": "overdue"}


def test_collect_lists_a_task_an_overdue_shard_never_reached(tmp_path):
    tasks = tmp_path / "tasks"
    tasks.mkdir()
    (tasks / "flask.jsonl").write_text("".join(json.dumps({"id": t, "model": "m", "meta": {"workspace": "flask"}}) + "\n" for t in ("a", "b")))
    shard = tmp_path / "run" / "shards" / "flask-000"
    shard.mkdir(parents=True)
    (shard / "manifest.jsonl").write_text(json.dumps({"task_id": "a", "root_session": "s1", "status": "overdue"}) + "\n" +
                                          json.dumps({"task_id": "b", "root_session": None, "status": "overdue"}) + "\n")
    (shard / "rows.jsonl").write_text(json.dumps({"root_session": "s1", "receipt": 1}) + "\n")
    _, stats = fleet.collect(tmp_path / "run", "engine-off", tasks)
    driven = [json.loads(line) for line in (tmp_path / "run" / "tasks-driven.jsonl").read_text().splitlines()]
    assert stats["rows"] == 1 and stats["tasks"] == 2
    assert [(t["id"], t["outcome"]["status"], t["outcome"]["root_session"]) for t in driven] == [("a", "overdue", "s1"), ("b", "overdue", None)]


def test_warm_fills_the_cache_for_repositories_and_stacks_that_name_packages(factory, monkeypatch, tmp_path):
    import dataclasses

    started = {}

    class Warm:
        def __init__(self, args, **kw):
            started[args[args.index("--name") + 1]] = args[-1]

        def wait(self):
            return 0

    monkeypatch.setattr(fleet.subprocess, "Popen", Warm)
    codes = fleet.warm_cache(dataclasses.replace(factory, root=tmp_path))
    assert set(codes) == {r.name for r in factory.repos} | {s.name for s in factory.stacks if s.warm}
    assert "bialy-warm-web-static" not in started
    assert started["bialy-warm-react-vite"].startswith("mkdir -p /work/react-vite && cd /work/react-vite && npm create vite")


def test_a_hosted_model_drives_at_its_configured_reasoning_effort(factory):
    import dataclasses

    glm = factory.model("glm-5.3-flash")
    quiet = dataclasses.replace(glm, hosted=dict(glm.hosted, reasoning_effort="none"))
    providers, *_ = fleet.sidecar_files(dataclasses.replace(factory, models=[quiet]), {}, 0)
    assert "models: [{id: accounts/fireworks/models/glm-5p3-flash, reasoning_effort: none}]" in providers
    providers, *_ = fleet.sidecar_files(factory, {}, 0)
    assert "models: [{id: accounts/fireworks/models/glm-5p3-flash}]" in providers
