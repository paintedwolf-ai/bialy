import json

import pytest
import yaml

from bialy import tasks


def test_near_duplicate_prompts_share_a_group():
    prompts = [
        "Run the test suite for the router package and tell me which tests fail.",
        "Run the test suite for the router package and tell me which tests fail please.",
        "Explain how the context object is pushed and popped in ctx.py.",
    ]
    groups = tasks.group_prompts(prompts)
    assert groups[0] == groups[1] != groups[2]


def test_requests_naming_missing_files_are_dropped():
    files = ["src/flask/ctx.py", "tests/test_config.py", "pyproject.toml"]
    assert tasks.mentioned_paths_exist("What does src/flask/ctx.py do?", files)
    assert tasks.mentioned_paths_exist("Run tests/test_config.py and pyproject.toml checks", files)
    assert not tasks.mentioned_paths_exist("Fix the bug in src/flask/router.py", files)


def test_model_text_is_valid_utf8():
    assert tasks.clean("  bad \ud800 text ") == "bad ? text"


def test_rebase_skips_settled_tasks_and_moves_models_in_order(factory, tmp_path):
    src = tmp_path / "tasks"
    src.mkdir()
    lines = [{"id": "a", "model": "old-1", "provider_id": "x"}, {"id": "b", "model": "old-2", "provider_id": "x"},
             {"id": "c", "model": "old-1", "provider_id": "x", "workflow": "implement-dispatch", "workflow_version": "2.3.4"}]
    (src / "flask.jsonl").write_text("".join(json.dumps(t) + "\n" for t in lines))
    counts = tasks.rebase(factory, src, tmp_path / "rebased", {"b"})
    out = [json.loads(line) for line in (tmp_path / "rebased" / "flask.jsonl").read_text().splitlines()]
    generators = [m.id for m in factory.generators()]
    assert counts == {"flask": 2} and [t["id"] for t in out] == ["a", "c"]
    assert {t["model"] for t in out} == {generators[0]} and out[0]["provider_id"] == tasks.provider_id(generators[0])

    assert (out[1]["workflow"], out[1]["workflow_version"]) == ("implement-dispatch", "2.3.4")


@pytest.mark.parametrize("version", [None, "2.3.4"])
def test_worker_archetypes_start_their_workflow_on_their_share(factory, tmp_path, monkeypatch, version):
    manifest = yaml.safe_load((tasks.WORKFLOW_ROOT / "implement-dispatch" / "workflow.yaml").read_text())
    if version is not None:
        manifest["version"] = version
        root = tmp_path / "workflows"
        path = root / "implement-dispatch" / "workflow.yaml"
        path.parent.mkdir(parents=True)
        path.write_text(yaml.safe_dump(manifest))
        monkeypatch.setattr(tasks, "WORKFLOW_ROOT", root)
    repo = factory.repos[0]
    raw = [{"prompt": "Update module number %d and its tests, and document the change in the guide." % i,
            "follow_ups": [], "archetype": "multi_part", "lang": "en", "author": "m"} for i in range(200)]
    raw += [{"prompt": "Where is the setting number %d defined, and what reads it at startup?" % i,
             "follow_ups": [], "archetype": "locate", "lang": "en", "author": "m"} for i in range(50)]
    tasks.finalize(factory, repo, raw, [], tmp_path / "out.jsonl", seed=7)
    out = [json.loads(line) for line in (tmp_path / "out.jsonl").read_text().splitlines()]
    by = {a: [t for t in out if t["meta"]["archetype"] == a] for a in ("multi_part", "locate")}
    share = sum(t.get("workflow") == "implement-dispatch" for t in by["multi_part"]) / len(by["multi_part"])
    assert 0.5 < share < 0.9 and not any("workflow" in t for t in by["locate"])
    for task in out:
        if task.get("workflow"):
            assert task["workflow_version"] == manifest["version"]
            assert task["meta"]["workflow_version"] == manifest["version"]
        else:
            assert "workflow_version" not in task
    assert all(t["meta"]["workflow"] == t.get("workflow") for t in out) and not any("posture" in t for t in out)


def test_follow_ups_pass_the_same_gate_as_the_request(factory, tmp_path):
    files = ["src/flask/ctx.py"]
    raw = [{"prompt": "Explain what src/flask/ctx.py does with the app context.", "archetype": "question", "lang": "en", "author": "m",
            "follow_ups": ["Now fix the bug in src/flask/router.py.", "ok", "And how is it torn down in src/flask/ctx.py?"]}]
    factory_all = factory.__class__(**{**factory.__dict__, "archetypes": [
        a.__class__(**{**a.__dict__, "follow_up_rate": 1.0}) for a in factory.archetypes]})
    tasks.finalize(factory_all, factory.repos[0], raw, files, tmp_path / "out.jsonl", seed=7)
    [task] = [json.loads(line) for line in (tmp_path / "out.jsonl").read_text().splitlines()]
    assert [f["prompt"] for f in task["follow_ups"]] == ["And how is it torn down in src/flask/ctx.py?"]


@pytest.mark.parametrize("manifest", [
    {"id": "other", "version": "2.3.4"},
    {"id": "implement-dispatch"},
    {"id": "implement-dispatch", "version": ""},
    {"id": "implement-dispatch", "version": 2},
])
def test_finalize_requires_declared_workflow_identity(factory, tmp_path, monkeypatch, manifest):
    root = tmp_path / "workflows"
    path = root / "implement-dispatch" / "workflow.yaml"
    path.parent.mkdir(parents=True)
    path.write_text(yaml.safe_dump(manifest))
    monkeypatch.setattr(tasks, "WORKFLOW_ROOT", root)
    with pytest.raises(tasks.ConfigError, match="must declare"):
        tasks.finalize(factory, factory.repos[0], [], [], tmp_path / "out.jsonl", seed=7)
    assert not (tmp_path / "out.jsonl").exists()
