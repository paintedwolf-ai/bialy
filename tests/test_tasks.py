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


def test_new_files_may_be_named_in_directories_the_repository_has():
    files = ["src/flask/ctx.py", "tests/test_config.py", "pyproject.toml"]
    request = "Add src/flask/limits.py with a rate limiter, tests/test_limits.py, and a CHANGES.md entry."
    assert not tasks.mentioned_paths_exist(request, files)
    assert tasks.mentioned_paths_exist(request, files, new_files=True)
    # A file to create still has to sit somewhere the repository has.
    assert not tasks.mentioned_paths_exist("Put it in lib/vendor/limits.py", files, new_files=True)


def test_what_a_request_may_name_follows_its_workspace(factory):
    flask, stack = factory.workspace("flask"), factory.workspace("react-vite")
    archetype = {a.id: a for a in factory.archetypes}
    locate, small, new = archetype["locate"], archetype["small_change"], archetype["new_project"]
    assert "Do not invent non-existent file paths" in tasks.system_prompt(flask, locate)
    assert "may have any name, in a directory the facts show" in tasks.system_prompt(flask, small)
    assert "starting in an empty directory" in tasks.system_prompt(stack, new) and "facts" not in tasks.system_prompt(stack, new)


class Writer:
    """A generator that answers every batch with two requests naming new files."""
    calls = []

    def __init__(self, base_url, model, **kw):
        self.model = model

    def json(self, system, user, **kw):
        Writer.calls.append((system, user))
        if len(Writer.calls) == 1:
            return {"tasks": [{"prompt": "Make a snake game: index.html, src/main.ts, and src/snake.ts, arrow keys to steer.",
                               "follow_ups": ["Add a score."]},
                              {"prompt": "Build a tiny counter app in src/App.tsx with a reset button and tests for it.", "follow_ups": []}]}
        return {"tasks": [{"prompt": "I want a budget tracker where I can log expenses by category and see monthly totals.",
                           "follow_ups": []},
                          {"prompt": "Set up a weather page that calls a public API and shows a five day forecast as cards.",
                           "follow_ups": []}]}


def test_a_stack_request_starts_from_an_idea_and_an_empty_directory(factory, tmp_path, monkeypatch):
    import dataclasses

    from bialy import llm

    Writer.calls = []
    monkeypatch.setattr(llm, "Chat", Writer)
    monkeypatch.setenv("FIREWORKS_API_KEY", "test")
    stack = factory.workspace("react-vite")
    new_project = next(dataclasses.replace(a, per_workspace=12) for a in factory.archetypes if a.id == "new_project")
    scoped = dataclasses.replace(factory, repos=[], stacks=[stack], archetypes=[new_project])
    counts = tasks.generate(scoped, tmp_path / "repos", tmp_path / "tasks", seed=7, workers=1)
    assert counts == {"react-vite": 4}
    for system, user in Writer.calls:
        assert "empty directory" in system and "Do not invent" not in system
        assert "Stack: react-vite - React with TypeScript" in user and "Project idea:" in user and "Scale:" in user
    out = [json.loads(line) for line in (tmp_path / "tasks" / "react-vite.jsonl").read_text().splitlines()]
    for task in out:
        meta = task["meta"]
        assert (meta["workspace"], meta["workspace_kind"], meta["split"], meta["archetype"]) == ("react-vite", "stack", "train", "new_project")
        assert meta["seed"]["domain"] in factory.seeds["domains"] and meta["seed"]["scale"] in factory.seeds["scales"]
        assert meta["prompt_group"].startswith("react-vite:")
    # Requests naming the files they create are kept: the workspace is empty.
    assert any("src/main.ts" in t["prompt"] for t in out)


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
    assert {t["model"] for t in out} == {generators[0]} and out[0]["provider_id"] == factory.provider_id(factory.model(generators[0]))

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
