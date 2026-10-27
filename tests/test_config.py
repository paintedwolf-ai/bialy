import pytest

from bialy import config


def test_every_model_and_repository_is_pinned(factory):
    assert {m.family for m in factory.models} >= {"qwen", "gemma"}
    for m in factory.models:
        assert len(m.revision) == 40 and m.license in config.PERMISSIVE
    for repo in factory.repos:
        assert len(repo.commit) == 40 and repo.spdx in config.PERMISSIVE
    assert {r.name for r in factory.repos if r.split == "holdout"} == {"zod", "sinatra", "ripgrep"}


def test_row_review_uses_the_first_configured_judge(factory):
    expected = factory.model(factory.run['judges'][0])
    assert factory.reviewer() == expected


def test_a_hosted_judge_needs_its_key(factory, monkeypatch):
    judge = next(m for m in factory.models if m.hosted)
    assert judge.ports() == [] and judge not in factory.served()
    monkeypatch.delenv(judge.hosted["key_env"], raising=False)
    with pytest.raises(config.ConfigError, match=judge.hosted["key_env"]):
        factory.chats(judge)


def test_unknown_names_are_refused(factory):
    with pytest.raises(config.ConfigError):
        factory.model("nope")
    with pytest.raises(config.ConfigError):
        factory.workspace("nope")


def test_hub_names_come_from_the_environment(monkeypatch):
    assert config.hub()["dataset_repo"].startswith("paintedwolfcode/")
    monkeypatch.setenv("BIALY_HF_DATASET_REPO", "someone/else")
    assert config.hub()["dataset_repo"] == "someone/else"


def test_stacks_are_workspaces_with_their_own_archetypes(factory):
    assert {s.name for s in factory.stacks if s.split == "holdout"} == {"vue-vite", "ruby-app", "java-maven"}
    assert factory.holdout() == {"zod", "sinatra", "ripgrep", "vue-vite", "ruby-app", "java-maven"}
    assert [w.name for w in factory.workspaces()] == [r.name for r in factory.repos] + [s.name for s in factory.stacks]
    assert factory.workspace("react-vite").kind == "stack" and factory.workspace("flask").kind == "repository"
    stack_archetypes = factory.archetypes_for(factory.workspace("react-vite"))
    assert {a.id for a in stack_archetypes} == {"new_project", "greenfield_auth", "prototype", "public_api_cli", "scaffold", "port"}
    assert all(a.new_files for a in stack_archetypes)
    assert not {a.id for a in stack_archetypes} & {a.id for a in factory.archetypes_for(factory.workspace("flask"))}
    assert all(factory.seeds[key] for key in ("domains", "scales"))


def copied_config(tmp_path, monkeypatch, edit):
    """config/ copied, one file edited by `edit(name, data)`, and loaded from there."""
    import shutil

    import yaml

    root = tmp_path / "config"
    shutil.copytree(config.CONFIG, root)
    monkeypatch.setattr(config, "CONFIG", root)
    for name in ("archetypes.yaml", "stacks.yaml", "models.yaml"):
        data = yaml.safe_load((root / name).read_text())
        edit(name, data)
        (root / name).write_text(yaml.safe_dump(data))


def test_a_stack_archetype_cannot_restrict_new_files(tmp_path, monkeypatch):
    def edit(name, data):
        if name == "archetypes.yaml":
            next(a for a in data["archetypes"] if a["id"] == "new_project")["new_files"] = False
    copied_config(tmp_path, monkeypatch, edit)
    with pytest.raises(config.ConfigError, match="every file is new"):
        config.load()


def test_a_workspace_name_is_unique_across_repositories_and_stacks(tmp_path, monkeypatch):
    def edit(name, data):
        if name == "stacks.yaml":
            data["stacks"][0]["name"] = "flask"
    copied_config(tmp_path, monkeypatch, edit)
    with pytest.raises(config.ConfigError, match="unique together"):
        config.load()


def test_every_kind_of_workspace_gets_requests(tmp_path, monkeypatch):
    def edit(name, data):
        if name == "archetypes.yaml":
            data["archetypes"] = [a for a in data["archetypes"] if a.get("workspace") != "stack"]
    copied_config(tmp_path, monkeypatch, edit)
    with pytest.raises(config.ConfigError, match="no archetype writes requests for a stack"):
        config.load()


def test_a_reasoning_effort_must_be_one_a_provider_takes(tmp_path, monkeypatch):
    def edit(name, data):
        if name == "models.yaml":
            next(m for m in data["models"] if m["id"] == "glm-5.3-flash")["hosted"]["reasoning_effort"] = "minimal"
    copied_config(tmp_path, monkeypatch, edit)
    with pytest.raises(config.ConfigError, match="reasoning_effort is one of none, low, medium, high"):
        config.load()


@pytest.mark.parametrize("codes", [None, "turn_closeout_tool_call", ["bad code"], [1], ["provider_request_rejected"]])
def test_task_failure_codes_require_structured_identifiers(codes):
    with pytest.raises(config.ConfigError, match="supported model failure codes"):
        config.run_settings({"fail_fast": {"task_failure_codes": codes}})


def test_task_local_failure_scope_requires_one_task_per_runner():
    with pytest.raises(config.ConfigError, match="tasks_per_runner: 1"):
        config.run_settings({"fail_fast": {"task_failure_codes": ["turn_closeout_tool_call"]}})
    assert config.run_settings({"tasks_per_runner": 1, "fail_fast": {"task_failure_codes": ["turn_closeout_tool_call"]}})["tasks_per_runner"] == 1


def test_same_family_judge_is_usable_when_it_is_the_configured_pool(factory):
    import dataclasses
    judge = next(model for model in factory.models if 'judge' in model.roles)
    limited = dataclasses.replace(factory, models=[judge], run={**factory.run, 'judges': [judge.id]})
    assert limited.reviewer() == judge


def scenario_entry(**changes):
    entry = {"id": "flaky-session-test", "workspace": "flask", "archetype": "bugfix",
             "variants": [{"prompt": "test_session_refresh fails 1 in 5 on CI, never locally", "follow_ups": ["why only on CI though"]},
                          {"prompt": "Can you look into an intermittent CI failure in the session tests?"}]}
    return {**entry, **changes}


def load_scenarios(factory, entries, variants=2):
    import dataclasses
    f = dataclasses.replace(factory, scenarios=[])
    config.load_scenarios(f, {"defaults": {"variants": variants}, "scenarios": entries})
    return f


def test_scenarios_load_with_their_variants(factory):
    f = load_scenarios(factory, [scenario_entry(lang="de")])
    [s] = f.scenarios
    assert (s.id, s.workspace, s.archetype, s.lang, f.scenario_variants) == ("flaky-session-test", "flask", "bugfix", "de", 2)
    assert s.variants[0]["follow_ups"] == ["why only on CI though"] and s.variants[1]["follow_ups"] == []


ANOTHER = {"prompt": "a different request about the same flaky session test"}


@pytest.mark.parametrize("entries, message", [
    ([scenario_entry(id="Flaky Test")], "unique id"),
    ([scenario_entry(), scenario_entry(variants=[ANOTHER, {"prompt": "yet another way to type that request"}])], "unique id"),
    ([scenario_entry(workspace="nowhere")], "archetype written for its kind of workspace"),
    ([scenario_entry(archetype="new_project")], "archetype written for its kind of workspace"),
    ([scenario_entry(lang="xx")], "lang is one of"),
    ([scenario_entry(variants=[ANOTHER])], "at least 2 variants, not 1"),
    ([scenario_entry(variants=[ANOTHER, {"prompt": "fix it"}])], "12 to 2400 characters"),
    ([scenario_entry(variants=[ANOTHER, {"prompt": "test_session_refresh fails on CI", "follow_ups": ["and why is that"] * 4}])],
     "at most 3 follow-ups"),
    ([scenario_entry(), scenario_entry(id="again")], "appears twice in flask"),
])
def test_a_scenario_must_be_well_formed(factory, entries, message):
    with pytest.raises(config.ConfigError, match=message):
        load_scenarios(factory, entries)


@pytest.mark.parametrize("variants", [None, 0, "10"])
def test_the_variant_count_is_a_positive_number(factory, variants):
    with pytest.raises(config.ConfigError, match="defaults.variants is a positive number"):
        load_scenarios(factory, [], variants=variants)
