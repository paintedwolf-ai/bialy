import pytest

from bialy import config


def test_every_model_and_repository_is_pinned(factory):
    assert {m.family for m in factory.models} >= {"qwen", "gemma"}
    for m in factory.models:
        assert len(m.revision) == 40 and m.license in config.PERMISSIVE
    for repo in factory.repos:
        assert len(repo.commit) == 40 and repo.spdx in config.PERMISSIVE
    assert {r.name for r in factory.repos if r.split == "holdout"} == {"zod", "sinatra", "ripgrep"}


def test_a_row_is_judged_by_another_family(factory):
    for m in factory.generators():
        assert factory.judge_for(m.id).family != m.family


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
        factory.repo("nope")


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
    assert {a.id for a in stack_archetypes} == {"new_project", "prototype", "scaffold", "port"}
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
    for name in ("archetypes.yaml", "stacks.yaml"):
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
