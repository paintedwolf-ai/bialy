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
