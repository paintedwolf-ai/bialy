import json

import pytest

from bialy import config

FIXTURES = __import__("pathlib").Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def factory():
    return config.load()


def row(repo, group, n, host="coordinator", judge_model="glm-5.3-flash", **labels):
    """A minimal pw-decide-row/1 row, as decide export writes it plus factory metadata."""
    base = {"tools": [], "requests": [], "requested_names": [], "requested_groups": [], "skills": [], "kind": "inspect", "guides": {}}
    base.update(labels)
    return {
        "schema": "pw-decide-row/1", "receipt": n, "session": "s%d" % n, "root_session": "s%d" % n, "opening_message_id": "m%d" % n,
        "host": host, "surface": "implement_investigate", "catalog_revision": "rev", "project": repo, "model": "qwen3.6-35b-a3b",
        "partial": False, "lang": "en", "engine": {"state": "abstained", "reason": "engine disabled", "preloaded": [], "omitted": []},
        "offered": {"floor": ["read"], "loadable": ["edit", "command"], "guides": []},
        "state": {"host": host, "user": "request %d" % n, "surface": "implement_investigate"},
        "labels": base, "judge": {"model": judge_model},
        "meta": {"workspace": repo, "prompt_group": group, "task_id": "t%d" % n},
    }


@pytest.fixture
def split_dir(tmp_path):
    """train/val/holdout files that split correctly: zod is held out."""
    out = tmp_path / "split"
    out.mkdir()
    layout = {"train.jsonl": [("flask", "flask:a", 1), ("flask", "flask:a", 2), ("gin", "gin:b", 3)],
              "val.jsonl": [("gin", "gin:c", 4)],
              "holdout.jsonl": [("zod", "zod:d", 5)]}
    for name, rows in layout.items():
        (out / name).write_text("".join(json.dumps(row(*r)) + "\n" for r in rows))
    return out


@pytest.fixture
def driven_file(tmp_path):
    """tasks-driven.jsonl for the split fixture's five rows."""
    path = tmp_path / "tasks-driven.jsonl"
    path.write_text("".join(json.dumps({"id": "t%d" % n, "prompt": "request %d" % n, "outcome": {"status": "settled", "run": "r"}}) + "\n"
                            for n in range(1, 6)))
    return path


@pytest.fixture
def corpus_file(tmp_path):
    path = tmp_path / "corpus.json"
    path.write_text(json.dumps({"catalog_revision": "rev", "skills": [{"name": "commit", "card": "Commit in logical groups."}],
                                "tools": [{"name": "edit", "card": "Edit a file."}]}))
    return path


@pytest.fixture(autouse=True)
def anchors(tmp_path, monkeypatch):
    """Anchors a test's releases record land in the test's own directory."""
    from bialy import anchor

    monkeypatch.setattr(anchor, "ANCHORS", tmp_path / "releases")
    return tmp_path / "releases"


@pytest.fixture(autouse=True)
def license_texts(monkeypatch):
    """Release builds read license texts from seed clones the tests do not have."""
    from bialy import repos

    monkeypatch.setattr(repos, "license_text", lambda factory, repo, name: ("%s %s\n" % (repo.name, name)).encode())
