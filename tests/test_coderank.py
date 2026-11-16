import dataclasses

from bialy import coderank
from bialy.campaign import ancestry

UNIT = {"repo": "flask", "file": "src/flask/sessions.py", "line": 10, "symbol": "SecureCookieSessionInterface",
        "kind": "class", "lang": "python", "code": "class SecureCookieSessionInterface: ...",
        "doc": "Stores the session in a signed cookie so a client cannot tamper with it."}


def test_a_request_that_names_the_unit_or_its_file_is_refused():
    assert coderank.mentions_identifier("where is the secure cookie code", UNIT)
    assert coderank.mentions_identifier("open sessions", UNIT)
    assert not coderank.mentions_identifier("how do we stop clients editing their login state", UNIT)


def test_doc_pairs_take_the_first_prose_sentence_that_names_nothing():
    kept = dict(UNIT, doc="Signs what the client keeps so nobody can quietly edit it. Second sentence here.")
    rows = coderank.doc_pairs([UNIT, kept, dict(UNIT, doc="Too short here.")])
    assert [r["task"] for r in rows] == ["Signs what the client keeps so nobody can quietly edit it."]
    assert rows[0]["source"] == "docs" and rows[0]["lang"] == "en"


class FakeChat:
    model = "gemma"

    def __init__(self, replies):
        self.replies = replies

    def json(self, system, user, **kw):
        assert kw["schema"] == coderank.REPLY_SCHEMA and kw["thinking"] is False
        return self.replies.pop(0)


def test_model_pairs_keep_requests_that_name_the_unit_apart(factory, monkeypatch):
    chat = FakeChat([{"task": "how do we stop clients editing their login state", "query": "signed cookie"},
                     {"task": "where is the secure cookie interface", "query": "cookie"}])
    monkeypatch.setattr(coderank, "endpoints", lambda f, w=None: [chat])
    rows, named = coderank.model_pairs(factory, [UNIT, dict(UNIT, line=20)], count=2, workers=1)
    assert [r["task"] for r in rows] == ["how do we stop clients editing their login state"]
    assert [r["task"] for r in named] == ["where is the secure cookie interface"] and named[0]["names_unit"]
    assert rows[0]["source"] == "model:gemma" and rows[0]["lang"] in coderank.LANGUAGES


def test_screened_replies_are_replaced_until_the_count_is_met(factory, monkeypatch):
    replies = [{"task": "where is the secure cookie thing", "query": "q"}, {"task": "how do we keep clients from editing state", "query": "q"},
               {"task": "session handling please", "query": "q"}, {"task": "what signs the data a browser holds", "query": "q"}]
    chat = FakeChat(list(replies))
    monkeypatch.setattr(coderank, "endpoints", lambda f, w=None: [chat])
    units = [dict(UNIT, line=n) for n in range(4)]
    rows, named = coderank.model_pairs(factory, units, count=2, workers=1)
    assert [r["task"] for r in rows] == ["how do we keep clients from editing state", "what signs the data a browser holds"]
    assert [r["task"] for r in named] == ["where is the secure cookie thing", "session handling please"]


def partitioned(factory, tmp_path):
    """Four of the factory's repositories, one per partition, and a fifth the ancestry does not hold."""
    repos = factory.repos[:5]
    graph = ancestry.Graph()
    for repo, partition in zip(repos, ancestry.PARTITIONS, strict=False):
        graph.add("repository:" + repo.name, "repository", source_class="synthetic", holdout=partition)
    return dataclasses.replace(factory, repos=repos, root=tmp_path), ancestry.allocate(graph, 7, {"train": 1.0}), repos


def test_dumps_exist_only_for_train_and_validation_repositories(factory, tmp_path, monkeypatch):
    scaled, assignment, repos = partitioned(factory, tmp_path)
    by_partition = coderank.repositories(scaled, assignment)
    assert {p: [r.name for r in rs] for p, rs in by_partition.items()} == {p: [r.name] for p, r in zip(ancestry.PARTITIONS, repos, strict=False)}
    calls = []

    def run(args, **kw):
        calls.append(args)
        open(args[args.index("--dump") + 1], "w").write("{}\n")
        return type("R", (), {"returncode": 0, "stdout": "ok", "stderr": ""})()
    monkeypatch.setattr(coderank.subprocess, "run", run)
    stale = tmp_path / "dumps" / ("summarize_definitions-docs-%s.jsonl" % repos[3].name)
    stale.parent.mkdir()
    stale.write_text("{}\n")
    written = coderank.dumps(scaled, "decide-rerank", tmp_path / "units", tmp_path / "pairs", tmp_path / "dumps", assignment)
    assert set(written) == {"train", "validation"} and all(len(paths) == len(coderank.SITES) * 2 for paths in written.values())
    assert all(repos[0].name in path.name and path.parent.name == "train" for path in written["train"])
    assert all(repos[1].name in path.name and path.parent.name == "validation" for path in written["validation"])
    dumped = {args[args.index("--name") + 1] for args in calls}
    assert dumped == {repos[0].name, repos[1].name}
    # Nothing of a development, held-out or unassigned repository is on disk for a trainer to glob.
    assert not stale.exists()
    assert not [p for p in (tmp_path / "dumps").rglob("*.jsonl") if any(r.name in p.name for r in repos[2:])]
    assert coderank.training_dumps(tmp_path / "dumps") == written
