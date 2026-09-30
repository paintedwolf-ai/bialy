from bialy import coderank

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


def test_model_pairs_drop_requests_that_name_the_unit(factory, monkeypatch):
    chat = FakeChat([{"task": "how do we stop clients editing their login state", "query": "signed cookie"},
                     {"task": "where is the secure cookie interface", "query": "cookie"}])
    monkeypatch.setattr(coderank, "endpoints", lambda f: [chat])
    rows = coderank.model_pairs(factory, [UNIT, dict(UNIT, line=20)], count=2, workers=1)
    assert [r["task"] for r in rows] == ["how do we stop clients editing their login state"]
    assert rows[0]["source"] == "model:gemma" and rows[0]["lang"] in coderank.LANGUAGES
