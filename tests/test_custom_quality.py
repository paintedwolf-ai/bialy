import json
from types import SimpleNamespace

import pytest
from test_dpo_qualification import captured_row

from bialy import custom_quality


def test_quality_review_binds_request_response_and_allows_same_family(tmp_path):
    row = captured_row()
    row["meta"]["model"] = "teacher"
    rows, transcripts, out = (tmp_path / name for name in ("rows", "transcripts", "quality"))
    rows.write_text(json.dumps(row) + "\n")
    transcripts.write_text(json.dumps({"session_id": "s1", "messages": []}) + "\n")
    calls = []

    class Chat:
        def turn(self, messages, **kwargs):
            assert kwargs["schema"] == custom_quality.REVIEW_SCHEMA
            assert kwargs["max_tokens"] == 4096
            calls.append(json.loads(messages[1]["content"]))
            return {"text": json.dumps({"verdict": "pass", "evidence_ids": ["validation-1"], "reason": "fixture assessment"})}

    factory = SimpleNamespace(model=lambda _: SimpleNamespace(id="teacher", family="gemma", revision="a"),
                              reviewer=lambda: SimpleNamespace(id="judge", family="qwen", revision="b"),
                              chats=lambda _: [Chat()])
    assert custom_quality.judge_rows(factory, rows, transcripts, out) == {"pass": 1}
    quality = json.loads(out.read_text())
    assert quality["request_sha256"] == row["request_capture"]["sha256"]
    assert quality["response_sha256"] == row["validation"]["response_sha256"]
    assert calls[0]["response"] == row["target"]
    assert next(iter(calls[0])) == "session"
    assert quality["review_input_sha256"] == custom_quality.digest(calls[0])
    factory.reviewer = lambda: SimpleNamespace(id="judge", family="gemma", revision="b")
    assert custom_quality.judge_rows(factory, rows, transcripts, tmp_path / "same-family") == {"pass": 1}


@pytest.mark.parametrize("verdict", [
    [], {"verdict": "pass", "evidence_ids": [], "reason": "unsupported"},
    {"verdict": "pass", "evidence_ids": ["invented"], "reason": "unsupported"},
    {"verdict": "maybe", "evidence_ids": ["validation-1"], "reason": "uncertain"},
])
def test_quality_review_rejects_invalid_verdicts(tmp_path, verdict):
    row = captured_row()
    row["meta"]["model"] = "teacher"
    rows, transcripts, out = (tmp_path / name for name in ("rows", "transcripts", "quality"))
    rows.write_text(json.dumps(row) + "\n")
    transcripts.write_text(json.dumps({"session_id": "s1", "messages": []}) + "\n")
    model = SimpleNamespace(id="judge", family="gemma", revision="a")
    chat = SimpleNamespace(turn=lambda *args, **kwargs: {"text": json.dumps(verdict)})
    factory = SimpleNamespace(model=lambda _: model, reviewer=lambda: model, chats=lambda _: [chat])
    with pytest.raises(ValueError, match="quality judge returned"):
        custom_quality.judge_rows(factory, rows, transcripts, out)
    assert out.read_text() == ""
