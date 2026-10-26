import json
from types import SimpleNamespace

import pytest
from test_dpo_qualification import captured_row

from bialy import custom_quality


def test_quality_review_binds_request_response_and_cross_family(tmp_path):
    row = captured_row()
    row["meta"]["model"] = "teacher"
    rows, transcripts, out = (tmp_path / name for name in ("rows", "transcripts", "quality"))
    rows.write_text(json.dumps(row) + "\n")
    transcripts.write_text(json.dumps({"session_id": "s1", "messages": []}) + "\n")
    calls = []

    class Chat:
        def turn(self, messages, **kwargs):
            calls.append(json.loads(messages[1]["content"]))
            return {"text": json.dumps({"verdict": "pass", "evidence_ids": ["validation-1"], "reason": "fixture assessment"})}

    factory = SimpleNamespace(model=lambda _: SimpleNamespace(id="teacher", family="gemma", revision="a"),
                              judge_for=lambda _: SimpleNamespace(id="judge", family="qwen", revision="b"),
                              chats=lambda _: [Chat()])
    assert custom_quality.judge_rows(factory, rows, transcripts, out) == {"pass": 1}
    quality = json.loads(out.read_text())
    assert quality["request_sha256"] == row["request_capture"]["sha256"]
    assert quality["response_sha256"] == row["validation"]["response_sha256"]
    assert calls[0]["response"] == row["target"]
    factory.judge_for = lambda _: SimpleNamespace(id="judge", family="gemma", revision="b")
    with pytest.raises(ValueError, match="another family"):
        custom_quality.judge_rows(factory, rows, transcripts, tmp_path / "same-family")
