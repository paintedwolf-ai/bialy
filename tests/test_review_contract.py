import json
from types import SimpleNamespace

import pytest
import yaml
from test_dpo_qualification import captured_row

from bialy.campaign import review_contract, reviewer_controls
from bialy.campaign.grading import DIMENSIONS
from bialy.campaign.journal import CampaignFault, Journal, digest


def test_action_review_binds_request_response_and_any_reviewer(tmp_path):
    row = captured_row()
    row["meta"]["model"] = "teacher"
    rows, transcripts, out = (tmp_path / name for name in ("rows", "transcripts", "quality"))
    rows.write_text(json.dumps(row) + "\n")
    transcripts.write_text(json.dumps({"session_id": "s1", "messages": []}) + "\n")
    calls = []

    class Chat:
        def turn(self, messages, **kwargs):
            assert kwargs["schema"] == review_contract.ACTION.bound(["validation-1"])
            assert kwargs["max_tokens"] == 4096
            calls.append(json.loads(messages[1]["content"]))
            return {"text": json.dumps({"verdict": "pass", "evidence_ids": ["validation-1"], "reason": "fixture assessment"})}

    factory = SimpleNamespace(model=lambda _: SimpleNamespace(id="teacher", family="gemma", revision="a"),
                              reviewer=lambda: SimpleNamespace(id="judge", family="qwen", revision="b"),
                              chats=lambda _: [Chat()])
    assert review_contract.judge_rows(factory, rows, transcripts, out) == {"pass": 1}
    quality = json.loads(out.read_text())
    assert quality["request_sha256"] == row["request_capture"]["sha256"]
    assert quality["response_sha256"] == row["validation"]["response_sha256"]
    assert calls[0]["response"] == row["target"]
    assert next(iter(calls[0])) == "session"
    assert quality["review_input_sha256"] == digest(calls[0])
    factory.reviewer = lambda: SimpleNamespace(id="judge", family="gemma", revision="b")
    assert review_contract.judge_rows(factory, rows, transcripts, tmp_path / "same-family") == {"pass": 1}


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
        review_contract.judge_rows(factory, rows, transcripts, out)
    assert out.read_text() == ""

def test_a_rubric_limits_references_to_supplied_evidence():
    schema = review_contract.ACTION.bound(["receipt-a", "receipt-b", "receipt-a"])
    assert schema["properties"]["evidence_ids"] == {"type": "array", "minItems": 1,
                                                    "items": {"type": "string", "enum": ["receipt-a", "receipt-b"]}}
    assert "enum" not in review_contract.ACTION.schema["properties"]["evidence_ids"]["items"]
    model = SimpleNamespace(id="m", revision="r")
    assert review_contract.ACTION.identity(model, 7) != review_contract.EPISODE.identity(model, 7)


def test_a_settled_intent_is_never_dispatched_again(tmp_path):
    journal = Journal(tmp_path, {})
    intents, calls = review_contract.Intents(journal.db), []
    assert intents.call("one", lambda: calls.append(1) or {"answer": 1}) == {"answer": 1}
    assert intents.call("one", lambda: calls.append(1) or {"answer": 2}) == {"answer": 1}
    assert calls == [1] and intents.unknown("one") == 0
    journal.close()


def test_an_interrupted_call_is_recorded_unknown_before_its_retry(tmp_path):
    journal = Journal(tmp_path, {})
    intents = review_contract.Intents(journal.db)
    def interrupted():
        raise KeyboardInterrupt
    with pytest.raises(KeyboardInterrupt):
        intents.call("two", interrupted)
    assert intents.call("two", lambda: {"answer": "retried"}) == {"answer": "retried"}
    assert intents.unknown("two") == 1
    states = [row[0] for row in journal.db.execute("SELECT state FROM provider_intents WHERE identity='two' ORDER BY attempt")]
    assert states == ["unknown", "settled"]
    for _ in range(review_contract.ATTEMPTS - 1):
        with pytest.raises(KeyboardInterrupt):
            intents.call("three", interrupted)
    with pytest.raises(KeyboardInterrupt):
        intents.call("three", interrupted)
    with pytest.raises(CampaignFault, match="provider_outcome_unknown"):
        intents.call("three", lambda: {"answer": "never"})
    journal.close()


class Reviewer:
    """A reviewer model: `judgement` maps a control's task to the verdict it gives."""
    def __init__(self, judgement):
        self.judgement, self.calls = judgement, []

    def turn(self, messages, **kwargs):
        payload = json.loads(messages[1]["content"])
        self.calls.append(payload)
        verdict = self.judgement(payload["receipt"].get("task", ""))
        return {"text": json.dumps({"dimensions": dict.fromkeys(DIMENSIONS, verdict),
                                    "evidence_ids": payload["evidence_ids"][:1], "reason": "fixture"})}


class Factory:
    def __init__(self, models):
        self.models = models
    def model(self, name):
        return SimpleNamespace(id=name, revision="revision", roles=["judge"])
    def chats(self, model):
        return [self.models[model.id]]


def controls():
    episode = {"receipt": {"evidence_ids": ["e"]}, "operations": [], "transcript": []}
    return {"version": 1, "sha256": "c" * 64, "controls": [
        {"id": "good", "kind": "positive", "task": "good", **episode},
        {"id": "bad", "kind": "negative", "negative": "unverified_artifact", "task": "bad", **episode}]}


def test_a_slot_that_accepts_a_negative_or_fails_a_positive_never_reviews(tmp_path):
    journal = Journal(tmp_path, {})
    models = {"careful": Reviewer(lambda task: "pass" if task == "good" else "fail"),
              "gullible": Reviewer(lambda task: "pass"), "harsh": Reviewer(lambda task: "fail")}
    reviewers = review_contract.Reviewers(journal, Factory(models), ["careful", "gullible", "harsh"])
    results = reviewer_controls.qualify(reviewers, controls())
    assert {slot: result["qualified"] for slot, result in results.items()} == {"0": True, "1": False, "2": False}
    assert results["1"]["accepted_negatives"] == ["bad"]
    calls = sum(len(model.calls) for model in models.values())
    assert reviewer_controls.qualify(reviewers, controls()) == results
    assert sum(len(model.calls) for model in models.values()) == calls
    reviews = reviewers.review({"evidence_ids": ["e"], "task": "good"}, [])
    assert [review["judge_id"] for review in reviews] == ["careful"]
    assert reviews[0]["controls_sha256"] == "c" * 64 and "judge_family" not in reviews[0]
    journal.close()


def test_no_qualified_slot_stops_the_campaign(tmp_path):
    journal = Journal(tmp_path, {})
    reviewers = review_contract.Reviewers(journal, Factory({"gullible": Reviewer(lambda task: "pass")}), ["gullible"])
    with pytest.raises(CampaignFault, match="reviewers_unqualified"):
        reviewer_controls.qualify(reviewers, controls())
    with pytest.raises(CampaignFault, match="no_qualified_episode_reviewer"):
        reviewers.review({"evidence_ids": ["e"]}, [])
    journal.close()


def test_the_configured_controls_cover_every_required_negative(tmp_path):
    loaded = reviewer_controls.load()
    negatives = {item["negative"] for item in loaded["controls"] if item["kind"] == "negative"}
    assert negatives == {"unverified_artifact", "read_or_write_cited_as_execution", "ignored_failed_check",
                         "duplicate_append", "unfinished_workers", "host_replaced_closeout"}
    payload = reviewer_controls.payload(loaded["controls"][0])
    assert payload["evidence_ids"] == loaded["controls"][0]["receipt"]["evidence_ids"] and payload["transcript"][0]["messages"]
    value = yaml.safe_load(reviewer_controls.CONTROLS.read_text())
    value["controls"] = [item for item in value["controls"] if item.get("negative") != "duplicate_append"]
    (tmp_path / "controls.yaml").write_text(yaml.safe_dump(value))
    with pytest.raises(CampaignFault, match="reviewer_controls_incomplete"):
        reviewer_controls.load(tmp_path / "controls.yaml")
