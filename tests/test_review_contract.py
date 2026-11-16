import json
from types import SimpleNamespace

import pytest
import yaml
from test_training_rows import call, candidate

from bialy import training_rows
from bialy.campaign import review_contract, reviewer_controls
from bialy.campaign.grading import DIMENSIONS
from bialy.campaign.journal import CampaignFault, Journal, digest


def eligible_row():
    """An eligible training row from session s1 whose call ran operation op-1."""
    item = candidate(calls=[call(operation_ids=['op-1'])], session='s1')
    item['execution']['receipt_id'] = 'validation-1'
    item['request_capture']['messages'] = [{'role': 'user', 'content': 'Read the file.'}]
    target = item['target']
    return {**item, 'schema': 'pw-training-row/1', 'response_sha256': digest(target), 'meta': {'model': 'teacher'},
            'admission': {'eligibility': 'eligible', 'category': 'ordinary', 'semantic_outcome': 'success'}}


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
    """A reviewer model: `episode` maps a control's task to its verdict on an episode, and
    `action` maps a response's text to its verdict on a training row."""
    def __init__(self, episode, action=None):
        self.episode, self.action, self.calls = episode, action or (lambda text: "fail"), []

    def turn(self, messages, **kwargs):
        payload = json.loads(messages[1]["content"])
        self.calls.append(payload)
        answer = {"evidence_ids": payload["evidence_ids"][:1], "reason": "fixture"}
        if messages[0]["content"] == review_contract.EPISODE.system:
            return {"text": json.dumps({**answer, "dimensions": dict.fromkeys(DIMENSIONS, self.episode(payload["receipt"].get("task", "")))})}
        assert messages[0]["content"] == review_contract.ACTION.system and kwargs["schema"] == review_contract.ACTION.bound(payload["evidence_ids"])
        return {"text": json.dumps({**answer, "verdict": self.action(payload["response"]["text"])})}


class Factory:
    def __init__(self, models):
        self.models = models
    def model(self, name):
        return SimpleNamespace(id=name, revision="revision", roles=["judge"])
    def chats(self, model):
        return [self.models[model.id]]


def controls():
    episode = {"receipt": {"evidence_ids": ["e"]}, "operations": [], "transcript": []}
    step = {"category": "closeout", "request": [{"role": "user", "content": "task"}], "tools": ["read"], "calls": []}
    return {"episode": {"version": 2, "sha256": "c" * 64, "controls": [
                {"id": "good", "kind": "positive", "task": "good", **episode},
                {"id": "bad", "kind": "negative", "negative": "unverified_artifact", "task": "bad", **episode}]},
            "action": {"version": 2, "sha256": "a" * 64, "controls": [
                {"id": "good", "kind": "positive", "response": {"text": "good"}, **step},
                {"id": "bad", "kind": "negative", "negative": "unsupported_success_claim", "response": {"text": "bad"}, **step}]}}


def careful(text):
    return "pass" if text == "good" else "fail"


def test_a_slot_fills_only_the_roles_whose_controls_it_passed(tmp_path):
    journal = Journal(tmp_path, {})
    models = {"episodes-only": Reviewer(careful, lambda text: "pass"), "gullible": Reviewer(lambda task: "pass", lambda text: "pass"),
              "rows-only": Reviewer(lambda task: "fail", careful), "both": Reviewer(careful, careful)}
    reviewers = review_contract.Reviewers(journal, Factory(models), list(models))
    results = reviewer_controls.qualify(reviewers, controls())
    assert {role: [slot for slot, result in slots.items() if result["qualified"]] for role, slots in results.items()} == \
        {"episode": ["0", "3"], "action": ["2", "3"]}
    assert results["action"]["0"]["accepted_negatives"] == ["bad"] and results["episode"]["1"]["accepted_negatives"] == ["bad"]
    calls = sum(len(model.calls) for model in models.values())
    assert reviewer_controls.qualify(reviewers, controls()) == results
    assert sum(len(model.calls) for model in models.values()) == calls
    reviews = reviewers.review({"evidence_ids": ["e"], "task": "good"}, [])
    assert [review["judge_id"] for review in reviews] == ["episodes-only", "both"]
    assert reviews[0]["controls_sha256"] == "c" * 64 and reviews[0]["role"] == "episode" and "judge_family" not in reviews[0]
    row = eligible_row()
    row["target"]["text"] = "good"
    row["response_sha256"] = digest(row["target"])
    quality = reviewers.judge(row)
    assert quality["judge_model"] == "rows-only" and quality["reviewer_slot"] == "2" and quality["role"] == "action"
    assert quality["controls_passed"] is True and quality["controls_sha256"] == "a" * 64 and quality["verdict"] == "pass"
    assert training_rows.training_ready({**row, "quality": quality})
    assert all(call.get("request") != row["request_capture"] for name in ("episodes-only", "gullible") for call in models[name].calls)
    journal.close()


def test_a_row_judge_sees_the_request_the_response_and_its_own_calls_only(tmp_path):
    journal = Journal(tmp_path, {})
    judge = Reviewer(careful, careful)
    reviewers = review_contract.Reviewers(journal, Factory({"judge": judge}), ["judge"])
    reviewer_controls.qualify(reviewers, controls())
    row = eligible_row()
    quality = reviewers.judge(row)
    seen = judge.calls[-1]
    assert set(seen) == {"request", "response", "execution", "admission", "evidence_ids"}
    assert seen["request"] == row["request_capture"] and seen["response"] == row["target"] and seen["execution"] == row["execution"]
    assert seen["evidence_ids"] == ["validation-1", "op-1"]
    assert quality["verdict"] == "fail" and quality["request_sha256"] == row["request_capture"]["sha256"]
    assert quality["response_sha256"] == row["response_sha256"] and quality["review_input_sha256"] == digest(seen)
    assert quality["method"] == "model_judge" and quality["judge_revision"] == "revision"
    calls = len(judge.calls)
    assert reviewers.judge(row) == quality and len(judge.calls) == calls
    for changed in ({**row, "response_sha256": "0" * 64}, {**row, "admission": {"eligibility": "excluded", "reason": "unknown_effects"}}):
        with pytest.raises(CampaignFault, match="response_review_row_invalid"):
            reviewers.judge(changed)
    journal.close()


@pytest.mark.parametrize("verdict,code", [
    ([], "action_judge_invalid_evidence"), ("not json", "action_judge_invalid_response"),
    ({"verdict": "pass", "evidence_ids": [], "reason": "unsupported"}, "action_judge_invalid_evidence"),
    ({"verdict": "pass", "evidence_ids": ["invented"], "reason": "unsupported"}, "action_judge_invalid_evidence"),
    ({"verdict": "maybe", "evidence_ids": ["validation-1"], "reason": "uncertain"}, "action_judge_invalid_evidence"),
])
def test_an_action_answer_that_is_not_an_evidence_bound_verdict_is_refused(verdict, code):
    chat = SimpleNamespace(turn=lambda *args, **kwargs: {"text": verdict if isinstance(verdict, str) else json.dumps(verdict)})
    with pytest.raises(CampaignFault, match=code):
        review_contract.ask_action(chat, review_contract.action_payload(eligible_row()), 7)
    with pytest.raises(CampaignFault, match="action_judge_tool_call"):
        review_contract.ask_action(SimpleNamespace(turn=lambda *args, **kwargs: {"tool_calls": [{}]}),
                                   review_contract.action_payload(eligible_row()), 7)


def test_a_role_without_a_qualified_slot_stops_the_campaign(tmp_path):
    journal = Journal(tmp_path, {})
    reviewers = review_contract.Reviewers(journal, Factory({"episodes-only": Reviewer(careful, lambda text: "pass")}), ["episodes-only"])
    with pytest.raises(CampaignFault, match="reviewers_unqualified") as stopped:
        reviewer_controls.qualify(reviewers, controls())
    assert stopped.value.details["roles"] == ["action"]
    assert reviewers.review({"evidence_ids": ["e"], "task": "good"}, [])
    with pytest.raises(CampaignFault, match="no_qualified_action_reviewer"):
        reviewers.judge(eligible_row())
    unqualified = review_contract.Reviewers(Journal(tmp_path / "other", {}), Factory({"gullible": Reviewer(lambda task: "pass")}), ["gullible"])
    with pytest.raises(CampaignFault, match="no_qualified_episode_reviewer"):
        unqualified.review({"evidence_ids": ["e"]}, [])
    journal.close()


def test_the_configured_controls_cover_every_required_negative_of_each_role(tmp_path):
    loaded = reviewer_controls.load()
    negatives = {role: {item["negative"] for item in loaded[role]["controls"] if item["kind"] == "negative"} for role in loaded}
    assert negatives == {"episode": {"unverified_artifact", "read_or_write_cited_as_execution", "ignored_failed_check",
                                     "duplicate_append", "unfinished_workers", "host_replaced_closeout"},
                         "action": {"unsupported_success_claim", "ignored_visible_failure", "unrequested_effect",
                                    "repeated_failed_call", "ungrounded_arguments"}}
    assert loaded["episode"]["sha256"] != loaded["action"]["sha256"]
    payload = reviewer_controls.episode_control(loaded["episode"]["controls"][0])
    assert payload["evidence_ids"] == loaded["episode"]["controls"][0]["receipt"]["evidence_ids"] and payload["transcript"][0]["messages"]
    for control in loaded["action"]["controls"]:
        payload = reviewer_controls.action_control(control)
        assert set(payload) == {"request", "response", "execution", "admission", "evidence_ids"} and payload["request"]["messages"]
        assert payload["evidence_ids"][0] == "out-" + control["id"] and payload["admission"]["category"] in training_rows.CATEGORIES
        offered = {tool["function"]["name"] for tool in payload["request"]["tools"]}
        assert {item["name"] for item in payload["response"]["tool_calls"]} <= offered
        assert [item["call_id"] for item in payload["execution"]["calls"]] == [item["id"] for item in payload["response"]["tool_calls"]]
    value = yaml.safe_load(reviewer_controls.CONTROLS.read_text())
    value["action"]["controls"] = [item for item in value["action"]["controls"] if item.get("negative") != "repeated_failed_call"]
    (tmp_path / "controls.yaml").write_text(yaml.safe_dump(value))
    with pytest.raises(CampaignFault, match="reviewer_controls_incomplete") as incomplete:
        reviewer_controls.load(tmp_path / "controls.yaml")
    assert incomplete.value.details == {"role": "action", "missing_negatives": ["repeated_failed_call"]}
    del value["action"]
    (tmp_path / "controls.yaml").write_text(yaml.safe_dump(value))
    with pytest.raises(CampaignFault, match="reviewer_controls_incomplete"):
        reviewer_controls.load(tmp_path / "controls.yaml")
