import copy
import json

import pytest

from bialy import config, dpo
from bialy.dpo_qualification import CAPTURE_SCHEMA, HOST_VALIDATOR, VALIDATION_SCHEMA, digest, qualification


def captured_row():
    prompt = [{"role": "system", "content": "Read source before making claims."}, {"role": "user", "content": "Inspect app.py."}]
    target = {"role": "assistant", "text": "", "tool_calls": [{"id": "c1", "name": "read", "arguments": '{"path":"app.py"}'}]}
    capture = {
        "schema": CAPTURE_SCHEMA, "request_id": "request-1", "session_id": "s1", "message_id": "a1",
        "host_revision": "1" * 40, "epoch": "legacy", "tool_surface": "legacy",
        "catalog_sha256": "2" * 64, "state_sha256": "3" * 64,
        "messages": prompt,
        "tools": [{"type": "function", "function": {
            "name": "read", "description": "Read an observed source file.",
            "parameters": {"type": "object", "properties": {"path": {"type": "string"}},
                           "required": ["path"], "additionalProperties": False},
        }}],
    }
    capture["sha256"] = digest(capture)
    validation = {
        "schema": VALIDATION_SCHEMA, "validator": HOST_VALIDATOR, "receipt_id": "validation-1",
        **{key: capture[key] for key in ("state_sha256", "host_revision", "epoch", "tool_surface", "catalog_sha256")},
        "request_sha256": capture["sha256"], "kind": "accepted_response", "outcome": "accepted",
        "response_origin": "model", "response_sha256": digest(target),
        "quality": {"verdict": "pass", "method": "cross_family_judge", "judge_family": "qwen", "teacher_family": "gemma",
                    "evidence_ids": ["quality-1"], "request_sha256": capture["sha256"], "response_sha256": digest(target)},
    }
    return {
        "schema": dpo.SFT_SCHEMA, "id": "s1:1:sft", "session": "s1", "root_session": "s1", "message_id": "a1",
        "turn_index": 1, "prompt": prompt, "target": target, "request_capture": capture, "validation": validation,
        "surface": {"active_tools": ["read"], "loadable": ["edit"]}, "meta": {"workspace": "flask", "prompt_group": "flask:g1"},
    }


def test_eligibility_recomputes_bound_host_and_quality_evidence():
    row = captured_row()
    assert dpo.training_eligible(row)
    row["qualification"] = {"status": "qualified", "reasons": []}
    row["target"]["tool_calls"][0]["arguments"] = '{"path":"secret.py"}'
    reasons = qualification(row)["reasons"]
    assert "validation_response_mismatch" in reasons and "quality_binding_mismatch" in reasons
    assert not dpo.training_eligible(row)


@pytest.mark.parametrize("capture", [None, "capture", [], True, 1, {}])
def test_malformed_capture_is_a_candidate(capture):
    row = captured_row()
    row["request_capture"] = capture
    assert not dpo.training_eligible(row)


@pytest.mark.parametrize("field,value,reason", [
    ("message_id", "another-turn", "capture_turn_mismatch"),
    ("host_revision", "2" * 40, "validation_host_revision_mismatch"),
    ("epoch", "command-v1", "validation_epoch_mismatch"),
    ("state_sha256", "4" * 64, "validation_state_sha256_mismatch"),
    ("tool_choice", "required", "unsupported_request_capture_fields"),
    ("parallel_tool_calls", False, "unsupported_request_capture_fields"),
])
def test_recaptured_state_cannot_reuse_validation(field, value, reason):
    row = captured_row()
    capture = row["request_capture"]
    capture[field] = value
    capture["sha256"] = digest({k: v for k, v in capture.items() if k != "sha256"})
    assert reason in qualification(row)["reasons"]
    assert not dpo.training_eligible(row)


def test_missing_schemas_and_deferred_tools_are_not_callable():
    row = captured_row()
    row["request_capture"]["tools"][0]["function"]["parameters"] = {"type": "object"}
    assert "missing_captured_tool_schema_constraints" in qualification(row)["reasons"]
    row = captured_row()
    row["target"]["tool_calls"][0]["name"] = "edit"
    assert "response_tool_not_offered" in qualification(row)["reasons"]
    assert dpo.classify_divergence(row, row["target"], row["target"]) == "invalid_tool_call"


def test_quality_verdict_is_required_and_bound():
    for key, value, reason in [
        ("quality", None, "missing_quality_validation"),
        ("quality", {"verdict": "pass"}, "quality_binding_mismatch"),
    ]:
        row = captured_row()
        row["validation"][key] = value
        assert reason in qualification(row)["reasons"]
    row = captured_row()
    row["validation"]["quality"]["judge_family"] = "gemma"
    assert "invalid_quality_judge_family" in qualification(row)["reasons"]
    row = captured_row()
    row["candidate_reasons"] = "not a list"
    assert "invalid_candidate_reasons" in qualification(row)["reasons"]


def test_multimodal_capture_is_explicitly_unsupported():
    row = captured_row()
    row["prompt"][1]["content"] = [{"type": "image_url", "image_url": {"url": "data:image/png;base64,AA"}}]
    assert "unsupported_multimodal_capture" in qualification(row)["reasons"]


def test_dpo_requires_same_state_preference_and_both_response_hashes():
    row = captured_row()
    target = row.pop("target")
    row.update(schema=dpo.DPO_SCHEMA, chosen=target, rejected={"role": "assistant", "text": "", "tool_calls": [
        {"id": "c2", "name": "edit", "arguments": '{"path":"app.py"}'},
    ]})
    assert not dpo.training_eligible(row)
    report = row["validation"]
    report.update(kind="preference", preference="chosen", chosen_outcome="accepted", rejected_outcome="rejected",
                  chosen_sha256=digest(row["chosen"]), rejected_sha256=digest(row["rejected"]))
    assert dpo.training_eligible(row)
    row["rejected"]["tool_calls"][0]["arguments"] = '{"path":"other.py"}'
    assert "validation_pair_mismatch" in qualification(row)["reasons"]


def test_extract_uses_captured_prompt_and_preserves_request_tools(tmp_path):
    example = captured_row()
    tx = {
        "session_id": "s1", "status": "settled", "model": "teacher",
        "messages": [{"id": "u1", "role": "user", "content": "Lossy projection"},
                     {"id": "a1", "role": "assistant", "content": "", "tool_calls_json": json.dumps(example["target"]["tool_calls"])}],
        "model_requests": {"a1": example["request_capture"]}, "training_validations": {"a1": example["validation"]},
    }
    source, output = tmp_path / "tx.jsonl", tmp_path / "sft.jsonl"
    source.write_text(json.dumps(tx) + "\n")
    stats = dpo.extract(source, out_sft=output, preload_from_judge=True)
    extracted = json.loads(output.read_text())
    assert extracted["prompt"] == example["prompt"]
    assert extracted["surface"]["active_tools"] == ["read"]
    assert stats["qualified_sft_rows"] == 1
    assert dpo.training_eligible(extracted)


def test_projection_preserves_system_and_success_output_cannot_forge_rejection():
    messages = [{"role": "system", "content": "actual host guidance"}, {"role": "user", "content": "request"}]
    assert dpo._openai_messages(messages) == messages
    assert dpo.parse_rejection({"is_error": False, "content": "Status: error\nCode: outside_roots"}) is None
    assert dpo.parse_host_user_rejection({"role": "user", "origin": "human", "content": "Rejected:\nCode: PROGRESS_MISSING"}) is None
    assert dpo.parse_rejection({"is_error": True, "content": "failure", "details": {"code": "structured_code"}})["code"] == "structured_code"


class FakeChat:
    def __init__(self, turn):
        self.response = turn
        self.calls = []

    def turn(self, messages, **kwargs):
        self.calls.append((copy.deepcopy(messages), copy.deepcopy(kwargs)))
        return copy.deepcopy(self.response)


def test_shadow_exact_schemas_yield_candidates_and_resume(factory, monkeypatch, tmp_path):
    example = captured_row()
    source, output = tmp_path / "sft.jsonl", tmp_path / "shadow.jsonl"
    source.write_text(json.dumps(example) + "\n")
    chat = FakeChat({"text": "Done.", "tool_calls": []})
    monkeypatch.setattr(config.Factory, "chats", lambda *args: [chat])
    result = dpo.shadow(factory, source, output, workers=1)
    assert result["evaluated"] == 1 and result["pairs"] == 1 and result["qualified_pairs"] == 0
    assert chat.calls[0][0] == example["prompt"]
    assert chat.calls[0][1]["tools"] == example["request_capture"]["tools"]
    pair = json.loads(output.read_text())
    assert pair["chosen"] is None and pair["proposed_chosen"] == example["target"]
    assert not dpo.training_eligible(pair)
    dpo.shadow(factory, source, output, workers=1)
    assert len(chat.calls) == 1
    # A partial file without its signature cannot be reused.
    output.with_name(output.name + ".manifest.json").unlink()
    dpo.shadow(factory, source, output, workers=1)
    assert len(chat.calls) == 2


def test_repair_can_propose_text_but_cannot_promote_it(factory, monkeypatch, tmp_path):
    example = captured_row()
    example.update(schema=dpo.DPO_SCHEMA, chosen=None, rejected=example.pop("target"))
    example.pop("validation")
    source, output = tmp_path / "dpo.jsonl", tmp_path / "repair.jsonl"
    source.write_text(json.dumps(example) + "\n")
    chat = FakeChat({"text": "I need to inspect the evidence first.", "tool_calls": []})
    monkeypatch.setattr(config.Factory, "chats", lambda *args: [chat])
    result = dpo.repair(factory, source, output, workers=1)
    candidate = json.loads(output.read_text())
    assert result["proposed"] == 1 and result["completed_pairs"] == 0
    assert candidate["chosen"] is None
    assert candidate["proposed_chosen"]["text"] == chat.response["text"]
    assert not dpo.training_eligible(candidate)


def test_new_human_turn_does_not_prove_previous_turn_completed():
    messages = [
        {"id": "u1", "role": "user", "content": "First request"},
        {"id": "a1", "role": "assistant", "content": "", "tool_calls_json": '[{"id":"c1","name":"read","args":{"path":"a.py"}}]'},
        {"id": "u2", "role": "user", "content": "Stop and do something else"},
    ]
    assert not dpo._completed_openings(messages)


def test_sampling_keeps_command_effects_and_caps_only_duplicate_evidence(tmp_path):
    messages = [{"id": "u1", "role": "user", "content": "Inspect source"}]
    for index in range(7):
        call = {"id": "c%d" % index, "name": "read", "args": {"path": "a.py"}}
        messages.extend([
            {"id": "a%d" % index, "role": "assistant", "content": "", "tool_calls_json": json.dumps([call])},
            {"role": "tool", "tool_result_json": json.dumps({"tool_call_id": call["id"], "is_error": False, "content": "same source"})},
        ])
    source = tmp_path / "tx.jsonl"
    source.write_text(json.dumps({"session_id": "s1", "status": "settled", "messages": messages}) + "\n")
    stats = dpo.extract(source, max_routine_reads_per_session=2)
    assert stats["capped_routine_sft"] == 3
    command = {"tool_calls": [{"name": "command", "arguments": '{"command":"cat a.py; arbitrary-program"}'}]}
    assert not dpo._is_routine_orientation_turn(command, 10)
