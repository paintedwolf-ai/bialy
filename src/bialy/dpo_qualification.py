"""Admission checks for custom-model data.

These checks bind trusted host capture and validation reports to dataset bytes.
They do not perform host execution, assess task success, or authenticate a report.
"""

import hashlib
import json
import re

from jsonschema import Draft202012Validator, SchemaError, ValidationError

CAPTURE_SCHEMA = "pw-model-request-capture/1"
VALIDATION_SCHEMA = "pw-training-validation/1"
HOST_VALIDATOR = "paintedwolf-host-training-validation/1"
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def capture_reasons(row):
    if not isinstance(row, dict):
        return ["invalid_training_row"]
    capture = row.get("request_capture")
    if not isinstance(capture, dict):
        return ["missing_exact_request_capture"]
    reasons = []
    if capture.get("schema") not in (CAPTURE_SCHEMA, "pw-model-request-capture/2"):
        reasons.append("unsupported_request_capture")
    supported_fields = {
        "schema", "request_id", "session_id", "message_id", "host_revision", "epoch", "tool_surface",
        "catalog_sha256", "state_sha256", "messages", "tools", "sha256", "controls",
        "student_provider", "student_kind", "student_model", "dispatch_sha256",
    }
    if set(capture) - supported_fields:
        reasons.append("unsupported_request_capture_fields")
    for key in ("request_id", "session_id", "message_id", "host_revision", "epoch", "tool_surface"):
        if not isinstance(capture.get(key), str) or not capture[key].strip():
            reasons.append("missing_capture_%s" % key)
    for key in ("catalog_sha256", "state_sha256"):
        if not isinstance(capture.get(key), str) or not SHA256_RE.fullmatch(capture[key]):
            reasons.append("invalid_capture_%s" % key)
    if capture.get("session_id") != row.get("session") or capture.get("message_id") != row.get("message_id"):
        reasons.append("capture_turn_mismatch")
    unsigned = {k: v for k, v in capture.items() if k != "sha256"}
    if capture.get("sha256") != digest(unsigned):
        reasons.append("capture_digest_mismatch")
    if capture.get("schema") == "pw-model-request-capture/2" and not isinstance(capture.get("controls"), dict):
        reasons.append("missing_request_controls")
    messages = capture.get("messages")
    if not isinstance(messages, list) or not messages or messages != row.get("prompt"):
        reasons.append("capture_prompt_mismatch")
    elif any(
        not isinstance(m, dict) or not isinstance(m.get("role"), str)
        or m["role"] not in {"system", "developer", "user", "assistant", "tool"}
        for m in messages
    ):
        reasons.append("invalid_captured_messages")
    elif not any(m.get("role") in {"system", "developer"} and m.get("content") for m in messages):
        reasons.append("missing_captured_host_guidance")
    elif any(m.get("content") is not None and not isinstance(m["content"], str) for m in messages):
        reasons.append("unsupported_multimodal_capture")
    tools = capture.get("tools")
    if not isinstance(tools, list):
        reasons.append("missing_captured_tool_schemas")
    else:
        names = []
        for tool in tools:
            fn = tool.get("function") if isinstance(tool, dict) else None
            if not isinstance(fn, dict) or tool.get("type") != "function" or not isinstance(fn.get("name"), str) or not fn["name"]:
                reasons.append("invalid_captured_tool_schema")
                continue
            names.append(fn["name"])
            schema = fn.get("parameters")
            # A bare object schema is a placeholder, not the tool's schema.
            if not isinstance(schema, dict) or schema == {"type": "object"} or schema.get("type") != "object":
                reasons.append("missing_captured_tool_schema_constraints")
                continue
            try:
                Draft202012Validator.check_schema(schema)
            except SchemaError:
                reasons.append("invalid_captured_tool_schema")
        if len(names) != len(set(names)):
            reasons.append("duplicate_captured_tool")
    return sorted(set(reasons))


def response_reasons(turn, tools):
    if not isinstance(turn, dict) or turn.get("role") != "assistant":
        return ["missing_assistant_response"]
    calls = turn.get("tool_calls")
    if not isinstance(calls, list) or not isinstance(turn.get("text"), str) or (not calls and not turn["text"].strip()):
        return ["empty_or_malformed_response"]
    schemas = {t["function"]["name"]: t["function"]["parameters"] for t in tools}
    for call in calls:
        if not isinstance(call, dict) or not isinstance(call.get("name"), str) or call["name"] not in schemas:
            return ["response_tool_not_offered"]
        try:
            args = call.get("arguments")
            args = json.loads(args) if isinstance(args, str) else args
            Draft202012Validator(schemas[call["name"]]).validate(args)
        except (ValueError, TypeError, ValidationError):
            return ["response_arguments_invalid"]
    return []


def qualification(row):
    if not isinstance(row, dict):
        return {"status": "candidate", "reasons": ["invalid_training_row"]}
    reasons = capture_reasons(row)
    report = row.get("validation")
    capture = row.get("request_capture")
    capture = capture if isinstance(capture, dict) else {}
    if not isinstance(report, dict):
        reasons.append("missing_host_validation")
    else:
        if report.get("schema") != VALIDATION_SCHEMA or report.get("validator") != HOST_VALIDATOR:
            reasons.append("unsupported_host_validator")
        if not isinstance(report.get("receipt_id"), str) or not report["receipt_id"].strip():
            reasons.append("missing_validation_receipt")
        for key in ("state_sha256", "host_revision", "epoch", "tool_surface", "catalog_sha256"):
            if not capture.get(key) or report.get(key) != capture[key]:
                reasons.append("validation_%s_mismatch" % key)
        if report.get("request_sha256") != capture.get("sha256"):
            reasons.append("validation_request_mismatch")
        if row.get("schema") == "pw-custom-sft/1":
            if report.get("kind") != "accepted_response" or report.get("response_origin") != "model" or report.get("outcome") != "accepted":
                reasons.append("response_not_host_accepted")
            if report.get("response_sha256") != digest(row.get("target")):
                reasons.append("validation_response_mismatch")
        elif row.get("schema") == "pw-custom-dpo/1":
            if report.get("kind") != "preference" or report.get("preference") != "chosen":
                reasons.append("missing_validated_preference")
            if report.get("chosen_outcome") != "accepted" or report.get("rejected_outcome") != "rejected":
                reasons.append("preference_outcomes_unvalidated")
            if report.get("chosen_sha256") != digest(row.get("chosen")) or report.get("rejected_sha256") != digest(row.get("rejected")):
                reasons.append("validation_pair_mismatch")
            if row.get("chosen") == row.get("rejected"):
                reasons.append("identical_preference_responses")
        else:
            reasons.append("unsupported_training_schema")
        response = row.get("target") if row.get("schema") == "pw-custom-sft/1" else row.get("chosen")
        quality = report.get("quality")
        if not isinstance(quality, dict) or quality.get("verdict") != "pass":
            reasons.append("missing_quality_validation")
        else:
            if quality.get("request_sha256") != capture.get("sha256") or quality.get("response_sha256") != digest(response):
                reasons.append("quality_binding_mismatch")
            evidence = quality.get("evidence_ids")
            if not isinstance(evidence, list) or not evidence or any(not isinstance(e, str) or not e.strip() for e in evidence):
                reasons.append("missing_quality_evidence")
            if quality.get("method") == "model_judge":
                if not quality.get("judge_model") or not quality.get("judge_revision"):
                    reasons.append("missing_quality_judge_identity")
            elif quality.get("method") != "deterministic":
                reasons.append("unsupported_quality_method")
    if not capture_reasons(row):
        response = row.get("target") if row.get("schema") == "pw-custom-sft/1" else row.get("chosen")
        reasons.extend(response_reasons(response, capture["tools"]))
    candidate_reasons = row.get("candidate_reasons", [])
    if not isinstance(candidate_reasons, list) or any(not isinstance(reason, str) for reason in candidate_reasons):
        reasons.append("invalid_candidate_reasons")
    else:
        reasons.extend(candidate_reasons)
    return {"status": "candidate" if reasons else "qualified", "reasons": sorted(set(reasons))}


def training_eligible(row):
    """Recheck bound evidence; a caller-supplied qualification flag has no authority."""
    return qualification(row)["status"] == "qualified"
