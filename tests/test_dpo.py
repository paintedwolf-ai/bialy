import dataclasses
import gzip
import json
import sqlite3
from pathlib import Path

import pytest
import zstandard
from conftest import emitted, row

from bialy import cli, config, dpo, fleet, tasks
from bialy import run as runmod

FIXTURES = Path(__file__).parent / "fixtures"


class FakeStudentChat:
    """Returns canned non-thinking tool turns keyed by a handler function."""

    def __init__(self, fn):
        self.fn = fn
        self.calls = []

    def turn(self, messages, tools=None, temperature=0.0, seed=None, max_tokens=4096, thinking=False):
        self.calls.append({"messages": messages, "tools": tools, "temperature": temperature, "seed": seed, "thinking": thinking})
        return self.fn(messages, tools)


def _sample_transcript(session_id="s1", workspace="flask", group="flask:g1", model="gemma-4-31b-it", status="settled"):
    return emitted({
        "shard": "flask-000",
        "session_id": session_id,
        "parent_session_id": None,
        "root_session": session_id,
        "task_id": "t1",
        "workspace": workspace,
        "prompt_group": group,
        "archetype": "bugfix",
        "model": model,
        "status": status,
        "title": "Fix ctx bug",
        "workflow_id": None,
        "messages": [
            {"id": "m1", "role": "user", "origin": "user", "content": "Inspect ctx.py and summarize the bug."},
            {
                "id": "m2",
                "role": "assistant",
                "content": "Presented, not what the model wrote.",
                "reasoning_json": json.dumps({"text": "internal chain of thought to strip"}),
                "model_output": {
                    "content": "Let me read the symbol.",
                    "tool_calls_json": [
                        {"id": "c0", "name": "request_tools", "args": {"names": ["summarize"]}},
                        {"id": "c1", "name": "skills_read", "args": {"name": "commit"}},
                        {"id": "c2", "name": "read", "args": {"file_path": "src/flask/ctx.py", "symbol": "AppContext"}},
                        {"id": "c3", "name": "summarize", "args": {"file_path": "src/flask/ctx.py"}},
                    ],
                },
            },
            {
                "id": "m3",
                "role": "tool",
                "tool_result_json": json.dumps([
                    {"tool_call_id": "c0", "tool_name": "request_tools", "content": "loaded summarize", "is_error": False},
                    {"tool_call_id": "c1", "tool_name": "skills_read", "content": "skill card", "is_error": False},
                    {"tool_call_id": "c2", "tool_name": "read", "content": "class AppContext: ...", "is_error": False},
                    {"tool_call_id": "c3", "tool_name": "summarize", "content": "summary...", "is_error": False},
                ]),
            },
            {
                "id": "m4",
                "role": "assistant",
                "content": "Trying to edit outside workspace.",
                "tool_calls_json": json.dumps([{"id": "c4", "name": "edit", "args": {"file_path": "../secret.py"}}]),
            },
            {
                "id": "m5",
                "role": "tool",
                "tool_result_json": json.dumps([{
                    "tool_call_id": "c4",
                    "tool_name": "edit",
                    "content": "Status: error\nTool: edit\nCode: outside_roots\nSummary: Path is outside allowed roots.",
                    "is_error": True,
                    "details": {"paintedwolf.confinement": {"blocked_path": "../secret.py"}},
                }]),
            },
            {
                "id": "m6",
                "role": "assistant",
                "content": "Editing inside src/flask/ctx.py instead.",
                "tool_calls_json": json.dumps([{"id": "c5", "name": "edit", "args": {"file_path": "src/flask/ctx.py"}}]),
            },
            {
                "id": "m7",
                "role": "tool",
                "tool_result_json": json.dumps([{"tool_call_id": "c5", "tool_name": "edit", "content": "ok", "is_error": False}]),
            },
        ],
    })


def test_extract_strips_thinking_preloads_judge_tools_and_pairs_rejections(tmp_path):
    tx_path = tmp_path / "transcripts.jsonl.gz"
    with gzip.open(tx_path, "wt", encoding="utf-8") as fh:
        fh.write(json.dumps(_sample_transcript()) + "\n")
    rows_path = tmp_path / "judged.jsonl"
    judged_row = row(
        "flask",
        "flask:g1",
        1,
        tool_scores={"summarize": 4, "edit": 3},
        skill_scores={"commit": 4},
        requests=[{"need": "summarize file", "exact": ["summarize"], "after": ["summarize"], "scores": {"summarize": 4}}],
    )
    rows_path.write_text(json.dumps(judged_row) + "\n")
    sft_out = tmp_path / "sft.jsonl"
    dpo_out = tmp_path / "dpo.jsonl"
    stats = dpo.extract(tx_path, rows_paths=[rows_path], out_sft=sft_out, out_dpo=dpo_out, min_score=3, preload_from_judge=True)
    assert stats["sft_rows"] == 2
    assert stats["dpo_rows"] == 2
    assert stats["by_source"] == {"host_rejection": 1, "post_rejection_recovery": 1}
    assert stats["rejection_codes"] == {"outside_roots": 1}

    sft_rows = [json.loads(line) for line in sft_out.read_text().splitlines()]
    first_sft = sft_rows[0]
    assert "summarize" in first_sft["surface"]["active_tools"]
    assert "commit" in first_sft["surface"]["preloaded_guides"]
    # Redundant request_tools for preloaded summarize was stripped, and thinking was removed.
    call_names = [c["name"] for c in first_sft["target"]["tool_calls"]]
    assert "request_tools" not in call_names
    assert call_names == ["skills_read", "read", "summarize"]
    assert "internal chain of thought" not in json.dumps(first_sft)
    assert first_sft["target"]["text"] == "Let me read the symbol."
    assert "Presented" not in json.dumps(first_sft)

    dpo_rows = [json.loads(line) for line in dpo_out.read_text().splitlines()]
    prevent, recover = dpo_rows
    assert prevent["source"] == "host_rejection" and prevent["rejection_code"] == "outside_roots"
    assert prevent["chosen"] is None
    assert prevent["proposed_chosen"]["tool_calls"][0]["arguments"] == json.dumps({"file_path": "src/flask/ctx.py"})
    assert not dpo.training_eligible(prevent)
    assert prevent["rejected"]["tool_calls"][0]["arguments"] == json.dumps({"file_path": "../secret.py"})
    assert recover["source"] == "post_rejection_recovery"
    assert any(m["role"] == "tool" and "Code: outside_roots" in m["content"] for m in recover["prompt"])


def test_shadow_harvests_divergences_with_thinking_off_and_resumes(factory, tmp_path, monkeypatch):
    tx_path = tmp_path / "transcripts.jsonl"
    tx_path.write_text(
        json.dumps(_sample_transcript("s1", "flask", "flask:g1")) + "\n"
        + json.dumps(_sample_transcript("s2", "zod", "zod:g2")) + "\n"
    )
    rows_path = tmp_path / "judged.jsonl"
    r1 = row("flask", "flask:g1", 1, tool_scores={"summarize": 4}, skill_scores={"commit": 4})
    r2 = row("zod", "zod:g2", 2, tool_scores={"summarize": 4}, skill_scores={"commit": 4})
    rows_path.write_text(json.dumps(r1) + "\n" + json.dumps(r2) + "\n")
    sft_out = tmp_path / "sft.jsonl"
    dpo.extract(tx_path, rows_paths=[rows_path], out_sft=sft_out, min_score=3)

    # Student omits skills_read and uses unscoped read / shell fallback.
    student_chat = FakeStudentChat(
        lambda messages, tools: {
            "text": "I will run a shell command instead.",
            "tool_calls": [{"id": "s0", "name": "command", "arguments": json.dumps({"cmd": "cat src/flask/ctx.py"})}],
        }
    )
    monkeypatch.setattr(config.Factory, "chats", lambda self, model: [student_chat])
    shadow_out = tmp_path / "shadow.jsonl"
    report = dpo.shadow(factory, sft_out, shadow_out, student_id="gemma-4-e4b-it", workers=2, seed=7)
    assert report["evaluated"] == 0 and report["pairs"] == 0
    assert report["skipped_missing_capture"] == 4
    assert not student_chat.calls
    assert all(c["thinking"] is False and c["temperature"] == 0.0 for c in student_chat.calls)

    # Re-running with the same signature reuses the partial file without making new LLM calls.
    calls_before = len(student_chat.calls)
    report_again = dpo.shadow(factory, sft_out, shadow_out, student_id="gemma-4-e4b-it", workers=2, seed=7)
    assert len(student_chat.calls) == calls_before and report_again["pairs"] == report["pairs"]



def test_repair_fills_missing_chosen_from_teacher(factory, tmp_path, monkeypatch):
    tx = _sample_transcript()
    # Remove the self-repair messages so the rejection has chosen=None.
    tx["messages"] = tx["messages"][:5]
    tx_path = tmp_path / "tx.jsonl"
    tx_path.write_text(json.dumps(tx) + "\n")
    dpo_raw = tmp_path / "dpo.raw.jsonl"
    dpo.extract(tx_path, out_dpo=dpo_raw)
    raw_rows = [json.loads(line) for line in dpo_raw.read_text().splitlines()]
    assert len(raw_rows) == 1 and raw_rows[0]["chosen"] is None

    teacher_chat = FakeStudentChat(
        lambda messages, tools: {
            "text": "Editing inside the repository root.",
            "tool_calls": [{"id": "t0", "name": "edit", "arguments": json.dumps({"file_path": "src/flask/ctx.py"})}],
        }
    )
    monkeypatch.setattr(config.Factory, "chats", lambda self, model: [teacher_chat])
    repaired_out = tmp_path / "dpo.repaired.jsonl"
    rep = dpo.repair(factory, dpo_raw, repaired_out, teacher_id="gemma-4-31b-it", workers=1, seed=7)
    assert rep == {"input_rows": 1, "queried": 0, "proposed": 0, "skipped_missing_capture": 1, "completed_pairs": 0}
    repaired_row = json.loads(repaired_out.read_text())
    assert repaired_row["chosen"] is None
    assert not teacher_chat.calls


MODEL_OUTPUT_SCHEMA = """
    CREATE TABLE sessions (id TEXT, parent_session_id TEXT, title TEXT, workflow_id TEXT, created_at TEXT);
    CREATE TABLE messages (id TEXT, session_id TEXT, role TEXT, content TEXT, kind TEXT, origin TEXT, visibility TEXT, seq INT, ord INT, ts TEXT,
        tool_calls_json TEXT, tool_result_json TEXT, reasoning_json TEXT, grounding_json TEXT, artifact_ids_json TEXT,
        completion_report_json TEXT, draft_version_count INT, draft_status TEXT);
    CREATE TABLE model_outputs (id TEXT, session_id TEXT, project_id TEXT, iteration INT, scripted INT, message_id TEXT,
        content_blob_sha256 TEXT, finish_reason TEXT, settled_at TEXT);
    CREATE TABLE draft_versions (session_id TEXT, slot_id TEXT, version_index INT, body TEXT, outcome_code TEXT, created_at TEXT);
"""


def _store_shard(tmp_path):
    tasks_dir = tmp_path / "tasks"
    tasks_dir.mkdir()
    (tasks_dir / "flask.jsonl").write_text(
        json.dumps({"id": "t1", "prompt": "fix", "model": "gemma-4-31b-it", "meta": {"workspace": "flask", "prompt_group": "flask:g1", "archetype": "bugfix"}})
        + "\n"
    )
    shard = tmp_path / "run" / "shards" / "flask-000"
    shard.mkdir(parents=True)
    (shard / "manifest.jsonl").write_text(json.dumps({"task_id": "t1", "root_session": "s1", "status": "settled"}) + "\n")
    conn = sqlite3.connect(str(shard / "store.db"))
    conn.executescript(MODEL_OUTPUT_SCHEMA)
    conn.execute("INSERT INTO sessions VALUES ('s1', NULL, 'test', NULL, '1')")
    columns = "id, session_id, role, content, kind, origin, seq"
    conn.execute("INSERT INTO messages (%s) VALUES ('u1', 's1', 'user', 'hi', '', 'user', 1)" % columns)
    conn.execute("INSERT INTO messages (%s) VALUES ('a1', 's1', 'assistant', 'Presented without its citation.', '', 'model', 2)" % columns)
    conn.commit()
    return tasks_dir, shard, conn


def _host_blob(shard, project="p1"):
    """Retain the body the host wrote, so this reads the host's own compression and envelope."""
    corpus = json.loads(gzip.decompress((FIXTURES / "citation-grammar.json.gz").read_bytes()))
    sha = corpus["model_output_blob"]
    target = shard / "projects" / project / "model-content" / sha[:2]
    target.mkdir(parents=True)
    (target / sha[2:]).write_bytes((FIXTURES / "model-output-body.zst").read_bytes())
    return sha


def test_export_reads_exact_model_text_from_retained_outputs(tmp_path):
    tasks_dir, shard, conn = _store_shard(tmp_path)
    sha = _host_blob(shard)
    columns = "id, session_id, project_id, iteration, scripted, message_id, content_blob_sha256, finish_reason, settled_at"
    # The latest attempt at a slot is the output its row settled on.
    conn.execute("INSERT INTO model_outputs (%s) VALUES ('o0', 's1', 'p1', 1, 0, 'a1', '', 'stop', '1')" % columns)
    conn.execute("INSERT INTO model_outputs (%s) VALUES ('o1', 's1', 'p1', 2, 0, 'a1', ?, 'stop', '2')" % columns, (sha,))
    conn.commit()
    conn.close()

    out = tmp_path / "transcripts.jsonl.gz"
    assert cli.main(["dpo", "export", "--run", str(tmp_path / "run"), "--tasks", str(tasks_dir), "--out", str(out)]) == 0
    [record] = list(dpo.iter_transcripts(out))
    assert record["workspace"] == "flask" and record["model"] == "gemma-4-31b-it"
    user, assistant = record["messages"]
    assert "model_output" not in user
    turn = dpo.model_turn(assistant)
    assert turn["text"] == "Fixed [read#3:42]."
    assert [call["name"] for call in turn["tool_calls"]] == ["read"]
    assert "hidden" not in json.dumps(turn)


def test_export_treats_scripted_output_as_no_model_text_and_counts_missing_bodies(tmp_path):
    tasks_dir, shard, conn = _store_shard(tmp_path)
    columns = "id, session_id, project_id, iteration, scripted, message_id, content_blob_sha256, finish_reason, settled_at"
    conn.execute("INSERT INTO model_outputs (%s) VALUES ('o1', 's1', 'p1', 1, 1, 'a1', ?, 'stop', '1')" % columns, ("a" * 64,))
    conn.commit()
    out = tmp_path / "scripted.jsonl"
    assert dpo.export_transcripts(tmp_path / "run", tasks_dir, out)["model_outputs"] == 0
    assert "model_output" not in list(dpo.iter_transcripts(out))[0]["messages"][1]

    # One lost body is counted; losing every body means the shards never retained them.
    conn.execute("INSERT INTO model_outputs (%s) VALUES ('o2', 's1', 'p1', 2, 0, 'a1', ?, 'stop', '2')" % columns, ("b" * 64,))
    conn.commit()
    with pytest.raises(ValueError, match="retained"):
        dpo.export_transcripts(tmp_path / "run", tasks_dir, out)
    sha = _host_blob(shard)
    conn.execute("INSERT INTO messages (id, session_id, role, content, seq) VALUES ('a2', 's1', 'assistant', '', 3)")
    conn.execute("INSERT INTO model_outputs (%s) VALUES ('o3', 's1', 'p1', 3, 0, 'a2', ?, 'stop', '3')" % columns, (sha,))
    conn.commit()
    conn.close()
    assert dpo.export_transcripts(tmp_path / "run", tasks_dir, out)["missing_model_outputs"] == 1

    target = shard / "projects" / "p1" / "model-content" / sha[:2] / sha[2:]
    target.write_bytes(zstandard.ZstdCompressor().compress(b"{}"))
    with pytest.raises(ValueError, match="does not match its digest"):
        dpo.export_transcripts(tmp_path / "run", tasks_dir, out)


def test_hybrid_generator_weights_local_workers_and_sidecar_request_metering(factory, tmp_path):
    settings = config.run_settings({
        "generators": ["gemma-4-31b-it", "qwen3.6-35b-a3b", "glm-5.3-flash"],
        "judges": ["qwen3.6-35b-a3b", "gemma-4-31b-it"],
        "generator_weights": {"gemma-4-31b-it": 0.45, "qwen3.6-35b-a3b": 0.40, "glm-5.3-flash": 0.15},
        "archetype_generators": {"locate": ["gemma-4-31b-it", "qwen3.6-35b-a3b"]},
        "local_workers_only": True,
    })
    run = runmod.Run(factory=dataclasses.replace(factory, root=tmp_path), name="hybrid", settings=settings)
    scaled = runmod.scaled_factory(run)
    assert [m.id for m in scaled.generators()] == ["qwen3.6-35b-a3b", "glm-5.3-flash", "gemma-4-31b-it"]
    assert scaled.reviewer(0).id == "qwen3.6-35b-a3b"
    assert scaled.reviewer(0).id == "qwen3.6-35b-a3b"
    assert scaled.reviewer(1).id in {"qwen3.6-35b-a3b", "gemma-4-31b-it"}

    # With local_workers_only=True, lite and agent_pool exclude hosted glm-5.3-flash.
    _, policy, _, _, _ = fleet.sidecar_files(scaled, {"LYCAON_DECIDE_DISABLED": "1"}, 0)
    agent_pool_section = policy.split("agent_pool:\n")[1]
    assert "fireworks-glm-5-3-flash" not in agent_pool_section
    assert "vllm-qwen3-6-35b-a3b" in agent_pool_section and "vllm-gemma-4-31b-it" in agent_pool_section

    # Archetype-constrained tasks for `locate` never assign glm-5.3-flash.
    raw = [{"prompt": "Where is the configuration setting number %d defined in this repo?" % i,
            "follow_ups": [], "archetype": "locate", "lang": "en", "author": "m"} for i in range(40)]
    tasks.finalize(scaled, scaled.repos[0], raw, [], tmp_path / "locate.jsonl", seed=7)
    assigned = {json.loads(line)["model"] for line in (tmp_path / "locate.jsonl").read_text().splitlines()}
    assert "glm-5.3-flash" not in assigned and assigned <= {"qwen3.6-35b-a3b", "gemma-4-31b-it"}

    # Sidecar requests.jsonl token usage is collected and metered by usage_summary / estimated_spend.
    tasks_dir = tmp_path / "tasks"
    tasks_dir.mkdir()
    (tasks_dir / "flask.jsonl").write_text(json.dumps({"id": "t0", "prompt": "p", "model": "glm-5.3-flash", "meta": {"workspace": "flask"}}) + "\n")
    shard = run.path("fleet-off", "shards", "flask-000")
    shard.mkdir(parents=True)
    (shard / "manifest.jsonl").write_text(json.dumps({"task_id": "t0", "root_session": "s1", "status": "settled", "meta": {"workspace": "flask"}}) + "\n")
    (shard / "rows.jsonl").write_text(json.dumps({"root_session": "s1", "receipt": 1}) + "\n")
    (shard / "requests.jsonl").write_text(
        json.dumps({"schema": "pw-decide-request/1", "root_session": "s1", "model": "accounts/fireworks/models/glm-5p3-flash", "input_tokens": 1000000, "output_tokens": 1000000})
        + "\n"
    )
    fleet.collect(run.path("fleet-off"), "engine-off", tasks_dir)
    totals = runmod.usage_summary(run)
    assert totals["accounts/fireworks/models/glm-5p3-flash"] == {"calls": 1, "prompt_tokens": 1000000, "completion_tokens": 1000000}
    assert runmod.estimated_spend(run, totals) == 0.65


def test_compact_prompt_preserves_head_tail_and_read_receipts():
    msgs = [
        {"role": "system", "content": "system prompt with tool rules"},
        {"role": "user", "content": "fix status.py"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "read", "arguments": '{"path":"status.py"}'}}]},
        {"role": "tool", "tool_call_id": "c1", "name": "read", "content": "def display_status(s):\n" + ("    # body line\n" * 100)},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "c2", "type": "function", "function": {"name": "read", "arguments": '{"path":"test_status.py"}'}}]},
        {"role": "tool", "tool_call_id": "c2", "name": "read", "content": "class TestStatus:\n" + ("    pass\n" * 100)},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "c3", "type": "function", "function": {"name": "edit", "arguments": '{"path":"status.py"}'}}]},
        {"role": "tool", "tool_call_id": "c3", "name": "edit", "content": "Rejected:\nCode: PROGRESS_MISSING"},
    ]
    with pytest.raises(ValueError, match="recapture host compaction"):
        dpo.compact_prompt(msgs, max_chars=800)
    intact = dpo.compact_prompt(msgs, max_chars=20000)
    assert intact == msgs and intact is not msgs
    assert "file already read in transcript" not in json.dumps(intact)


def test_real_store_db_schema_turn_closeout_rejection_and_attachments(factory, tmp_path):
    # 1. Real store.db / sessions/transcripts.jsonl.gz schema with tool_calls_json and turn-level host rejection.
    transcript = {
        "id": "sess-real-1",
        "parent_session_id": None,
        "status": "idle",
        "model": "gemma-4-26b-a4b-it",
        "messages": [
            {"id": "u1", "role": "user", "origin": "user", "content": "Don't make changes. Tell me how ctx.py works."},
            {"id": "a1", "role": "assistant", "origin": "model", "content": "",
             "tool_calls_json": json.dumps([{"name": "read", "id": "c1", "args": {"path": "src/flask/ctx.py", "ranges": [{"start_line": 1, "end_line": 40}]}}])},
            {"id": "t1", "role": "tool", "origin": "tool", "content": "[read#1]\ndef push(): pass",
             "tool_result_json": json.dumps({"content": "[read#1]\ndef push(): pass"})},
            {"id": "a2", "role": "assistant", "origin": "model", "content": "Done, everything is complete.", "tool_calls_json": None},
            {"id": "u2", "role": "user", "origin": "host",
             "content": "Rejected: This tool-free response would end the turn with 2 progress steps still pending.\n\nCause: Work remains unfinished.\nCode: TURN_CLOSEOUT_PENDING_PROGRESS"},
            {"id": "a3", "role": "assistant", "origin": "model", "content": "",
             "tool_calls_json": json.dumps([{"name": "update_progress", "id": "c2", "args": {"content": "## Progress\n- [x] Done"}}])},
            {"id": "t2", "role": "tool", "origin": "tool", "content": "Updated progress.", "tool_result_json": json.dumps({"content": "Updated progress."})},
        ],
    }
    tpath = tmp_path / "real_transcripts.jsonl"
    tpath.write_text(json.dumps(emitted(transcript)) + "\n")
    sft_out = tmp_path / "sft.jsonl"
    dpo_out = tmp_path / "dpo.jsonl"
    stats = dpo.extract(tpath, out_sft=sft_out, out_dpo=dpo_out)
    assert stats["sft_rows"] == 2
    assert stats["rejection_codes"] == {"TURN_CLOSEOUT_PENDING_PROGRESS": 1}
    dpo_rows = [json.loads(line) for line in dpo_out.read_text().splitlines()]
    assert {r["source"] for r in dpo_rows} == {"host_rejection"}
    prevent = next(r for r in dpo_rows if r["source"] == "host_rejection")
    assert prevent["rejected"]["tool_calls"] == [] and prevent["chosen"] is None
    assert prevent["proposed_chosen"] is None  # No matching success receipt was exported.

    # 2. design_review archetype and explicit path references.
    arch_by_id = {a.id: a for a in factory.archetypes}
    assert "design_review" in arch_by_id and arch_by_id["design_review"].reference_rate > 0
    files = ["src/flask/ctx.py", "src/flask/app.py", "README.md"]
    raw = [{
        "prompt": "Don't make changes yet. Review src/flask/ctx.py and tell me how we should restructure it.",
        "follow_ups": ["What if we keep the existing signature?", "Do as you would recommend.", "Now commit the changes."],
        "archetype": "design_review",
        "lang": "en",
        "author": "m",
    }]
    forced = dataclasses.replace(factory, archetypes=[dataclasses.replace(a, reference_rate=1.0, follow_up_rate=1.0) for a in factory.archetypes])
    tasks.finalize(forced, forced.repos[0], raw, files, tmp_path / "design.jsonl", seed=7)
    [task] = [json.loads(line) for line in (tmp_path / "design.jsonl").read_text().splitlines()]
    assert "existing file `src/flask/ctx.py`" in task["prompt"]
    assert "```attachment" not in task["prompt"] and "[User attached" not in task["prompt"]
    assert len(task["follow_ups"]) == 3

    # 3. Citation-grounding repair, AMBIENT_UNGROUNDED_COMPLETION extraction, and shallow_investigation divergence.
    cit_tx = {
        "id": "sess-cit-1",
        "parent_session_id": None,
        "status": "settled",
        "model": "glm-5.3-flash",
        "messages": [
            {"id": "u1", "role": "user", "origin": "user", "content": "Audit oar-gateway compliance."},
            {"id": "a1", "role": "assistant", "origin": "model", "content": "",
             "tool_calls_json": json.dumps([{"name": "read", "id": "c1", "args": {"path": "src/engine.rs"}}])},
            {"id": "t1", "role": "tool", "origin": "tool", "content": "[read#1]\npub struct Engine;"},
            {"id": "a2", "role": "assistant", "origin": "model", "content": "",
             "tool_calls_json": json.dumps([{"name": "verify", "id": "c2", "args": {"command": "cargo test"}}])},
            {"id": "t2", "role": "tool", "origin": "tool", "content": "[tui#1]\ntest result: ok."},
            {"id": "a3", "role": "assistant", "origin": "model", "content": "The engine is sound [src/engine.rs:999]."},
            {"id": "u2", "role": "user", "origin": "host",
             "content": "[host:coordinator-citation-grounding]\n\nRepair the rejected citation; the host keeps the pinned body. Attempt 2/3.\n\nRejected: src/engine.rs:999 (Code: INVEST_HANDLE_NOT_OBSERVED)"},
            {"id": "a4", "role": "assistant", "origin": "model", "content": "[cite: c1 => read#1]",
             "grounding_json": json.dumps({"traced": True, "cited_evidence": [{"evidence": "read#1", "verdict": "traced"}]})},
        ],
    }
    cit_path = tmp_path / "cit_tx.jsonl"
    cit_path.write_text(json.dumps(emitted(cit_tx)) + "\n")
    cit_sft = tmp_path / "cit_sft.jsonl"
    cit_dpo = tmp_path / "cit_dpo.jsonl"
    cit_stats = dpo.extract(cit_path, out_sft=cit_sft, out_dpo=cit_dpo)
    assert cit_stats["rejection_codes"] == {"COORDINATOR_CITATION_GROUNDING": 1}
    assert dpo.parse_host_user_rejection({"role": "user", "content": ">>> Tool feedback\nGap.\nCode: AMBIENT_UNGROUNDED_COMPLETION"})["code"] == "AMBIENT_UNGROUNDED_COMPLETION"
    cit_sft_rows = [json.loads(line) for line in cit_sft.read_text().splitlines()]
    assert cit_sft_rows[-1]["target"]["text"] == "[cite: c1 => read#1]"
    assert cit_sft_rows[-1]["closeout"] == {"outcome": "grounded", "report_phase": False}
    cit_dpo_rows = [json.loads(line) for line in cit_dpo.read_text().splitlines()]
    cit_recover = next(r for r in cit_dpo_rows if r["source"] == "post_rejection_recovery")
    assert cit_recover["chosen"] is None
    assert "read#1" in cit_recover["proposed_chosen"]["text"]

    div = dpo.classify_divergence(
        {"surface": {"active_tools": ["list_dir", "summarize", "read", "verify"], "loadable": []}, "prompt": []},
        {"tool_calls": [{"name": "verify", "arguments": '{"command":"cargo test"}'}]},
        {"tool_calls": [{"name": "list_dir", "arguments": '{"path":"."}'}]},
    )
    assert div == "shallow_investigation"

    web_div = dpo.classify_divergence(
        {"surface": {"active_tools": ["web_search", "fetch_url"], "loadable": []}, "prompt": []},
        {"tool_calls": [{"name": "fetch_url", "arguments": '{"url":"https://open-meteo.com/en/docs"}'}]},
        {"tool_calls": [{"name": "web_search", "arguments": '{"query":"free weather api python"}'}]},
    )
    assert web_div == "snippet_only_web_research"
    assert "public_api_cli" in arch_by_id and arch_by_id["public_api_cli"].workspace == "stack"

    # 4. An exact closeout target, with rejected draft attempts as their own Markdown.
    final_text = (
        "Built and verified the game in [write#1:1-40].\n"
        "[artifact:9823b977-2c86-4057-9fcb-ef92acbbc97d]\n"
        "[verify: targeted - Captured and played the board] [page#1]"
    )
    closeout_tx = {
        "id": "sess-closeout-1",
        "parent_session_id": None,
        "status": "settled",
        "workspace": "oar-gateway",
        "model": "claude-sonnet-4-6",
        "messages": [
            {"id": "u1", "role": "user", "origin": "user", "content": "Build tic tac toe and verify it."},
            {"id": "a1", "role": "assistant", "origin": "model", "content": "",
             "tool_calls_json": json.dumps([{"name": "write", "id": "c1", "args": {"path": "index.html", "content": "<html></html>"}}])},
            {"id": "t1", "role": "tool", "origin": "tool", "content": "[write#1]\nWrote index.html"},
            {"id": "a2", "role": "assistant", "origin": "model", "content": "",
             "tool_calls_json": json.dumps([{"name": "capture_page", "id": "c2", "args": {"path": "index.html", "capture": "screenshot"}}])},
            {"id": "t2", "role": "tool", "origin": "tool", "content": '[page#1]\n{"artifact_id":"9823b977-2c86-4057-9fcb-ef92acbbc97d"}'},
            {
                "id": "slot-final",
                "role": "assistant",
                "origin": "model",
                "kind": "completion_report",
                "content": "Built and verified the game in index.html.",
                "model_output": {"content": final_text},
                "grounding_json": json.dumps({
                    "traced": True,
                    "cited_evidence": [
                        {"evidence": "write#1", "path": "index.html", "verdict": "traced"},
                        {"evidence": "page#1", "verdict": "traced"},
                    ],
                    "verification": {"method": "targeted", "reason": "Captured and played the board."},
                }),
                "artifact_ids_json": json.dumps(["9823b977-2c86-4057-9fcb-ef92acbbc97d"]),
            },
        ],
        "draft_versions": [
            {"slot_id": "slot-final", "version_index": 0, "outcome_code": "INVEST_HANDLE_NOT_OBSERVED", "body": "Built the game [index.html:999]."},
            {"slot_id": "slot-final", "version_index": 1, "outcome_code": "PRESENT_MARKDOWN_EMBED",
             "body": "Built the game: ![shot](9823b977-2c86-4057-9fcb-ef92acbbc97d) [write#1]"},
        ],
    }
    co_path = tmp_path / "closeout_tx.jsonl"
    co_path.write_text(json.dumps(emitted(closeout_tx)) + "\n")
    co_sft = tmp_path / "co_sft.jsonl"
    co_dpo = tmp_path / "co_dpo.jsonl"
    co_stats = dpo.extract(co_path, out_sft=co_sft, out_dpo=co_dpo)
    assert co_stats["rejection_codes"] == {"INVEST_HANDLE_NOT_OBSERVED": 1, "PRESENT_MARKDOWN_EMBED": 1}
    co_sft_rows = [json.loads(line) for line in co_sft.read_text().splitlines()]
    assert len(co_sft_rows) == 3
    final_sft = co_sft_rows[-1]
    assert final_sft["target"]["tool_calls"] == []
    assert final_sft["target"]["text"] == final_text
    assert final_sft["closeout"]["outcome"] == "grounded"

    co_dpo_rows = [json.loads(line) for line in co_dpo.read_text().splitlines()]
    assert [r["rejected"]["text"] for r in co_dpo_rows] == [
        "Built the game [index.html:999].", "Built the game: ![shot](9823b977-2c86-4057-9fcb-ef92acbbc97d) [write#1]"]
    assert all(r["chosen"] is None for r in co_dpo_rows)
    assert all("draft_original_request_unavailable" in r["qualification"]["reasons"] for r in co_dpo_rows)

    # 5. Progress, visual capture, and closeout citation divergence rules + multi-root synthesis.
    assert dpo.classify_divergence(
        {"surface": {"active_tools": ["update_progress", "write"], "loadable": []}, "prompt": []},
        {"tool_calls": [{"name": "update_progress", "arguments": '{"markdown":"- [ ] Build app"}'}]},
        {"tool_calls": [{"name": "write", "arguments": '{"path":"index.html","content":"x"}'}]},
    ) == "missing_progress_before_edit"

    assert dpo.classify_divergence(
        {"surface": {"active_tools": ["capture_page", "command"], "loadable": []}, "prompt": []},
        {"tool_calls": [{"name": "capture_page", "arguments": '{"path":"index.html"}'}]},
        {"tool_calls": [{"name": "command", "arguments": '{"command":"python3 -m http.server 8080"}'}]},
    ) == "missed_visual_capture"

    # A user who asked to see a proposed screen gets a rendered mockup, not the code for it.
    mockup = {"name": "render_view", "arguments": '{"mime":"html","markup":"<main>Dashboard</main>","caption":"Proposed dashboard"}'}
    assert dpo.classify_divergence(
        {"surface": {"active_tools": ["render_view", "write"], "loadable": []}, "prompt": []},
        {"tool_calls": [mockup]},
        {"tool_calls": [{"name": "write", "arguments": '{"path":"dashboard.html","content":"<main/>"}'}]},
    ) == "missed_visual_capture"
    assert dpo.classify_divergence(
        {"surface": {"active_tools": ["render_view", "capture_page"], "loadable": []}, "prompt": []},
        {"tool_calls": [mockup]},
        {"tool_calls": [{"name": "capture_page", "arguments": '{"path":"index.html"}'}]},
    ) != "missed_visual_capture"

    closeout_row = {"surface": {"active_tools": ["read"], "loadable": []}, "prompt": []}
    teacher = {"text": final_sft["target"]["text"], "tool_calls": []}
    artifact = "[artifact:9823b977-2c86-4057-9fcb-ef92acbbc97d]"
    verify = "[verify: inspection - Read the page]"
    for text, code in (
        ("Built the game [write#1:3] and captured it [page#1].\n%s\n%s" % (artifact, verify), None),
        ("Built and verified the game in index.html.\n%s\n%s" % (artifact, verify), "closeout_citation_malformed"),
        ("Built the game [write#1:3].\n%s" % verify, "closeout_citation_malformed"),
        ("Built the game [write#1:3].\n%s" % artifact, "closeout_citation_malformed"),
        # The retired protocol: a fence is literal, so it cites nothing.
        ('Built it.\n\n```json\n{"cited_evidence": [{"evidence": "write#1"}]}\n```\n%s\n%s' % (artifact, verify), "closeout_citation_malformed"),
        ("Built the game [a.go:42:10].\n%s\n%s" % (artifact, verify), "closeout_citation_malformed"),
        ("Built the game [write#1:9-3].\n%s\n%s" % (artifact, verify), "closeout_citation_malformed"),
        ("Built the game [write#1] [verify: guessed - it works].\n%s" % artifact, "closeout_citation_malformed"),
        ("Built the game [write#1] [cite: c1 => read#1].\n%s\n%s" % (artifact, verify), "closeout_citation_malformed"),
        # Code is literal even when it looks like a citation.
        ("Built the game `[write#1]`.\n%s\n%s" % (artifact, verify), "closeout_citation_malformed"),
    ):
        assert dpo.classify_divergence(closeout_row, teacher, {"text": text, "tool_calls": []}) == code, text
    # A student that stops calling tools at a closeout state is not judged by text it never wrote.
    assert dpo.classify_divergence(closeout_row, teacher, {"text": "", "tool_calls": [{"name": "read", "arguments": "{}"}]}) is None

    with pytest.raises(ValueError, match="real attached-root trajectories"):
        dpo.synthesize_multi_root_pairs(co_sft_rows, companion_label="open-agent-rules")


CLEAN_REPORT = {
    "headline": "Session signing requires explicit SECRET_KEY rotation discipline.",
    "summary": "Surveyed request and session surfaces.",
    "rating": {"level": "hardening", "why": "No exploitable RCE; one session hardening claim survived."},
    "findings": [{
        "id": "claim-1",
        "title": "Static SECRET_KEY fallback risk",
        "disposition": "act",
        "where": [{"evidence": "scan#1", "path": "src/flask/sessions.py", "line": 42, "end_line": 44}],
        "action": "Enforce non-default SECRET_KEY at startup.",
    }],
    "set_asides": [{"scanner": "lycaon-secrets", "paths": ["tests/**"], "reason": "Test fixtures only"}],
    "limits": ["Static analysis only"],
    "ask": {"do": "Enforce SECRET_KEY validation", "effort": "small", "why": "Prevent weak signing keys"},
}


def _report_student(**changes):
    document = {**CLEAN_REPORT, **changes}
    return {"text": "## Security survey\n\nAccounted for [scan#1].\n\n```json\n" + json.dumps(document) + "\n```", "tool_calls": []}


def test_report_document_fence_is_judged_only_on_report_phase_rows():
    finding = CLEAN_REPORT["findings"][0]
    surface = {"surface": {"active_tools": ["read"]}, "prompt": [], "closeout": {"report_phase": True}}
    teacher = _report_student()
    assert dpo.classify_divergence(surface, teacher, _report_student()) is None
    for changes in (
        {"headline": " "},
        {"rating": {"level": "fine", "why": "w"}},
        {"rating": {"level": "hardening"}},
        {"findings": []},
        {"findings": [dict(finding, id="other")]},
        {"findings": [dict(finding, disposition="maybe")]},
        {"findings": [dict(finding, where=[{"handle": "scan#1"}])]},
        {"findings": [dict(finding, where=[{"evidence": "scan"}])]},
        {"findings": [dict(finding, where=[{"evidence": "pr#1"}])]},
        {"findings": [dict(finding, where=[{"line": 4}])]},
        {"findings": [dict(finding, where=[{"path": "a.py", "line": "4"}])]},
        {"findings": [dict(finding, where=[{"path": "a.py", "line": 4, "end_line": 2}])]},
        {"set_asides": "none"},
        {"ask": "do it"},
        {"cited_evidence": [{"evidence": "scan#1"}]},
        {"verification": {"method": "targeted", "reason": "r"}},
        {"synthesis": "The answer belongs above the fence."},
    ):
        assert dpo.classify_divergence(surface, teacher, _report_student(**changes)) == "closeout_report_malformed", changes
    no_document = {"text": "## Security survey\n\nAccounted for [scan#1].", "tool_calls": []}
    assert dpo.classify_divergence(surface, teacher, no_document) == "closeout_report_malformed"
    broken = {"text": "Accounted for [scan#1].\n\n```json\n{not json\n```", "tool_calls": []}
    assert dpo.classify_divergence(surface, teacher, broken) == "closeout_report_malformed"
    unclosed = {"text": "Accounted for [scan#1].\n\n```json\n{}", "tool_calls": []}
    assert dpo.classify_divergence(surface, teacher, unclosed) == "closeout_report_malformed"
    # The same reply off a report phase holds no document to judge.
    assert dpo.classify_divergence({"surface": {}, "prompt": []}, teacher, no_document) is None


def test_security_survey_workflow_and_report_document_defects(factory, tmp_path):
    archs = {a.id: a for a in factory.archetypes}
    assert archs["security_survey"].workflows == {"security-auto": 1.0}
    assert archs["security_survey"].timeout("security-auto") == "360m"
    assert archs["security"].workflows == {"implement-dispatch": 0.35}
    assert archs["security"].timeout("implement-dispatch") == "300m"
    assert archs["question"].timeout(None) == "120m"
    assert factory.fleet["shard_minutes"] == 720
    assert factory.run["engine_on"]["deadline_ms"] == 120000
    assert "security-auto" not in archs["security"].workflows

    # Coordinator session running security-auto + child security-reviewer session + defective vs clean report closeout.
    coord_session = {
        "session_id": "sec-coord",
        "parent_session_id": None,
        "root_session": "sec-coord",
        "workspace": "flask",
        "archetype": "security_survey",
        "model": "glm-5.3-flash",
        "status": "settled",
        "workflow_id": "security-auto",
        "messages": [
            {"id": "u-sec", "role": "user", "content": "Run a full security survey of Flask's session and request parsing surfaces."},
            {
                "id": "a-plan-scan",
                "role": "assistant",
                "content": "",
                "tool_calls_json": json.dumps([
                    {"id": "c1", "name": "scan_summary", "args": {"scan_id": "scan-secrets-1"}},
                    {"id": "c2", "name": "scan_query", "args": {"scan_id": "scan-sast-1", "view": "groups"}},
                ]),
            },
            {
                "id": "t-plan-scan",
                "role": "tool",
                "tool_result_json": json.dumps([
                    {"tool_call_id": "c1", "tool_name": "scan_summary", "content": "0 secret findings"},
                    {"tool_call_id": "c2", "tool_name": "scan_query", "content": "group g-cookie: cookie signing"},
                ]),
            },
            {
                "id": "a-plan-fanout",
                "role": "assistant",
                "content": "",
                "tool_calls_json": json.dumps([
                    {
                        "id": "c3",
                        "name": "fanout_plan",
                        "args": {
                            "threat_model": {
                                "context": "Web framework processing untrusted HTTP requests",
                                "entrypoints": ["src/flask/wrappers.py"],
                                "trust_boundaries": ["HTTP client -> WSGI request/session"],
                                "assets": ["session signing key"],
                                "attacker_goals": ["session forgery"],
                            },
                            "legs": [
                                {
                                    "id": "leg-1",
                                    "title": "Audit cookie session signing",
                                    "role": "security-reviewer",
                                    "scope": ["src/flask/sessions.py"],
                                    "scan_refs": ["scan-sast-1"],
                                    "objectives": ["Verify fallback signing key handling"],
                                }
                            ],
                        },
                    },
                    {"id": "c4", "name": "workflow_advance", "args": {}},
                ]),
            },
            {
                "id": "t-plan-fanout",
                "role": "tool",
                "tool_result_json": json.dumps([
                    {"tool_call_id": "c3", "tool_name": "fanout_plan", "content": "Planned execute/leg-1"},
                    {"tool_call_id": "c4", "tool_name": "workflow_advance", "content": "Advanced to execute"},
                ]),
            },
            {
                "id": "a-report-defective",
                "role": "assistant",
                "content": "## Security survey\n\nIncomplete report missing claim-1 [scan#1].",
                "grounding_json": json.dumps({
                    "traced": True,
                    "cited_evidence": [{"evidence": "scan#1", "verdict": "traced"}],
                }),
                "completion_report_json": json.dumps({
                    "scope": "run",
                    "headline": "Incomplete survey",
                    "rating": {"level": "hardening", "why": "Hardening needed."},
                    "defects": [{"code": "REPORT_CLAIM_UNREPORTED", "message": "Claim claim-1 is missing from findings."}],
                }),
            },
            {
                "id": "a-report-clean",
                "role": "assistant",
                "content": "## Security survey\n\nAll claims and scan groups accounted for.",
                "model_output": {"content": "## Security survey\n\nAll claims and scan groups accounted for [scan#1].\n\n```json\n" + json.dumps(CLEAN_REPORT) + "\n```"},
                "grounding_json": json.dumps({
                    "traced": True,
                    "cited_evidence": [{"evidence": "scan#1", "path": "src/flask/sessions.py", "line": 42, "verdict": "traced"}],
                }),
                "completion_report_json": json.dumps(dict(CLEAN_REPORT, scope="run")),
            },
        ],
    }
    worker_session = {
        "session_id": "sec-worker-1",
        "parent_session_id": "sec-coord",
        "root_session": "sec-coord",
        "workspace": "flask",
        "archetype": "security_survey",
        "model": "glm-5.3-flash",
        "status": "settled",
        "workflow_id": "security-auto",
        "messages": [
            {"id": "u-w1", "role": "user", "content": "Audit cookie session signing in src/flask/sessions.py."},
            {
                "id": "a-w1-read",
                "role": "assistant",
                "content": "",
                "tool_calls_json": json.dumps([
                    {"id": "cw1", "name": "read", "args": {"path": "src/flask/sessions.py", "start_line": 1, "end_line": 80}},
                ]),
            },
            {
                "id": "t-w1-read",
                "role": "tool",
                "tool_result_json": json.dumps([
                    {"tool_call_id": "cw1", "tool_name": "read", "content": "42: secret_key = app.config['SECRET_KEY']"},
                ]),
            },
            {
                "id": "a-w1-ungrounded",
                "role": "assistant",
                "content": "",
                "tool_calls_json": json.dumps([
                    {"id": "cw2", "name": "complete_leg", "args": {"leg_status": "completed", "findings": [{"title": "Weak key"}]}},
                ]),
            },
            {
                "id": "t-w1-ungrounded",
                "role": "tool",
                "tool_result_json": json.dumps([
                    {
                        "tool_call_id": "cw2",
                        "tool_name": "complete_leg",
                        "is_error": True,
                        "content": "Status: error\nCode: FINDING_UNGROUNDED\nFindings require path, line, and excerpt.",
                    },
                ]),
            },
            {
                "id": "a-w1-repaired",
                "role": "assistant",
                "content": "",
                "tool_calls_json": json.dumps([
                    {
                        "id": "cw3",
                        "name": "complete_leg",
                        "args": {
                            "leg_status": "completed",
                            "summary": "Reviewed session signing.",
                            "findings": [
                                {
                                    "title": "Static SECRET_KEY fallback risk",
                                    "claim": "hardening",
                                    "severity": "medium",
                                    "path": "src/flask/sessions.py",
                                    "line": 42,
                                    "excerpt": "secret_key = app.config['SECRET_KEY']",
                                    "reachable": "Every signed cookie request",
                                    "attacker": "Remote client holding forged cookie",
                                    "impact": "Session tampering if key is predictable",
                                    "remediation": "Require high-entropy key",
                                }
                            ],
                            "objectives_met": ["Verify fallback signing key handling"],
                            "coverage_gaps": [],
                        },
                    },
                ]),
            },
            {
                "id": "t-w1-repaired",
                "role": "tool",
                "tool_result_json": json.dumps([
                    {"tool_call_id": "cw3", "tool_name": "complete_leg", "content": "Leg execute/leg-1 completed.", "is_error": False},
                ]),
            },
        ],
    }

    tx_path = tmp_path / "sec_tx.jsonl"
    tx_path.write_text(json.dumps(emitted(coord_session)) + "\n" + json.dumps(emitted(worker_session)) + "\n")
    sft_out = tmp_path / "sec_sft.jsonl"
    dpo_out = tmp_path / "sec_dpo.jsonl"
    stats = dpo.extract(tx_path, out_sft=sft_out, out_dpo=dpo_out)
    assert stats["rejection_codes"] == {"REPORT_CLAIM_UNREPORTED": 1, "FINDING_UNGROUNDED": 1}

    sft_rows = [json.loads(line) for line in sft_out.read_text().splitlines()]
    # Defective report is excluded from SFT; clean report and clean worker complete_leg are kept.
    assert len(sft_rows) == 5
    clean_report_sft = next(r for r in sft_rows if not r["target"]["tool_calls"])
    assert '"id": "claim-1"' in clean_report_sft["target"]["text"]
    assert '"disposition": "act"' in clean_report_sft["target"]["text"]
    assert '"evidence": "scan#1"' in clean_report_sft["target"]["text"]
    assert clean_report_sft["closeout"] == {"outcome": "grounded", "report_phase": True}

    # Check shadow divergence rules for security workflow tools, scan accounting, and report document fences.
    surface = {"surface": {"active_tools": ["scan_summary", "scan_query", "scan_list", "fanout_plan", "workflow_advance", "complete_leg", "submit_verdict"], "loadable": []}, "prompt": []}
    assert dpo.classify_divergence(
        surface,
        {"tool_calls": [{"name": "fanout_plan", "arguments": '{"legs":[{"id":"leg-1","role":"security-reviewer"}],"threat_model":{"context":"web"}}'}, {"name": "workflow_advance", "arguments": "{}"}]},
        {"tool_calls": [{"name": "scan_summary", "arguments": '{"scan_id":"s1"}'}]},
    ) == "missed_workflow_verdict_or_plan"

    assert dpo.classify_divergence(
        surface,
        {"tool_calls": [{"name": "fanout_plan", "arguments": '{"legs":[{"id":"leg-1","role":"security-reviewer"}],"threat_model":{"context":"web"}}'}]},
        {"tool_calls": [{"name": "fanout_plan", "arguments": '{"legs":[{"id":"leg-1","role":"skeptic"}],"threat_model":{"context":"web"}}'}]},
    ) == "malformed_workflow_control_args"

    assert dpo.classify_divergence(
        surface,
        {"tool_calls": [{"name": "complete_leg", "arguments": '{"leg_status":"completed","findings":[{"path":"a.py","line":1,"excerpt":"x"}]}'}]},
        {"tool_calls": [{"name": "complete_leg", "arguments": '{"leg_status":"completed","findings":[{"path":"a.py"}]}'}]},
    ) == "malformed_workflow_control_args"

    assert dpo.classify_divergence(
        surface,
        {"tool_calls": [{"name": "scan_summary", "arguments": '{"scan_id":"s1"}'}, {"name": "scan_query", "arguments": '{"scan_id":"s1","view":"groups"}'}]},
        {"tool_calls": [{"name": "scan_list", "arguments": "{}"}]},
    ) == "missed_scan_accounting"

    assert dpo.classify_divergence(
        dict(surface, closeout=clean_report_sft["closeout"]),
        {"text": clean_report_sft["target"]["text"], "tool_calls": []},
        {"text": '## Security survey\n\nDone [scan#1].\n\n```json\n{"headline":"h","rating":{"level":"hardening","why":"w"},"findings":[],"set_asides":[],"ask":{"do":"d","effort":"s","why":"w"},"cited_evidence":[{"evidence":"scan#1"}]}\n```', "tool_calls": []},
    ) == "closeout_report_malformed"

    # Verify that if a multi-turn task settles Turn 1 and times out on Turn 2, Turn 1 and completed worker legs are still kept in SFT.
    timeout_coord = dict(
        coord_session,
        session_id="sec-coord-timeout",
        status="timeout",
        messages=coord_session["messages"] + [
            {"id": "u-followup-2", "role": "user", "content": "Now write a patch for claim-1."},
            {"id": "a-incomplete", "role": "assistant", "content": "", "tool_calls_json": json.dumps([{"id": "c9", "name": "read", "args": {"path": "src/flask/sessions.py"}}])},
        ],
    )
    timeout_worker = dict(worker_session, session_id="sec-worker-timeout", status="timeout")
    to_path = tmp_path / "sec_timeout.jsonl"
    to_path.write_text(json.dumps(emitted(timeout_coord)) + "\n" + json.dumps(emitted(timeout_worker)) + "\n")
    to_sft = tmp_path / "sec_timeout_sft.jsonl"
    dpo.extract(to_path, out_sft=to_sft)
    to_sft_rows = [json.loads(line) for line in to_sft.read_text().splitlines()]
    # The 5 completed turns from Turn 1 and the completed worker leg are retained; the incomplete Turn 2 read is excluded.
    assert len(to_sft_rows) == 5


def test_per_session_routine_read_capping_and_stage_custom_model_multi_root(factory, tmp_path):
    # Build a long session with:
    # - 1 opening read (step 0 -> always kept)
    # - 10 novel mid-loop unscoped reads (always kept)
    # - 1 mid-loop scoped read with start_line/end_line (always kept)
    # - 1 mid-loop multi-root @repo/ read (always kept)
    # - 1 mid-loop verify tool call (always kept)
    # - 1 grounded closeout (always kept)
    msgs = [{"id": "u-long", "role": "user", "content": "Audit all modules and run tests."}]
    # Opening read (idx=1, step_offset=0)
    msgs.append({
        "id": "a-open",
        "role": "assistant",
        "content": "",
        "tool_calls_json": json.dumps([{"id": "c0", "name": "read", "args": {"path": "src/mod0.py"}}]),
    })
    msgs.append({
        "id": "t-open",
        "role": "tool",
        "tool_result_json": json.dumps([{"tool_call_id": "c0", "tool_name": "read", "content": "def f0(): pass"}]),
    })
    # 10 mid-loop unscoped reads
    for i in range(1, 11):
        msgs.append({
            "id": "a-mid-%d" % i,
            "role": "assistant",
            "content": "",
            "tool_calls_json": json.dumps([{"id": "c%d" % i, "name": "read", "args": {"path": "src/mod%d.py" % i}}]),
        })
        msgs.append({
            "id": "t-mid-%d" % i,
            "role": "tool",
            "tool_result_json": json.dumps([{"tool_call_id": "c%d" % i, "tool_name": "read", "content": "def f%d(): pass" % i}]),
        })
    # Scoped read (never capped)
    msgs.append({
        "id": "a-scoped",
        "role": "assistant",
        "content": "",
        "tool_calls_json": json.dumps([{"id": "c-sc", "name": "read", "args": {"path": "src/core.py", "start_line": 10, "end_line": 40}}]),
    })
    msgs.append({
        "id": "t-scoped",
        "role": "tool",
        "tool_result_json": json.dumps([{"tool_call_id": "c-sc", "tool_name": "read", "content": "10: x = 1"}]),
    })
    # Multi-root read (never capped)
    msgs.append({
        "id": "a-mr",
        "role": "assistant",
        "content": "",
        "tool_calls_json": json.dumps([{"id": "c-mr", "name": "read", "args": {"path": "@spec/clause.md"}}]),
    })
    msgs.append({
        "id": "t-mr",
        "role": "tool",
        "tool_result_json": json.dumps([{"tool_call_id": "c-mr", "tool_name": "read", "content": "Clause 1"}]),
    })
    # Verify call (never capped)
    msgs.append({
        "id": "a-ver",
        "role": "assistant",
        "content": "",
        "tool_calls_json": json.dumps([{"id": "c-v", "name": "verify", "args": {}}]),
    })
    msgs.append({
        "id": "t-ver",
        "role": "tool",
        "tool_result_json": json.dumps([{"tool_call_id": "c-v", "tool_name": "verify", "content": "ok"}]),
    })
    # Grounded closeout (never capped)
    msgs.append({
        "id": "a-close",
        "role": "assistant",
        "content": "Audit complete [read#1].",
        "grounding_json": json.dumps({"traced": True, "cited_evidence": [{"evidence": "read#1"}]}),
    })

    session = {
        "session_id": "long-sess",
        "root_session": "long-sess",
        "workspace": "flask",
        "prompt_group": "flask:g1",
        "archetype": "compliance_audit",
        "model": "glm-5.3-flash",
        "status": "settled",
        "messages": msgs,
    }
    tx_path = tmp_path / "long_tx.jsonl"
    tx_path.write_text(json.dumps(emitted(session)) + "\n")
    sft_capped = tmp_path / "sft_capped.jsonl"
    stats = dpo.extract(tx_path, out_sft=sft_capped, max_routine_reads_per_session=3)
    assert stats["capped_routine_sft"] == 0
    # Novel evidence is not discarded merely because all calls use read.
    assert stats["sft_rows"] == 15

    # Stored archives without pinned collection lineage cannot be silently reused.
    import dataclasses

    f = dataclasses.replace(factory, root=tmp_path)
    run = runmod.Run(factory=f, name="cm-test", settings=runmod.settings_for(f, {"custom_model.enabled": True}))
    with pytest.raises(ValueError, match="epoch must name"):
        runmod.stage_custom_model(run)

    # The configured workspaces and archetypes give the planned raw task total.
    repo_per_ws = sum(a.per_workspace for a in factory.archetypes if a.workspace == "repository")
    stack_per_ws = sum(a.per_workspace for a in factory.archetypes if a.workspace == "stack")
    total_raw = len(factory.repos) * repo_per_ws + len(factory.stacks) * stack_per_ws
    assert len(factory.repos) == 16 and len(factory.stacks) == 15
    assert repo_per_ws == 222 and stack_per_ws == 50 and total_raw == 4302


def test_command_rework_surface_and_legacy_rejection_filtering(tmp_path):
    # 1. Historical rejections remain evidence in their original epoch by default.
    tx = {
        "session_id": "cmd-rework-1",
        "root_session": "cmd-rework-1",
        "workspace": "oar-gateway",
        "prompt_group": "oar-gateway:g1",
        "archetype": "compliance_audit",
        "model": "glm-5.3-flash",
        "status": "settled",
        "messages": [
            {"id": "u1", "role": "user", "content": "Inspect crates/oar-engine/src/engine.rs and run cargo test."},
            {
                "id": "a1",
                "role": "assistant",
                "content": "",
                "tool_calls_json": json.dumps([
                    {"id": "c1", "name": "command", "args": {"command": "sed -n '1,80p' crates/oar-engine/src/engine.rs"}},
                ]),
            },
            {
                "id": "t1",
                "role": "tool",
                "tool_result_json": json.dumps([
                    {
                        "tool_call_id": "c1",
                        "tool_name": "command",
                        "is_error": True,
                        "content": "Status: error\nCode: USE_READ_NATIVE\nUse native read instead.",
                    }
                ]),
            },
            {
                "id": "a2",
                "role": "assistant",
                "content": "",
                "tool_calls_json": json.dumps([
                    {"id": "c2", "name": "command", "args": {"command": "sed -n '1,80p' crates/oar-engine/src/engine.rs"}},
                ]),
            },
            {
                "id": "t2",
                "role": "tool",
                "tool_result_json": json.dumps([
                    {"tool_call_id": "c2", "tool_name": "command", "content": "[cmd#1]\n1: pub struct Engine;"}
                ]),
            },
            {
                "id": "a3",
                "role": "assistant",
                "content": "Inspected engine [command#1].",
                "grounding_json": json.dumps({"traced": True, "cited_evidence": [{"evidence": "command#1"}]}),
            },
        ],
    }
    tx_path = tmp_path / "cmd_rework.jsonl"
    tx_path.write_text(json.dumps(emitted(tx)) + "\n")
    sft_out = tmp_path / "sft.jsonl"
    dpo_out = tmp_path / "dpo.jsonl"
    stats = dpo.extract(tx_path, out_sft=sft_out, out_dpo=dpo_out)
    assert stats["dropped_retired_rejections"] == 0
    assert stats["dpo_rows"] == 1
    assert stats["sft_rows"] == 2

    # 2. classify_divergence works across the Language v1 `split` (`command`-first) surface.
    split_surface = {
        "surface": {
            "name": "verify",
            "active_tools": ["command", "command_output", "command_stop", "summarize", "survey_repo", "update_progress", "request_tools", "skills_read"],
            "loadable": ["edit", "replace_lines", "verify"],
            "required_tools": ["survey_repo"],
        },
        "prompt": [],
    }
    # Calling `repo-map` inside `command` satisfies `survey_repo` via COMMAND_BUILTIN_EQUIVALENTS.
    assert dpo.classify_divergence(
        split_surface,
        {"tool_calls": [{"name": "survey_repo", "arguments": '{"path":"."}'}]},
        {"tool_calls": [{"name": "command", "arguments": '{"command":"repo-map ."}'}]},
    ) is None

    # Calling `ls -la` inside `command` when teacher runs `cargo test` or reads code is `shallow_investigation`.
    assert dpo.classify_divergence(
        {"surface": {"active_tools": ["command"], "loadable": []}, "prompt": []},
        {"tool_calls": [{"name": "command", "arguments": '{"command":"cargo test --workspace"}'}]},
        {"tool_calls": [{"name": "command", "arguments": '{"command":"ls -la"}'}]},
    ) == "shallow_investigation"

    # Calling unscoped `cat` when teacher uses scoped `sed -n '10,50p'` is `unscoped_read`.
    assert dpo.classify_divergence(
        {"surface": {"active_tools": ["command"], "loadable": []}, "prompt": []},
        {"tool_calls": [{"name": "command", "arguments": '{"command":"sed -n \'10,50p\' src/engine.rs"}'}]},
        {"tool_calls": [{"name": "command", "arguments": '{"command":"cat src/engine.rs"}'}]},
    ) == "unscoped_read"

    # Mutating `command` without `update_progress` is `missing_progress_before_edit`; read-only `command` is `unclosed_progress_before_closeout`.
    assert dpo.classify_divergence(
        {"surface": {"active_tools": ["command", "update_progress"], "loadable": []}, "prompt": []},
        {"tool_calls": [{"name": "update_progress", "arguments": '{"markdown":"- [ ] Patch file"}'}]},
        {"tool_calls": [{"name": "command", "arguments": '{"command":"sed -i \'s/a/b/\' src/engine.rs"}'}]},
    ) == "missing_progress_before_edit"
    assert dpo.classify_divergence(
        {"surface": {"active_tools": ["command", "update_progress"], "loadable": []}, "prompt": []},
        {"tool_calls": [{"name": "update_progress", "arguments": '{"markdown":"- [x] Done"}'}]},
        {"tool_calls": [{"name": "command", "arguments": '{"command":"cat src/engine.rs"}'}]},
    ) == "unclosed_progress_before_closeout"

    # Mutating `command` on a read-only surface (`observe_verify` / `worker_readonly`) is `readonly_command_mutation`.
    assert dpo.classify_divergence(
        {"surface": {"name": "worker_readonly", "active_tools": ["command"], "loadable": []}, "prompt": []},
        {"tool_calls": [{"name": "command", "arguments": '{"command":"grep -rn SECRET_KEY src/"}'}]},
        {"tool_calls": [{"name": "command", "arguments": '{"command":"cat <<\'EOF\' > src/sessions.py\\nx=1\\nEOF"}'}]},
    ) == "readonly_command_mutation"

    # Historical translation and synthetic attached-root negatives cannot qualify.
    sft_rows = [json.loads(line) for line in sft_out.read_text().splitlines()]
    with pytest.raises(ValueError, match="real attached-root trajectories"):
        dpo.synthesize_multi_root_pairs(sft_rows, companion_label="open-agent-rules")
    with pytest.raises(ValueError, match="fresh command-surface run"):
        dpo.translate_legacy_turn_to_command({"tool_calls": []})
    with pytest.raises(ValueError, match="fresh command-surface run"):
        dpo.extract(tx_path, translate_legacy_tools=True)


def test_supported_44_language_matrix_coverage():
    assert len(tasks.SUPPORTED_LANGUAGES) == 44
    assert len(tasks.SECURITY_SUPPORTED_LANGUAGES) == 37
    assert set(tasks.LANGUAGE_FILE_EXTENSIONS) == set(tasks.SUPPORTED_LANGUAGES)

    # Every supported language's canonical extension is recognized by PATHLIKE / mentioned_paths_exist.
    for lang, exts in tasks.LANGUAGE_FILE_EXTENSIONS.items():
        sample_file = "module.%s" % exts[0]
        assert tasks.mentioned_paths_exist("Inspect %s and explain its structure." % sample_file, [sample_file]), lang
        assert not tasks.mentioned_paths_exist("Inspect missing_%s" % sample_file, [sample_file]), lang

    # Native test, build, and lint commands of every supported language are recognized.
    matrix_test_commands = [
        "pytest -q",
        "python3 -m unittest discover",
        "vitest run",
        "deno test",
        "go test ./...",
        "cargo test --workspace",
        "scarb test",
        "./gradlew test",
        "mvn test",
        "sbt test",
        "lein test",
        "dotnet test",
        "Invoke-Pester",
        "bundle exec rspec",
        "vendor/bin/phpunit",
        "hh_client",
        "ctest --output-on-failure",
        "meson test -C build",
        "dart test",
        "mix test",
        "swift test",
        "dune runtest",
        "prove -lv t/",
        "busted",
        "julia --project=. -e 'using Pkg; Pkg.test()'",
        "R CMD check .",
        "sbcl --eval '(asdf:test-system :app)'",
        "raco test .",
        "forge test",
        "circom circuit.circom --r1cs",
        "aptos move test",
        "terraform test",
        "docker build .",
        "buf lint",
        "jsonnet config.jsonnet",
        "promtool test rules alerts.rules",
        "codeql test run queries/",
        "sf apex run test",
        "bats test.bats",
        "xmllint --noout pom.xml",
        "yamllint config.yaml",
    ]
    for cmd in matrix_test_commands:
        call = {"name": "command", "arguments": json.dumps({"command": cmd})}
        assert dpo._is_exec_or_test_command(call), cmd

    # Package-manager mutations are recognized as mutating commands.
    matrix_pkg_commands = [
        "npm install zod",
        "pnpm add fastify",
        "pip install httpx",
        "uv add pydantic",
        "cargo add serde",
        "go get github.com/go-chi/chi/v5",
        "dotnet add package Newtonsoft.Json",
        "bundle add sinatra",
        "composer require slim/slim",
        "dart pub add http",
        "mix archive.install hex phx_new",
        "cpanm Plack",
        "luarocks install busted",
        "opam install dune",
        "conan install . --build=missing",
        "scarb add alexandria_math",
        "Install-Module -Name Pester",
    ]
    for cmd in matrix_pkg_commands:
        call = {"name": "command", "arguments": json.dumps({"command": cmd})}
        assert dpo._is_mutating_command(call), cmd


def _closeout_session(reply, *before):
    return {
        "session_id": "s", "root_session": "s", "status": "settled", "model": "teacher",
        "messages": [
            {"id": "u1", "role": "user", "origin": "user", "content": "Inspect a.py"},
            {"id": "a1", "role": "assistant", "origin": "model", "content": "",
             "tool_calls_json": json.dumps([{"id": "c1", "name": "read", "args": {"path": "a.py"}}])},
            {"id": "t1", "role": "tool", "origin": "tool",
             "tool_result_json": json.dumps({"tool_call_id": "c1", "tool_name": "read", "is_error": False, "content": "x"})},
            *before,
            reply,
        ],
    }


def _extract(tmp_path, *sessions):
    source, sft, pairs = tmp_path / "tx.jsonl", tmp_path / "sft.jsonl", tmp_path / "dpo.jsonl"
    source.write_text("".join(json.dumps(emitted(session)) + "\n" for session in sessions))
    stats = dpo.extract(source, out_sft=sft, out_dpo=pairs)
    return stats, [json.loads(line) for line in sft.read_text().splitlines()], [json.loads(line) for line in pairs.read_text().splitlines()]


def test_only_host_traced_citations_make_a_reply_a_teacher_example(tmp_path):
    cited = [{"evidence": "read#1"}]
    for grounding, outcome, kept in (
        ({"traced": True, "cited_evidence": cited}, "grounded", True),
        ({"traced": True, "cited_urls": ["https://example.test/a"]}, "grounded", True),
        ({"traced": True, "evidence_optional": True}, "evidence_optional", False),
        ({"traced": True, "evidence_optional": True, "cited_evidence": cited}, "evidence_optional", False),
        ({"traced": True, "host_assembled": True, "cited_evidence": cited}, "host_assembled", False),
        ({"traced": False, "cited_evidence": cited}, "host_assembled", False),
        ({"traced": True}, "uncited", False),
    ):
        # The text claims citations in every case; only the row's facts count.
        reply = {"id": "a2", "role": "assistant", "origin": "model", "content": "Done [read#1].", "grounding_json": json.dumps(grounding)}
        assert dpo.closeout_outcome(reply) == outcome
        assert dpo.has_grounded_citations(reply) is (outcome == "grounded")
        stats, sft, _ = _extract(tmp_path, _closeout_session(reply))
        assert any(not row["target"]["tool_calls"] for row in sft) is kept, grounding
        assert stats["evidence_optional_replies"] == (1 if outcome == "evidence_optional" else 0)
    plain = {"id": "a2", "role": "assistant", "origin": "model", "content": "Done [read#1]."}
    assert dpo.closeout_outcome(plain) is None and not dpo.has_grounded_citations(plain)
    defective = dict(plain, grounding_json=json.dumps({"traced": True, "cited_evidence": cited}),
                     completion_report_json=json.dumps({"scope": "run", "defects": [{"code": "REPORT_CLAIM_UNREPORTED"}]}))
    assert dpo.closeout_outcome(defective) == "report_defect"


def test_a_row_with_no_model_output_is_not_a_teacher_turn(tmp_path):
    host_written = {"id": "a2", "role": "assistant", "origin": "host", "content": "Host assembled reply.",
                    "model_output": None, "grounding_json": json.dumps({"traced": True, "host_assembled": True})}
    stats, sft, pairs = _extract(tmp_path, _closeout_session(host_written))
    assert stats["without_model_output"] == 1
    assert all("Host assembled" not in json.dumps(row["target"]) for row in sft)
    assert not pairs
    # Its retracted attempts are still rejected model text.
    session = dict(_closeout_session(host_written), draft_versions=[
        {"slot_id": "a2", "version_index": 0, "body": "Done [a.go:999].", "outcome_code": "INVEST_HANDLE_NOT_OBSERVED"}])
    _, _, pairs = _extract(tmp_path, session)
    assert [p["rejected"]["text"] for p in pairs] == ["Done [a.go:999]."]


def test_published_notes_are_not_counted_with_their_carrier(tmp_path):
    carrier = {"id": "a2", "role": "assistant", "origin": "model", "content": "Reading b.py after [read#1:1].",
               "tool_calls_json": json.dumps([{"id": "c2", "name": "read", "args": {"path": "b.py"}}])}
    note = {"id": "note", "role": "assistant", "kind": "agent_note", "origin": "model", "content": "Reading b.py after a.py."}
    result = {"id": "t2", "role": "tool", "origin": "tool",
              "tool_result_json": json.dumps({"tool_call_id": "c2", "tool_name": "read", "is_error": False, "content": "y"})}
    reply = {"id": "a3", "role": "assistant", "origin": "model", "content": "Done [read#1].",
             "grounding_json": json.dumps({"traced": True, "cited_evidence": [{"evidence": "read#1"}]})}
    stats, sft, _ = _extract(tmp_path, _closeout_session(reply, carrier, note, result))
    assert [row["message_id"] for row in sft] == ["a1", "a2", "a3"]
    assert stats["without_model_output"] == 0
    # The carrier's tool result is still found behind its note.
    assert sft[1]["target"]["text"] == "Reading b.py after [read#1:1]."
    assert "Reading b.py after a.py." not in json.dumps(sft)


def test_withheld_text_is_feedback_not_an_answer_or_a_rejection(tmp_path):
    carrier = {"id": "a2", "role": "assistant", "origin": "model", "content": "Reading b.py next.",
               "tool_calls_json": json.dumps([{"id": "c2", "name": "read", "args": {"path": "b.py"}}])}
    result = {"id": "t2", "role": "tool", "origin": "tool",
              "tool_result_json": json.dumps({"tool_call_id": "c2", "tool_name": "read", "is_error": False, "content": "y"})}
    notice = {"id": "n", "role": "user", "origin": "host",
              "content": ">>> Tool feedback\n[HOST: MESSAGE WITHHELD FROM USER] Your text cited nothing.\nCode: MID_TURN_NOTE_UNGROUNDED"}
    reply = {"id": "a3", "role": "assistant", "origin": "model", "content": "Done [read#1].",
             "grounding_json": json.dumps({"traced": True, "cited_evidence": [{"evidence": "read#1"}]})}
    stats, sft, pairs = _extract(tmp_path, _closeout_session(reply, carrier, result, notice))
    assert stats["withheld_carriers"] == {"MID_TURN_NOTE_UNGROUNDED": 1}
    assert stats["rejection_codes"] == {} and not pairs
    assert [row["message_id"] for row in sft] == ["a1", "a3"]
    assert dpo.parse_host_user_rejection(notice) is None
    # The notice opens no new request, even with no origin on the row.
    bare = {key: value for key, value in notice.items() if key != "origin"}
    _, sft, _ = _extract(tmp_path, _closeout_session(reply, carrier, result, bare))
    assert [row["message_id"] for row in sft] == ["a1", "a3"]
    assert not dpo._is_user_opening(bare)


def test_rejection_codes_read_plain_lines_only():
    # A rejection's wording never names its code.
    assert dpo.parse_host_user_rejection({"role": "user", "content": "Rejected: 2 progress steps still pending."})["code"] == "TURN_CLOSEOUT_REJECTED"
    assert dpo.REJECTION_CODE_RE.search("Rejected: no\nCode: OUTSIDE_ROOTS").group(1) == "OUTSIDE_ROOTS"
    assert dpo.REJECTION_CODE_RE.search("⟦D⟧Code: OUTSIDE_ROOTS") is None
    assert dpo.parse_rejection({"is_error": True, "content": "Status: error\nCode: PROGRESS_MISSING"})["code"] == "PROGRESS_MISSING"
