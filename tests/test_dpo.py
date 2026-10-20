import dataclasses
import gzip
import json
import sqlite3

from conftest import row

from bialy import cli, config, dpo, fleet, split, tasks
from bialy import run as runmod


class FakeStudentChat:
    """Returns canned non-thinking tool turns keyed by a handler function."""

    def __init__(self, fn):
        self.fn = fn
        self.calls = []

    def turn(self, messages, tools=None, temperature=0.0, seed=None, max_tokens=4096, thinking=False):
        self.calls.append({"messages": messages, "tools": tools, "temperature": temperature, "seed": seed, "thinking": thinking})
        return self.fn(messages, tools)


def _sample_transcript(session_id="s1", workspace="flask", group="flask:g1", model="gemma-4-31b-it", status="settled"):
    return {
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
            {"id": "m1", "role": "user", "turn_kind": "user", "text": "Inspect ctx.py and summarize the bug.", "content": [], "created_at": "1"},
            {
                "id": "m2",
                "role": "assistant",
                "turn_kind": None,
                "text": "Let me read the symbol.",
                "content": [
                    {"type": "reasoning", "text": "internal chain of thought to strip"},
                    {"type": "text", "text": "Let me read the symbol."},
                    {
                        "type": "tool_calls",
                        "calls": [
                            {"id": "c0", "name": "request_tools", "arguments": json.dumps({"names": ["summarize"]})},
                            {"id": "c1", "name": "skills_read", "arguments": json.dumps({"name": "commit"})},
                            {"id": "c2", "name": "read", "arguments": json.dumps({"file_path": "src/flask/ctx.py", "symbol": "AppContext"})},
                            {"id": "c3", "name": "summarize", "arguments": json.dumps({"file_path": "src/flask/ctx.py"})},
                        ],
                    },
                ],
                "created_at": "2",
            },
            {
                "id": "m3",
                "role": "tool",
                "turn_kind": None,
                "text": "",
                "content": [
                    {
                        "type": "tool_results",
                        "results": [
                            {"tool_call_id": "c0", "tool_name": "request_tools", "content": "loaded summarize", "is_error": False},
                            {"tool_call_id": "c1", "tool_name": "skills_read", "content": "skill card", "is_error": False},
                            {"tool_call_id": "c2", "tool_name": "read", "content": "class AppContext: ...", "is_error": False},
                            {"tool_call_id": "c3", "tool_name": "summarize", "content": "summary...", "is_error": False},
                        ],
                    }
                ],
                "created_at": "3",
            },
            {
                "id": "m4",
                "role": "assistant",
                "turn_kind": None,
                "text": "Trying to edit outside workspace.",
                "content": [
                    {"type": "reasoning", "text": "thinking before bad call"},
                    {"type": "text", "text": "Trying to edit outside workspace."},
                    {"type": "tool_calls", "calls": [{"id": "c4", "name": "edit", "arguments": json.dumps({"file_path": "../secret.py"})}]},
                ],
                "created_at": "4",
            },
            {
                "id": "m5",
                "role": "tool",
                "turn_kind": None,
                "text": "",
                "content": [
                    {
                        "type": "tool_results",
                        "results": [
                            {
                                "tool_call_id": "c4",
                                "tool_name": "edit",
                                "content": "Status: error\nTool: edit\nCode: outside_roots\nSummary: Path is outside allowed roots.",
                                "is_error": True,
                                "details": {"paintedwolf.confinement": {"blocked_path": "../secret.py"}},
                            }
                        ],
                    }
                ],
                "created_at": "5",
            },
            {
                "id": "m6",
                "role": "assistant",
                "turn_kind": None,
                "text": "Editing inside src/flask/ctx.py instead.",
                "content": [
                    {"type": "reasoning", "text": "recovering after outside_roots"},
                    {"type": "text", "text": "Editing inside src/flask/ctx.py instead."},
                    {"type": "tool_calls", "calls": [{"id": "c5", "name": "edit", "arguments": json.dumps({"file_path": "src/flask/ctx.py"})}]},
                ],
                "created_at": "6",
            },
            {
                "id": "m7",
                "role": "tool",
                "turn_kind": None,
                "text": "",
                "content": [{"type": "tool_results", "results": [{"tool_call_id": "c5", "tool_name": "edit", "content": "ok", "is_error": False}]}],
                "created_at": "7",
            },
        ],
    }


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

    dpo_rows = [json.loads(line) for line in dpo_out.read_text().splitlines()]
    prevent, recover = dpo_rows
    assert prevent["source"] == "host_rejection" and prevent["rejection_code"] == "outside_roots"
    assert prevent["chosen"]["tool_calls"][0]["arguments"] == json.dumps({"file_path": "src/flask/ctx.py"})
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
    assert report["evaluated"] == 4 and report["pairs"] >= 2
    assert all(c["thinking"] is False and c["temperature"] == 0.0 for c in student_chat.calls)

    # Re-running with the same signature reuses the partial file without making new LLM calls.
    calls_before = len(student_chat.calls)
    report_again = dpo.shadow(factory, sft_out, shadow_out, student_id="gemma-4-e4b-it", workers=2, seed=7)
    assert len(student_chat.calls) == calls_before and report_again["pairs"] == report["pairs"]

    # Splitting shadow pairs preserves held-out workspace isolation.
    split_counts = split.split(factory, [str(shadow_out)], tmp_path / "dpo_split")
    assert split_counts["holdout"]["rows"] >= 1
    holdout_workspaces = {json.loads(line)["meta"]["workspace"] for line in (tmp_path / "dpo_split" / "holdout.jsonl").read_text().splitlines()}
    assert holdout_workspaces == {"zod"}


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
    assert rep == {"input_rows": 1, "repaired": 1, "completed_pairs": 1}
    repaired_row = json.loads(repaired_out.read_text())
    assert repaired_row["chosen"]["tool_calls"][0]["name"] == "edit"
    assert teacher_chat.calls[0]["thinking"] is False


def test_export_transcripts_from_shard_sqlite_store(tmp_path):
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
    conn.execute("CREATE TABLE sessions (id TEXT, parent_session_id TEXT, title TEXT, workflow_id TEXT, created_at TEXT)")
    conn.execute("CREATE TABLE messages (id TEXT, session_id TEXT, role TEXT, turn_kind TEXT, text TEXT, content_json TEXT, created_at TEXT)")
    conn.execute("INSERT INTO sessions VALUES ('s1', NULL, 'test', NULL, '1')")
    conn.execute("INSERT INTO messages VALUES ('m1', 's1', 'user', 'user', 'hi', '[]', '1')")
    conn.commit()
    conn.close()

    out = tmp_path / "transcripts.jsonl.gz"
    assert cli.main(["dpo", "export", "--run", str(tmp_path / "run"), "--tasks", str(tasks_dir), "--out", str(out)]) == 0
    records = list(dpo.iter_transcripts(out))
    assert len(records) == 1 and records[0]["workspace"] == "flask" and records[0]["model"] == "gemma-4-31b-it"


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
    assert scaled.judge_for("gemma-4-31b-it").id == "qwen3.6-35b-a3b"
    assert scaled.judge_for("qwen3.6-35b-a3b").id == "gemma-4-31b-it"
    assert scaled.judge_for("glm-5.3-flash").id in {"qwen3.6-35b-a3b", "gemma-4-31b-it"}

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
        json.dumps({"schema": "pw-decide-request/1", "model": "accounts/fireworks/models/glm-5p3-flash", "input_tokens": 1000000, "output_tokens": 1000000})
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
        {"role": "tool", "tool_call_id": "c3", "name": "edit", "content": "⟦D⟧Rejected:\n⟦D⟧Code: PROGRESS_MISSING"},
    ]
    compacted = dpo.compact_prompt(msgs, max_chars=800)
    assert len(compacted) == len(msgs)
    assert compacted[0]["content"] == "system prompt with tool rules"
    assert "file already read in transcript" in compacted[3]["content"]
    # Recent tail turns and rejection cards are never compacted.
    assert compacted[-1]["content"] == "⟦D⟧Rejected:\n⟦D⟧Code: PROGRESS_MISSING"
    rej = dpo.parse_rejection({"content": compacted[-1]["content"], "is_error": True})
    assert rej["code"] == "PROGRESS_MISSING"


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
             "content": "Rejected: This tool-free response would end the turn with 2 progress steps still pending.\n\nCause: Work remains unfinished."},
            {"id": "a3", "role": "assistant", "origin": "model", "content": "",
             "tool_calls_json": json.dumps([{"name": "update_progress", "id": "c2", "args": {"content": "## Progress\n- [x] Done"}}])},
            {"id": "t2", "role": "tool", "origin": "tool", "content": "Updated progress.", "tool_result_json": json.dumps({"content": "Updated progress."})},
        ],
    }
    tpath = tmp_path / "real_transcripts.jsonl"
    tpath.write_text(json.dumps(transcript) + "\n")
    sft_out = tmp_path / "sft.jsonl"
    dpo_out = tmp_path / "dpo.jsonl"
    stats = dpo.extract(tpath, out_sft=sft_out, out_dpo=dpo_out)
    assert stats["sft_rows"] == 2
    assert stats["rejection_codes"] == {"TURN_CLOSEOUT_PENDING_PROGRESS": 1}
    dpo_rows = [json.loads(line) for line in dpo_out.read_text().splitlines()]
    assert {r["source"] for r in dpo_rows} == {"host_rejection", "post_rejection_recovery"}
    prevent = next(r for r in dpo_rows if r["source"] == "host_rejection")
    assert prevent["rejected"]["tool_calls"] == [] and prevent["chosen"]["tool_calls"][0]["name"] == "update_progress"

    # 2. design_review archetype and Den attachment block generation.
    arch_by_id = {a.id: a for a in factory.archetypes}
    assert "design_review" in arch_by_id and arch_by_id["design_review"].attachment_rate > 0
    files = ["src/flask/ctx.py", "src/flask/app.py", "README.md"]
    raw = [{
        "prompt": "Don't make changes yet. Review src/flask/ctx.py and tell me how we should restructure it.",
        "follow_ups": ["What if we keep the existing signature?", "Do as you would recommend.", "Now commit the changes."],
        "archetype": "design_review",
        "lang": "en",
        "author": "m",
    }]
    forced = dataclasses.replace(factory, archetypes=[dataclasses.replace(a, attachment_rate=1.0, follow_up_rate=1.0) for a in factory.archetypes])
    tasks.finalize(forced, forced.repos[0], raw, files, tmp_path / "design.jsonl", seed=7)
    [task] = [json.loads(line) for line in (tmp_path / "design.jsonl").read_text().splitlines()]
    assert '```attachment filename="ctx.py"' in task["prompt"]
    assert "[User attached file: src/flask/ctx.py]" in task["prompt"]
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
            {"id": "a3", "role": "assistant", "origin": "model", "content": "Report body citing bare path.\n```citations\n{\"cited_evidence\": [\"src/engine.rs\"]}\n```"},
            {"id": "u2", "role": "user", "origin": "host",
             "content": "[host:coordinator-citation-grounding]\n\nTo repair citations, emit only the citations fence; the host keeps the pinned body. Attempt 2/3.\n\nRejected: src/engine.rs (Code: INVEST_HANDLE_NOT_OBSERVED)"},
            {"id": "a4", "role": "assistant", "origin": "model", "content": "```citations\n{\"cited_evidence\": [\"read#1\", \"tui#1\"]}\n```"},
        ],
    }
    cit_path = tmp_path / "cit_tx.jsonl"
    cit_path.write_text(json.dumps(cit_tx) + "\n")
    cit_sft = tmp_path / "cit_sft.jsonl"
    cit_dpo = tmp_path / "cit_dpo.jsonl"
    cit_stats = dpo.extract(cit_path, out_sft=cit_sft, out_dpo=cit_dpo)
    assert cit_stats["rejection_codes"] == {"COORDINATOR_CITATION_GROUNDING": 1}
    assert dpo.parse_host_user_rejection({"role": "user", "content": ">>> Tool feedback\nGap.\nCode: AMBIENT_UNGROUNDED_COMPLETION"})["code"] == "AMBIENT_UNGROUNDED_COMPLETION"
    cit_sft_rows = [json.loads(line) for line in cit_sft.read_text().splitlines()]
    assert any("```citations" in (r["target"].get("text") or "") for r in cit_sft_rows)
    cit_dpo_rows = [json.loads(line) for line in cit_dpo.read_text().splitlines()]
    cit_recover = next(r for r in cit_dpo_rows if r["source"] == "post_rejection_recovery")
    assert '"read#1"' in cit_recover["chosen"]["text"]

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


