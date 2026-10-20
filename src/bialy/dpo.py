"""SFT and DPO dataset construction for custom small-model training.

Extracts non-thinking tool-calling trajectories (pw-custom-sft/1) and
multi-source preference pairs (pw-custom-dpo/1) from shard SQLite stores or
released transcript archives joined with cross-family judge labels:
  1. Host-rejection prevention pairs at s_t (chosen = teacher/repaired call,
     rejected = call that triggered a host Code: rejection or nudge).
  2. Post-rejection recovery pairs at s_{t+1} (chosen = recovery call after
     reading the structured rejection card, rejected = repeating the bad call).
  3.Dropped request_tools follow-through pairs (after == []).
  4. Shadow-mode divergence pairs at teacher states s_t (student e4b queried
     with thinking=False diverges on loadable tools, skills_read, scoped read,
     or premature stops).
"""

import concurrent.futures
import gzip
import hashlib
import json
import re
import sqlite3
import threading
from pathlib import Path

from . import provenance

SFT_SCHEMA = "pw-custom-sft/1"
DPO_SCHEMA = "pw-custom-dpo/1"

REJECTION_CODE_RE = re.compile(r"(?:^|\n)(?:⟦D⟧)?Code:\s*([^\s,]+)", re.MULTILINE)
STATUS_ERROR_RE = re.compile(r"(?:^|\n)(?:Status:\s*error|(?:⟦D⟧)?Rejected:)", re.MULTILINE)
SCOPED_READ_KEYS = {"ranges", "symbol", "start_line", "end_line", "offset", "limit"}


def is_coordinator_feedback(text):
    return isinstance(text, str) and text.strip().startswith(">>> Tool feedback")


def compact_prompt(messages, max_chars=24000):
    """Compact older tool outputs while preserving system instructions, read-file receipts, and recent turns.

    Unlike raw token-slice truncation (p_ids[:head] + p_ids[-tail:]), this never severs
    the <tools> declaration header or erases which files have already been read.
    """
    total = sum(len(json.dumps(m, ensure_ascii=False)) for m in messages)
    if total <= max_chars or len(messages) <= 6:
        return list(messages)
    head = []
    idx = 0
    while idx < len(messages) and messages[idx].get("role") in ("system", "user") and idx < 2:
        head.append(messages[idx])
        idx += 1
    tail = list(messages[-4:])
    middle = messages[idx:-4]
    compacted_mid = []
    for m in middle:
        if m.get("role") == "tool":
            content = str(m.get("content") or "")
            if len(content) > 320 and not STATUS_ERROR_RE.search(content) and not REJECTION_CODE_RE.search(content):
                first_line = content.splitlines()[0] if content.splitlines() else ""
                m = dict(m, content="%s\n[...compacted %d chars; file already read in transcript...]" % (first_line[:160], len(content)))
        compacted_mid.append(m)
    return head + compacted_mid + tail



def _open_jsonl(path):
    path = Path(path)
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, encoding="utf-8")


def export_transcripts(run_dir, tasks_dir, out_path):
    """Export sessions/transcripts.jsonl(.gz) from a pass directory's shard SQLite stores."""
    run_dir, out_path = Path(run_dir), Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    specs = {}
    for path in Path(tasks_dir).glob("*.jsonl"):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                task = json.loads(line)
                specs[task["id"]] = task
    opener = (lambda p: gzip.open(p, "wt", encoding="utf-8")) if out_path.suffix == ".gz" else (lambda p: open(p, "w", encoding="utf-8"))
    sessions_written = 0
    with opener(out_path) as out:
        shards_dir = run_dir / "shards"
        for shard in sorted(shards_dir.iterdir()) if shards_dir.exists() else []:
            db_path = shard / "store.db"
            manifest_path = shard / "manifest.jsonl"
            if not db_path.exists() or not manifest_path.exists():
                continue
            by_root = {}
            for line in manifest_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                entry = json.loads(line)
                if entry.get("root_session"):
                    by_root[entry["root_session"]] = (entry, specs.get(entry["task_id"], {}))
            conn = sqlite3.connect(str(db_path))
            conn.row_factory = sqlite3.Row
            try:
                sess_rows = conn.execute("SELECT id, parent_session_id, title, workflow_id FROM sessions ORDER BY created_at, id").fetchall()
                parent_of = {r["id"]: r["parent_session_id"] for r in sess_rows}

                def root_of(sid, parent_of=parent_of):
                    cur = sid
                    seen = set()
                    while parent_of.get(cur) and cur not in seen:
                        seen.add(cur)
                        cur = parent_of[cur]
                    return cur

                msg_cols = {r[1] for r in conn.execute("PRAGMA table_info(messages)").fetchall()}
                for srow in sess_rows:
                    sid = srow["id"]
                    root = root_of(sid)
                    if root not in by_root:
                        continue
                    mentry, spec = by_root[root]
                    meta = spec.get("meta") or mentry.get("meta") or {}
                    if "tool_calls_json" in msg_cols:
                        extra_cols = [
                            c for c in ("origin", "grounding_json", "artifact_ids_json", "completion_report_json", "draft_version_count", "draft_status")
                            if c in msg_cols
                        ]
                        col_sql = ", ".join(["id", "role", "content", "kind", "visibility", "seq", "ord", "ts", "tool_calls_json", "tool_result_json", "reasoning_json"] + extra_cols)
                        msgs = conn.execute(
                            "SELECT %s FROM messages WHERE session_id = ? ORDER BY seq, ord, rowid" % col_sql,
                            (sid,),
                        ).fetchall()
                        messages = [dict(m) for m in msgs]
                    else:
                        msgs = conn.execute(
                            "SELECT id, role, turn_kind, text, content_json, created_at FROM messages WHERE session_id = ? ORDER BY created_at, rowid",
                            (sid,),
                        ).fetchall()
                        messages = []
                        for m in msgs:
                            try:
                                content = json.loads(m["content_json"]) if m["content_json"] else []
                            except ValueError:
                                content = []
                            messages.append({
                                "id": m["id"],
                                "role": m["role"],
                                "turn_kind": m["turn_kind"],
                                "text": m["text"] or "",
                                "content": content,
                                "created_at": m["created_at"],
                            })
                    draft_versions = []
                    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
                    if "draft_versions" in tables:
                        drows = conn.execute(
                            "SELECT slot_id, version_index, body, outcome_code, created_at "
                            "FROM draft_versions WHERE session_id = ? ORDER BY created_at, version_index",
                            (sid,),
                        ).fetchall()
                        draft_versions = [dict(d) for d in drows]
                    record = {
                        "shard": shard.name,
                        "session_id": sid,
                        "parent_session_id": srow["parent_session_id"],
                        "root_session": root,
                        "task_id": mentry["task_id"],
                        "workspace": meta.get("workspace", ""),
                        "prompt_group": meta.get("prompt_group", "%s:%s" % (meta.get("workspace", "unknown"), mentry["task_id"])),
                        "archetype": meta.get("archetype", ""),
                        "model": spec.get("model") or mentry.get("model", ""),
                        "status": mentry.get("status", ""),
                        "title": srow["title"] or "",
                        "workflow_id": srow["workflow_id"],
                        "messages": messages,
                        "draft_versions": draft_versions,
                    }
                    out.write(json.dumps(record, ensure_ascii=False) + "\n")
                    sessions_written += 1
            finally:
                conn.close()
    return {"sessions": sessions_written, "out": str(out_path)}


def iter_transcripts(path):
    """Yield session transcript records from a .jsonl or .jsonl.gz file."""
    with _open_jsonl(path) as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def _parse_json_field(val):
    if not val:
        return None
    if isinstance(val, (dict, list)):
        return val
    if isinstance(val, str):
        try:
            return json.loads(val)
        except ValueError:
            return None
    return None


def is_host_assembled_closeout(message):
    """Return True when a completion_report row was host-assembled after model citation/closeout failure."""
    grounding = _parse_json_field(message.get("grounding_json") or message.get("grounding"))
    if not isinstance(grounding, dict):
        return False
    return bool(grounding.get("host_assembled")) or (grounding.get("traced") is False)


def report_document_defect_code(message):
    """Return the first fatal defect code from completion_report_json.defects when a run report failed validation."""
    comp_meta = _parse_json_field(message.get("completion_report_json") or message.get("completion_report"))
    if not isinstance(comp_meta, dict):
        return None
    defects = comp_meta.get("defects")
    if not isinstance(defects, list) or not defects:
        return None
    for d in defects:
        if isinstance(d, dict) and d.get("code"):
            return str(d["code"])
    return "REPORT_DOCUMENT_INVALID"


def reconstruct_closeout_fence(message):
    """Reconstruct the trailing ```json closeout fence stripped from messages.content by closeoutWireContent."""
    grounding = _parse_json_field(message.get("grounding_json") or message.get("grounding"))
    if not isinstance(grounding, dict):
        return ""
    if not grounding.get("traced") or grounding.get("host_assembled"):
        return ""
    fence_obj = {}
    cited_ev = []
    for ce in grounding.get("cited_evidence") or []:
        if not isinstance(ce, dict):
            continue
        item = {}
        handle = ce.get("evidence") or ce.get("handle")
        if handle:
            item["evidence"] = handle
        if ce.get("path"):
            item["path"] = ce["path"]
        if ce.get("line"):
            item["line"] = ce["line"]
        if ce.get("excerpt"):
            item["excerpt"] = ce["excerpt"]
        if item:
            cited_ev.append(item)
    if cited_ev:
        fence_obj["cited_evidence"] = cited_ev
    if grounding.get("cited_urls"):
        fence_obj["cited_urls"] = list(grounding["cited_urls"])
    artifact_ids = _parse_json_field(message.get("artifact_ids_json") or message.get("artifact_ids"))
    if isinstance(artifact_ids, list) and artifact_ids:
        fence_obj["artifact_ids"] = artifact_ids
    if isinstance(grounding.get("verification"), dict) and grounding["verification"]:
        fence_obj["verification"] = grounding["verification"]
    comp_meta = _parse_json_field(message.get("completion_report_json") or message.get("completion_report"))
    if isinstance(comp_meta, dict):
        for k in ("headline", "summary", "limits", "ask", "rating", "set_asides"):
            if comp_meta.get(k):
                fence_obj[k] = comp_meta[k]
        if comp_meta.get("findings"):
            findings = []
            for f in comp_meta["findings"]:
                if not isinstance(f, dict):
                    continue
                fc = dict(f)
                if "where" in fc and isinstance(fc["where"], list):
                    where_list = []
                    for w in fc["where"]:
                        if isinstance(w, dict):
                            wc = {}
                            if w.get("evidence") or w.get("handle"):
                                wc["evidence"] = w.get("evidence") or w.get("handle")
                            if w.get("path"):
                                wc["path"] = w["path"]
                            if w.get("line"):
                                wc["line"] = w["line"]
                            if wc:
                                where_list.append(wc)
                    fc["where"] = where_list
                findings.append(fc)
            if findings:
                fence_obj["findings"] = findings
    if not fence_obj:
        return ""
    return "```json\n%s\n```" % json.dumps(fence_obj, ensure_ascii=False)


def render_draft_version_body(body_str):
    """Render a retracted draft_versions.body JSON envelope back into the assistant's Markdown + trailing fence."""
    parsed = _parse_json_field(body_str)
    if not isinstance(parsed, dict):
        return str(body_str or "").strip()
    data = dict(parsed)
    synthesis = str(data.pop("synthesis", "") or "").strip()
    if not data:
        return synthesis
    fence = "```json\n%s\n```" % json.dumps(data, ensure_ascii=False)
    if not synthesis:
        return fence
    return "%s\n\n%s" % (synthesis, fence)


def strip_thinking(message):
    """Return an assistant turn with reasoning blocks removed and text + tool_calls normalized."""
    text_parts = []
    tool_calls = []
    raw_content = message.get("content")
    if isinstance(raw_content, list):
        for block in raw_content:
            btype = block.get("type")
            if btype == "reasoning":
                continue
            if btype == "text" and block.get("text"):
                text_parts.append(block["text"].strip())
            elif btype == "tool_calls":
                for i, call in enumerate(block.get("calls") or []):
                    args = call.get("arguments", call.get("args", "{}"))
                    if not isinstance(args, str):
                        args = json.dumps(args, ensure_ascii=False, sort_keys=True)
                    tool_calls.append({
                        "id": call.get("id") or ("call_%d" % i),
                        "name": call.get("name", ""),
                        "arguments": args,
                    })
    elif isinstance(raw_content, str) and raw_content.strip():
        text_parts.append(raw_content.strip())
    tc_json = message.get("tool_calls_json")
    if tc_json and not tool_calls:
        try:
            parsed_calls = json.loads(tc_json) if isinstance(tc_json, str) else tc_json
        except ValueError:
            parsed_calls = []
        for i, call in enumerate(parsed_calls or []):
            fn = call.get("function") if isinstance(call.get("function"), dict) else {}
            name = call.get("name") or fn.get("name") or ""
            args = call.get("args") if "args" in call else call.get("arguments", fn.get("arguments", "{}"))
            if not isinstance(args, str):
                args = json.dumps(args, ensure_ascii=False, sort_keys=True)
            tool_calls.append({
                "id": call.get("id") or ("call_%d" % i),
                "name": name,
                "arguments": args,
            })
    text = "\n\n".join(p for p in text_parts if p) or (message.get("text") or "").strip()
    if not tool_calls and "```json" not in text and "```citations" not in text:
        fence = reconstruct_closeout_fence(message)
        if fence:
            text = ("%s\n\n%s" % (text, fence)).strip() if text else fence
    return {"role": "assistant", "text": text, "tool_calls": tool_calls}


def parse_rejection(result_block):
    """Extract structured rejection Code and paintedwolf.* details from a tool result entry."""
    content = str(result_block.get("content") or "")
    details = result_block.get("details") if isinstance(result_block.get("details"), dict) else {}
    is_error = bool(result_block.get("is_error")) or bool(STATUS_ERROR_RE.search(content))
    code_match = REJECTION_CODE_RE.search(content)
    code = code_match.group(1) if code_match else details.get("code")
    if not code and is_error:
        if "no progress checklist yet" in content:
            code = "PROGRESS_REQUIRED_BEFORE_EDIT"
        elif "Shell file-writing redirection" in content or "stream piping targeting" in content:
            code = "SHELL_FILE_WRITE_FORBIDDEN"
    if not is_error and not code:
        return None
    return {
        "tool_call_id": result_block.get("tool_call_id", ""),
        "tool_name": result_block.get("tool_name", ""),
        "code": code or "tool_error",
        "details": details,
        "content": content[:1000],
    }


def parse_host_user_rejection(message):
    """Extract structured turn-level host rejections or coordinator nudges delivered as user-role messages."""
    if not message or message.get("role") != "user":
        return None
    raw = message.get("content")
    text = (message.get("text") or (raw if isinstance(raw, str) else "") or "").strip()
    if not text:
        return None
    code_match = REJECTION_CODE_RE.search(text)
    if text.startswith(">>> Tool feedback"):
        code = code_match.group(1) if code_match else "COORDINATOR_FEEDBACK"
        return {"code": code, "details": {}, "content": text[:1000]}
    if text.startswith("[host:coordinator-citation-grounding]"):
        code = code_match.group(1) if code_match else "COORDINATOR_CITATION_GROUNDING"
        return {"code": code, "details": {}, "content": text[:1000]}
    if text.startswith("Rejected:"):
        if code_match:
            code = code_match.group(1)
        elif "progress steps" in text:
            code = "TURN_CLOSEOUT_PENDING_PROGRESS"
        elif "could not parse the arguments" in text:
            code = "TOOL_ARGS_MALFORMED"
        elif "Markdown answer" in text:
            code = "TURN_CLOSEOUT_FORMAT"
        elif "complete_leg" in text:
            code = "WORKER_COMPLETION_REPORT_MISSING"
        elif "submit_verdict" in text:
            code = "REVIEW_LOOP_VERDICT_INVALID"
        elif "fanout_plan" in text or "workflow_advance" in text:
            code = "WORKFLOW_ADVANCE_REQUIRED"
        else:
            code = "TURN_CLOSEOUT_REJECTED"
        return {"code": code, "details": {}, "content": text[:1000]}
    if text.startswith("[host:coordinator-survey-streak]"):
        return {"code": "COORDINATOR_SURVEY_STREAK", "details": {}, "content": text[:1000]}
    if text.startswith("[host:coordinator-progress-missing]"):
        return {"code": "COORDINATOR_PROGRESS_MISSING", "details": {}, "content": text[:1000]}
    return None


def _tool_results_in(message):
    out = []
    raw_content = message.get("content")
    if isinstance(raw_content, list):
        for block in raw_content:
            if block.get("type") == "tool_results":
                out.extend(block.get("results") or [])
        return out
    tr_json = message.get("tool_result_json")
    if tr_json:
        try:
            parsed = json.loads(tr_json) if isinstance(tr_json, str) else tr_json
        except ValueError:
            parsed = None
        if isinstance(parsed, dict):
            if "content" not in parsed and isinstance(raw_content, str):
                parsed = dict(parsed, content=raw_content)
            return [parsed]
        if isinstance(parsed, list):
            return [r for r in parsed if isinstance(r, dict)]
    if isinstance(raw_content, str) and raw_content:
        return [{"content": raw_content}]
    return out


def _openai_messages(prefix):
    """Project a transcript message prefix into OpenAI-compatible chat messages with thinking stripped."""
    out = []
    for msg in prefix:
        role = msg.get("role")
        if role == "user":
            raw_c = msg.get("content")
            text = msg.get("text") or (raw_c if isinstance(raw_c, str) else "")
            out.append({"role": "user", "content": text})
        elif role == "assistant":
            clean = strip_thinking(msg)
            entry = {"role": "assistant", "content": clean["text"] or ""}
            if clean["tool_calls"]:
                entry["tool_calls"] = [
                    {"id": c["id"], "type": "function", "function": {"name": c["name"], "arguments": c["arguments"]}}
                    for c in clean["tool_calls"]
                ]
            out.append(entry)
        elif role == "tool":
            for res in _tool_results_in(msg):
                out.append({
                    "role": "tool",
                    "tool_call_id": res.get("tool_call_id") or "call_0",
                    "name": res.get("tool_name") or res.get("name") or "",
                    "content": str(res.get("content") or ""),
                })
    return out


def load_judged_index(rows_paths):
    """Index judged pw-decide-row/1 records by session_id and opening_message_id."""
    by_session = {}
    by_opening = {}
    paths = [rows_paths] if isinstance(rows_paths, (str, Path)) else list(rows_paths or [])
    for path in paths:
        if not path or not Path(path).exists():
            continue
        with _open_jsonl(path) as fh:
            for line in fh:
                if not line.strip():
                    continue
                row = json.loads(line)
                sid = row.get("session")
                if sid:
                    by_session.setdefault(sid, []).append(row)
                    if row.get("opening_message_id"):
                        by_opening[(sid, row["opening_message_id"])] = row
    return by_session, by_opening


WORKFLOW_CONTROL_TOOLS = {
    "fanout_plan",
    "submit_verdict",
    "pack_board",
    "workflow_advance",
    "complete_leg",
    "request_decision",
    "extend_worker_budget",
    "decline_worker_budget",
}
SCAN_ACCOUNTING_TOOLS = {"scan_summary", "scan_query", "scan_run", "scan_compare"}
VALID_REPORT_RATINGS = {"clear", "hardening", "elevated", "critical"}
VALID_FINDING_DISPOSITIONS = {"act", "accept", "held", "unresolved"}


def synthesized_surface(row, min_score=3, preload_from_judge=True, observed_tools=None):
    """Compute the offered and preloaded tool/skill surface for a turn."""
    obs = set(observed_tools or ())
    if not row:
        base = sorted({"read", "edit", "write", "command", "request_tools", "skills_read"} | obs)
        return {"floor": base, "active_tools": base, "loadable": [],
                "preloaded_guides": [], "required_tools": [], "required_skills": []}
    offered = row.get("offered") or {}
    loadable = list(offered.get("loadable") or [])
    floor = sorted(set(offered.get("floor") or []) | (obs & (WORKFLOW_CONTROL_TOOLS | {"update_progress"})))
    labels = row.get("labels") or {}
    tool_scores = labels.get("tool_scores") or {}
    skill_scores = labels.get("skill_scores") or {}
    req_scores = {}
    for req in labels.get("requests") or []:
        for t, s in (req.get("scores") or {}).items():
            req_scores[t] = max(req_scores.get(t, 0), int(s))
    required_tools = sorted({t for t, s in {**tool_scores, **req_scores}.items() if int(s) >= min_score})
    required_skills = sorted({k for k, s in skill_scores.items() if int(s) >= min_score})
    engine = row.get("engine") or {}
    preloaded = list(engine.get("preloaded") or [])
    if preload_from_judge and engine.get("state") != "decided":
        preloaded = sorted(set(preloaded) | set(required_tools))
    active = sorted(set(floor) | set(preloaded))
    return {
        "floor": floor,
        "active_tools": active,
        "loadable": loadable,
        "preloaded_guides": sorted(set(required_skills) | {g for g, v in (labels.get("guides") or {}).items() if v == "kept"}),
        "required_tools": required_tools,
        "required_skills": required_skills,
    }


def _adapt_turn_for_preloaded(turn, active_tools, preload_from_judge=True):
    """Drop redundant request_tools calls when all requested tools are already preloaded."""
    if not preload_from_judge:
        return turn
    active = set(active_tools)
    kept_calls = []
    for call in turn["tool_calls"]:
        if call["name"] == "request_tools":
            try:
                args = json.loads(call["arguments"]) if isinstance(call["arguments"], str) else call["arguments"]
            except ValueError:
                args = {}
            names = set(args.get("names") or args.get("tools") or [])
            if names and names <= active:
                continue
        kept_calls.append(call)
    return dict(turn, tool_calls=kept_calls)


def _session_meta(session, row=None):
    row_meta = (row or {}).get("meta") or {}
    sid = session.get("session_id") or session.get("id") or ""
    workspace = session.get("workspace") or row_meta.get("workspace") or row_meta.get("repo") or (row or {}).get("project") or "unknown"
    task_id = session.get("task_id") or row_meta.get("task_id") or session.get("root_session") or sid
    prompt_group = session.get("prompt_group") or row_meta.get("prompt_group") or ("%s:%s" % (workspace, task_id))
    return {
        "workspace": workspace,
        "prompt_group": prompt_group,
        "task_id": task_id,
        "archetype": session.get("archetype") or row_meta.get("archetype") or "",
        "model": session.get("model") or (row or {}).get("model") or "",
    }


def extract(transcripts_path, rows_paths=(), out_sft=None, out_dpo=None, min_score=3, preload_from_judge=True, teacher_models=None):
    """Extract Stage 1 SFT rows and Stage 2 DPO pairs from transcripts and judged rows."""
    by_session, by_opening = load_judged_index(rows_paths)
    teachers = set(teacher_models) if teacher_models else None
    sft_rows = []
    dpo_rows = []
    stats = {"sessions": 0, "sft_rows": 0, "dpo_rows": 0, "by_source": {}, "rejection_codes": {}}

    for session in iter_transcripts(transcripts_path):
        stats["sessions"] += 1
        sid = session.get("session_id") or session.get("id")
        sess_rows = by_session.get(sid) or []
        primary_row = sess_rows[0] if sess_rows else None
        meta = _session_meta(session, primary_row)
        messages = session.get("messages") or []
        called_in_session = {
            call.get("name")
            for msg in messages if msg.get("role") == "assistant"
            for call in strip_thinking(msg)["tool_calls"] if call.get("name")
        }
        draft_by_slot = {}
        for dv in session.get("draft_versions") or []:
            slot = dv.get("slot_id")
            if slot:
                draft_by_slot.setdefault(slot, []).append(dv)
        current_opening = None
        for idx, msg in enumerate(messages):
            if msg.get("role") == "user":
                if parse_host_user_rejection(msg) is None and msg.get("origin") != "host":
                    current_opening = msg.get("id")
                continue
            if msg.get("role") != "assistant":
                continue
            row = by_opening.get((sid, current_opening)) or primary_row
            surface = synthesized_surface(row, min_score=min_score, preload_from_judge=preload_from_judge, observed_tools=called_in_session)
            turn = strip_thinking(msg)
            next_msg = messages[idx + 1] if idx + 1 < len(messages) else None
            results = _tool_results_in(next_msg) if (next_msg and next_msg.get("role") == "tool") else []
            rejections = [r for r in (parse_rejection(res) for res in results) if r is not None]
            host_user_rej = parse_host_user_rejection(next_msg)
            if not rejections and host_user_rej is not None:
                rejections = [host_user_rej]
            turn_text = turn.get("text") or ""
            host_assembled = is_host_assembled_closeout(msg)
            defect_code = report_document_defect_code(msg)
            has_grounded_citations = (
                "```citations" in turn_text
                or ("```json" in turn_text and not host_assembled and not defect_code)
            )
            prefix = _openai_messages(messages[:idx])

            # Extract retracted closeout drafts from draft_versions for this message slot.
            slot_drafts = draft_by_slot.get(msg.get("id")) or []
            if slot_drafts:
                chosen_closeout = turn if (has_grounded_citations or turn["tool_calls"]) and not host_assembled and not defect_code else None
                if chosen_closeout is None and idx + 1 < len(messages):
                    for later in messages[idx + 1:idx + 4]:
                        if later.get("role") == "assistant":
                            cand = strip_thinking(later)
                            if any(c.get("name") == "update_progress" for c in cand["tool_calls"]):
                                chosen_closeout = cand
                                break
                for dv_idx, dv in enumerate(slot_drafts):
                    dv_code = dv.get("outcome_code") or "TURN_CLOSEOUT_REJECTED"
                    stats["rejection_codes"][dv_code] = stats["rejection_codes"].get(dv_code, 0) + 1
                    rej_turn = {
                        "role": "assistant",
                        "text": render_draft_version_body(dv.get("body", "")),
                        "tool_calls": [],
                    }
                    dpo_rows.append({
                        "schema": DPO_SCHEMA,
                        "id": "%s:%d:draft:%d" % (sid, idx, dv_idx),
                        "session": sid,
                        "root_session": session.get("root_session") or sid,
                        "turn_index": idx,
                        "source": "host_rejection",
                        "rejection_code": dv_code,
                        "rejection_details": {"slot_id": msg.get("id", ""), "version_index": dv.get("version_index", dv_idx)},
                        "surface": surface,
                        "prompt": prefix,
                        "chosen": chosen_closeout,
                        "rejected": rej_turn,
                        "meta": meta,
                    })
                    stats["by_source"]["host_rejection"] = stats["by_source"].get("host_rejection", 0) + 1
            elif (host_assembled or defect_code) and not rejections:
                grounding = _parse_json_field(msg.get("grounding_json") or msg.get("grounding")) or {}
                comp_meta = _parse_json_field(msg.get("completion_report_json") or msg.get("completion_report")) or {}
                hint_code = defect_code or grounding.get("hint_code") or "INVEST_CITATIONS_REQUIRED"
                stats["rejection_codes"][hint_code] = stats["rejection_codes"].get(hint_code, 0) + 1
                rej_details = {"defects": comp_meta.get("defects")} if defect_code else {"host_assembled": True}
                dpo_rows.append({
                    "schema": DPO_SCHEMA,
                    "id": "%s:%d:untraced" % (sid, idx),
                    "session": sid,
                    "root_session": session.get("root_session") or sid,
                    "turn_index": idx,
                    "source": "host_rejection",
                    "rejection_code": hint_code,
                    "rejection_details": rej_details,
                    "surface": surface,
                    "prompt": prefix,
                    "chosen": None,
                    "rejected": turn,
                    "meta": meta,
                })
                stats["by_source"]["host_rejection"] = stats["by_source"].get("host_rejection", 0) + 1
                continue

            if not turn["tool_calls"] and not rejections and not has_grounded_citations:
                continue

            if rejections:
                rej = rejections[0]
                code = rej["code"]
                stats["rejection_codes"][code] = stats["rejection_codes"].get(code, 0) + 1
                repaired_turn = None
                if idx + 2 < len(messages) and messages[idx + 2].get("role") == "assistant":
                    candidate_msg = messages[idx + 2]
                    candidate = strip_thinking(candidate_msg)
                    later_msg = messages[idx + 3] if idx + 3 < len(messages) else None
                    later_results = _tool_results_in(later_msg) if (later_msg and later_msg.get("role") == "tool") else []
                    later_rejs = [r for r in (parse_rejection(res) for res in later_results) if r is not None]
                    if parse_host_user_rejection(later_msg) is not None:
                        later_rejs.append(parse_host_user_rejection(later_msg))
                    cand_text = candidate.get("text") or ""
                    cand_grounded = (
                        "```citations" in cand_text
                        or ("```json" in cand_text and not is_host_assembled_closeout(candidate_msg) and not report_document_defect_code(candidate_msg))
                    )
                    if (candidate["tool_calls"] or cand_grounded) and not later_rejs:
                        repaired_turn = candidate
                # 1. Pre-rejection prevention pair at s_t.
                dpo_rows.append({
                    "schema": DPO_SCHEMA,
                    "id": "%s:%d:prevent" % (sid, idx),
                    "session": sid,
                    "root_session": session.get("root_session") or sid,
                    "turn_index": idx,
                    "source": "host_rejection",
                    "rejection_code": code,
                    "rejection_details": rej["details"],
                    "surface": surface,
                    "prompt": prefix,
                    "chosen": repaired_turn,
                    "rejected": turn,
                    "meta": meta,
                })
                stats["by_source"]["host_rejection"] = stats["by_source"].get("host_rejection", 0) + 1
                # 2. Post-rejection recovery pair at s_{t+1} when the session recovered on the next turn.
                if repaired_turn is not None and next_msg is not None:
                    recovery_prefix = _openai_messages(messages[:idx + 2])
                    dpo_rows.append({
                        "schema": DPO_SCHEMA,
                        "id": "%s:%d:recover" % (sid, idx + 2),
                        "session": sid,
                        "root_session": session.get("root_session") or sid,
                        "turn_index": idx + 2,
                        "source": "post_rejection_recovery",
                        "rejection_code": code,
                        "rejection_details": rej["details"],
                        "surface": surface,
                        "prompt": recovery_prefix,
                        "chosen": repaired_turn,
                        "rejected": turn,
                        "meta": meta,
                    })
                    stats["by_source"]["post_rejection_recovery"] = stats["by_source"].get("post_rejection_recovery", 0) + 1
                continue

            # Check for dropped request_tools follow-through on this turn.
            call_names = [c["name"] for c in turn["tool_calls"]]
            if "request_tools" in call_names and row:
                reqs = (row.get("labels") or {}).get("requests") or []
                dropped = [r for r in reqs if not (r.get("after") or [])]
                if dropped:
                    dpo_rows.append({
                        "schema": DPO_SCHEMA,
                        "id": "%s:%d:followup" % (sid, idx),
                        "session": sid,
                        "root_session": session.get("root_session") or sid,
                        "turn_index": idx,
                        "source": "dropped_request_followup",
                        "rejection_code": "dropped_request_followup",
                        "rejection_details": {"need": dropped[0].get("need", "")},
                        "surface": surface,
                        "prompt": prefix,
                        "chosen": None,
                        "rejected": turn,
                        "meta": meta,
                    })
                    stats["by_source"]["dropped_request_followup"] = stats["by_source"].get("dropped_request_followup", 0) + 1
                    continue

            # Clean turn candidate for Stage 1 SFT warmup.
            if host_assembled or defect_code:
                continue
            task_status = ((row or {}).get("meta") or {}).get("task_status")
            if session.get("status") not in ("settled", "idle", "") or task_status not in (None, "", "settled"):
                continue
            if teachers and meta["model"] and meta["model"] not in teachers:
                continue
            if surface["required_skills"] and "skills_read" not in called_in_session:
                continue
            if surface["required_tools"] and not (set(surface["required_tools"]) & called_in_session):
                continue
            adapted = _adapt_turn_for_preloaded(turn, surface["active_tools"], preload_from_judge=preload_from_judge)
            prior_tool_msgs = sum(1 for m in prefix if m.get("role") == "tool")
            if not adapted["tool_calls"] and not (has_grounded_citations and prior_tool_msgs >= 1):
                continue
            sft_rows.append({
                "schema": SFT_SCHEMA,
                "id": "%s:%d:sft" % (sid, idx),
                "session": sid,
                "root_session": session.get("root_session") or sid,
                "turn_index": idx,
                "surface": surface,
                "prompt": prefix,
                "target": adapted,
                "meta": meta,
            })

    stats["sft_rows"] = len(sft_rows)
    stats["dpo_rows"] = len(dpo_rows)
    if out_sft:
        Path(out_sft).parent.mkdir(parents=True, exist_ok=True)
        with open(out_sft, "w", encoding="utf-8") as fh:
            for r in sft_rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    if out_dpo:
        Path(out_dpo).parent.mkdir(parents=True, exist_ok=True)
        with open(out_dpo, "w", encoding="utf-8") as fh:
            for r in dpo_rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    return stats


def _prior_tool_signatures(prompt_messages):
    seen = set()
    for msg in prompt_messages:
        if msg.get("role") == "assistant":
            for tc in msg.get("tool_calls") or []:
                fn = tc.get("function") or {}
                seen.add((fn.get("name", ""), fn.get("arguments", "")))
    return seen


def _is_scoped_read(call):
    if call.get("name") != "read":
        return False
    try:
        args = json.loads(call.get("arguments") or "{}")
    except ValueError:
        return False
    return bool(isinstance(args, dict) and (set(args) & SCOPED_READ_KEYS))


TRAILING_FENCE_RE = re.compile(r"```(?:json|citations)\s*\n(\{[\s\S]*?\})\s*\n```\s*$", re.IGNORECASE)
UUID_RE = re.compile(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b", re.IGNORECASE)


def _extract_trailing_fence_json(text):
    if not text:
        return None
    m = TRAILING_FENCE_RE.search(text.strip())
    if not m:
        return None
    try:
        parsed = json.loads(m.group(1))
        return parsed if isinstance(parsed, dict) else None
    except ValueError:
        return None


def _is_malformed_closeout_fence(teacher_fence, student_text, student_fence):
    if student_fence is None:
        return True
    s_ev = student_fence.get("cited_evidence")
    if isinstance(s_ev, list):
        for item in s_ev:
            if not isinstance(item, dict):
                return True
            if "handle" in item:
                return True
            ev_val = str(item.get("evidence") or "")
            if ev_val and "#" not in ev_val:
                return True
            if UUID_RE.search(ev_val) or UUID_RE.search(str(item.get("path") or "")):
                return True
    if teacher_fence.get("cited_evidence") and not s_ev:
        return True
    if teacher_fence.get("verification") and not isinstance(student_fence.get("verification"), dict):
        return True
    if teacher_fence.get("artifact_ids") and not student_fence.get("artifact_ids"):
        return True
    if teacher_fence.get("headline") and not str(student_fence.get("headline") or "").strip():
        return True
    if teacher_fence.get("rating"):
        s_rating = student_fence.get("rating")
        if not isinstance(s_rating, dict) or s_rating.get("level") not in VALID_REPORT_RATINGS or not str(s_rating.get("why") or "").strip():
            return True
    if teacher_fence.get("findings"):
        s_findings = student_fence.get("findings")
        if not isinstance(s_findings, list) or not s_findings:
            return True
        t_claim_ids = {f.get("id") for f in teacher_fence["findings"] if isinstance(f, dict) and f.get("id")}
        s_claim_ids = set()
        for f in s_findings:
            if not isinstance(f, dict):
                return True
            if f.get("id"):
                s_claim_ids.add(f["id"])
            if f.get("disposition") and f["disposition"] not in VALID_FINDING_DISPOSITIONS:
                return True
            if t_claim_ids and not f.get("disposition"):
                return True
            for w in f.get("where") or []:
                if not isinstance(w, dict) or "handle" in w:
                    return True
                ev_w = str(w.get("evidence") or "")
                if ev_w and "#" not in ev_w:
                    return True
        if t_claim_ids and not (t_claim_ids <= s_claim_ids):
            return True
    if teacher_fence.get("set_asides") and not isinstance(student_fence.get("set_asides"), list):
        return True
    if teacher_fence.get("ask") and not isinstance(student_fence.get("ask"), dict):
        return True
    body_without_fence = TRAILING_FENCE_RE.sub("", student_text or "")
    if teacher_fence.get("artifact_ids") and UUID_RE.search(body_without_fence):
        return True
    return False


def _has_malformed_workflow_control_args(t_calls, s_calls):
    """Return True when the student calls a workflow/worker control tool with missing required fields or ungrounded findings."""
    t_by_name = {c.get("name"): c for c in t_calls if c.get("name") in WORKFLOW_CONTROL_TOOLS}
    for sc in s_calls:
        name = sc.get("name")
        if name not in t_by_name:
            continue
        try:
            s_args = json.loads(sc.get("arguments") or "{}")
            t_args = json.loads(t_by_name[name].get("arguments") or "{}")
        except ValueError:
            return True
        if not isinstance(s_args, dict) or not isinstance(t_args, dict):
            return True
        if name == "fanout_plan":
            if not isinstance(s_args.get("legs"), list) or not s_args["legs"]:
                return True
            if t_args.get("threat_model") and not isinstance(s_args.get("threat_model"), dict):
                return True
            t_roles = {leg.get("role") for leg in (t_args.get("legs") or []) if isinstance(leg, dict)}
            s_roles = {leg.get("role") for leg in s_args["legs"] if isinstance(leg, dict)}
            if t_roles and (s_roles - t_roles) & {"skeptic", "web-researcher"}:
                return True
        elif name == "complete_leg":
            if not s_args.get("leg_status"):
                return True
            for finding in s_args.get("findings") or []:
                if not isinstance(finding, dict):
                    return True
                if not finding.get("path") or not finding.get("line") or not finding.get("excerpt"):
                    return True
        elif name == "submit_verdict":
            if not s_args.get("verdict") or (t_args.get("verdict") and s_args.get("verdict") != t_args.get("verdict")):
                return True
            if t_args.get("coverage") and not isinstance(s_args.get("coverage"), list):
                return True
    return False


def classify_divergence(sft_row, teacher_turn, student_turn):
    """Return a structured divergence code when the student's non-thinking turn is inferior to the teacher's."""
    surface = sft_row.get("surface") or {}
    allowed_tools = set(surface.get("active_tools") or []) | set(surface.get("loadable") or [])
    t_calls = teacher_turn.get("tool_calls") or []
    s_calls = student_turn.get("tool_calls") or []
    t_text = teacher_turn.get("text") or ""
    s_text = student_turn.get("text") or ""
    if not t_calls:
        t_fence = _extract_trailing_fence_json(t_text)
        if t_fence is not None and not s_calls:
            s_fence = _extract_trailing_fence_json(s_text)
            if _is_malformed_closeout_fence(t_fence, s_text, s_fence):
                return "closeout_fence_missing_or_malformed"
        return None
    if t_calls and not s_calls:
        return "premature_stop"
    for call in s_calls:
        name = call.get("name", "")
        if allowed_tools and name not in allowed_tools:
            return "invalid_tool_call"
        try:
            parsed = json.loads(call.get("arguments") or "{}")
            if not isinstance(parsed, dict):
                return "invalid_tool_call"
        except ValueError:
            return "invalid_tool_call"
    prior = _prior_tool_signatures(sft_row.get("prompt") or [])
    if s_calls and all((c.get("name", ""), c.get("arguments", "")) in prior for c in s_calls):
        return "repeated_call"
    t_names = {c.get("name") for c in t_calls}
    s_names = {c.get("name") for c in s_calls}
    if "skills_read" in t_names and "skills_read" not in s_names:
        return "missed_skill_read"
    if (t_names & WORKFLOW_CONTROL_TOOLS) and not ((t_names & WORKFLOW_CONTROL_TOOLS) <= s_names):
        return "missed_workflow_verdict_or_plan"
    if (t_names & WORKFLOW_CONTROL_TOOLS) and _has_malformed_workflow_control_args(t_calls, s_calls):
        return "malformed_workflow_control_args"
    if (t_names & SCAN_ACCOUNTING_TOOLS) and not (s_names & (t_names & SCAN_ACCOUNTING_TOOLS)):
        return "missed_scan_accounting"
    if "update_progress" in t_names and "update_progress" not in s_names:
        if s_names & {"write", "edit", "replace_lines", "command"}:
            return "missing_progress_before_edit"
        return "unclosed_progress_before_closeout"
    if (t_names & {"capture_page", "capture_tui"}) and not (s_names & {"capture_page", "capture_tui"}):
        return "missed_visual_capture"
    req_tools = set(surface.get("required_tools") or [])
    teacher_req = t_names & req_tools
    if teacher_req and not (s_names & teacher_req):
        if "command" in s_names and "command" not in t_names:
            return "shell_fallback"
        return "missed_loadable_tool"
    if any(_is_scoped_read(c) for c in t_calls) and any(c.get("name") == "read" for c in s_calls) and not any(_is_scoped_read(c) for c in s_calls):
        return "unscoped_read"
    if "fetch_url" in t_names and "fetch_url" not in s_names and s_names <= {"web_search"}:
        return "snippet_only_web_research"
    shallow_only = {"list_dir", "summarize"}
    deep_tools = {"read", "grep", "verify", "command", "task", "fanout_plan", "delegate_dispatch", "web_search", "fetch_url", "http_request"}
    if (t_names & deep_tools) and s_names and s_names <= shallow_only and not (t_names <= shallow_only):
        return "shallow_investigation"
    return None


def _tool_schemas_for(surface):
    names = sorted(set(surface.get("active_tools") or []) | set(surface.get("loadable") or []))
    return [{"type": "function", "function": {"name": name, "parameters": {"type": "object"}}} for name in names]


def shadow_signature(sft_path, student_id, seed):
    return {"sft_sha256": provenance.file_sha256(sft_path), "student": student_id, "seed": seed}


def shadow(factory, sft_path, out_path, student_id="gemma-4-e4b-it", workers=16, seed=7):
    """Query the student model (thinking=False) at teacher SFT states and write shadow_divergence DPO pairs."""
    sft_path, out_path = Path(sft_path), Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    partial = Path(str(out_path) + ".partial.jsonl")
    manifest = Path(str(out_path) + ".manifest.json")
    sig = shadow_signature(sft_path, student_id, seed)
    if manifest.exists():
        try:
            if json.loads(manifest.read_text(encoding="utf-8")).get("signature") != sig:
                partial.unlink(missing_ok=True)
        except ValueError:
            partial.unlink(missing_ok=True)
    manifest.write_text(json.dumps({"signature": sig}, indent=2) + "\n", encoding="utf-8")

    done_ids = set()
    kept = []
    if partial.exists():
        for line in partial.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                done_ids.add(rec["sft_id"])
                if rec.get("pair"):
                    kept.append(rec["pair"])

    sft_rows = [json.loads(line) for line in sft_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    todo = [r for r in sft_rows if r["id"] not in done_ids]
    student = factory.model(student_id)
    chats = factory.chats(student)
    lock = threading.Lock()
    counts = {}

    def run_one(index_row):
        idx, row = index_row
        chat = chats[idx % len(chats)]
        row_seed = int(hashlib.sha256(("%s:%s" % (seed, row["id"])).encode()).hexdigest()[:8], 16)
        tools = _tool_schemas_for(row.get("surface") or {})
        student_turn = chat.turn(row["prompt"], tools=tools, temperature=0.0, seed=row_seed, thinking=False)
        student_turn["role"] = "assistant"
        code = classify_divergence(row, row["target"], student_turn)
        pair = None
        if code:
            pair = {
                "schema": DPO_SCHEMA,
                "id": "%s:shadow" % row["id"],
                "session": row["session"],
                "root_session": row["root_session"],
                "turn_index": row["turn_index"],
                "source": "shadow_divergence",
                "rejection_code": code,
                "rejection_details": {"student": student_id, "teacher": row.get("meta", {}).get("model", "")},
                "surface": row["surface"],
                "prompt": row["prompt"],
                "chosen": row["target"],
                "rejected": student_turn,
                "meta": row["meta"],
            }
        return {"sft_id": row["id"], "pair": pair, "code": code}

    with open(partial, "a", encoding="utf-8") as fh, concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        for res in pool.map(run_one, enumerate(todo)):
            with lock:
                fh.write(json.dumps(res, ensure_ascii=False) + "\n")
                fh.flush()
                if res["pair"]:
                    kept.append(res["pair"])

    for pair in kept:
        code = pair["rejection_code"]
        counts[code] = counts.get(code, 0) + 1
    with open(out_path, "w", encoding="utf-8") as fh:
        for pair in kept:
            fh.write(json.dumps(pair, ensure_ascii=False) + "\n")
    return {"evaluated": len(sft_rows), "pairs": len(kept), "by_code": counts}


def repair(factory, dpo_path, out_path, teacher_id="gemma-4-31b-it", workers=16, seed=7):
    """Fill missing `chosen` turns in extracted DPO rows by querying the teacher model with thinking=False."""
    dpo_path, out_path = Path(dpo_path), Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(line) for line in dpo_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    need_repair = [(i, r) for i, r in enumerate(rows) if not r.get("chosen")]
    if need_repair:
        teacher = factory.model(teacher_id)
        chats = factory.chats(teacher)

        def repair_one(item):
            idx, row = item
            chat = chats[idx % len(chats)]
            row_seed = int(hashlib.sha256(("%s:%s" % (seed, row["id"])).encode()).hexdigest()[:8], 16)
            tools = _tool_schemas_for(row.get("surface") or {})
            turn = chat.turn(row["prompt"], tools=tools, temperature=0.0, seed=row_seed, thinking=False)
            turn["role"] = "assistant"
            rej_calls = [(c.get("name"), c.get("arguments")) for c in (row.get("rejected") or {}).get("tool_calls") or []]
            new_calls = [(c.get("name"), c.get("arguments")) for c in turn.get("tool_calls") or []]
            if not new_calls or new_calls == rej_calls:
                return idx, None
            return idx, turn

        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            for idx, chosen in pool.map(repair_one, need_repair):
                rows[idx]["chosen"] = chosen

    completed = [r for r in rows if r.get("chosen") and (r["chosen"].get("tool_calls") or r["chosen"].get("text"))]
    with open(out_path, "w", encoding="utf-8") as fh:
        for r in completed:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    return {"input_rows": len(rows), "repaired": len(need_repair), "completed_pairs": len(completed)}


def _prefix_path_with_root(path_val, root_label):
    if not isinstance(path_val, str) or not path_val.strip():
        return path_val
    p = path_val.strip()
    if p.startswith("@"):
        return p
    if p == ".":
        return "@%s/" % root_label
    return "@%s/%s" % (root_label, p.lstrip("./"))


def synthesize_multi_root_pairs(sft_rows, companion_label="spec-reference", max_pairs=200):
    """Synthesize multi-root @<folder>/ SFT and DPO examples from single-root trajectories.

    When a project has two or more attached roots (e.g. an implementation repo and a
    specification/companion repo), Painted Wolf Code requires @<folder>/ path prefixes
    in file tools and closeout citations, and expects cross-root audits to inspect both roots.
    """
    synth_sft = []
    synth_dpo = []
    for row in sft_rows:
        if len(synth_dpo) >= max_pairs:
            break
        meta = dict(row.get("meta") or {})
        ws = (meta.get("workspace") or "primary-repo").strip("@/")
        target = row.get("target") or {}
        calls = target.get("tool_calls") or []
        file_tools = {"read", "list_dir", "grep", "summarize", "find_by_name", "edit", "write"}
        if not any(c.get("name") in file_tools for c in calls):
            continue
        prefixed_calls = []
        unprefixed_calls = []
        changed = False
        for c in calls:
            unprefixed_calls.append(dict(c))
            if c.get("name") in file_tools:
                try:
                    args = json.loads(c.get("arguments") or "{}")
                except ValueError:
                    args = None
                if isinstance(args, dict) and isinstance(args.get("path"), str) and not args["path"].startswith("@"):
                    new_args = dict(args, path=_prefix_path_with_root(args["path"], ws))
                    prefixed_calls.append(dict(c, arguments=json.dumps(new_args, ensure_ascii=False, sort_keys=True)))
                    changed = True
                    continue
            prefixed_calls.append(dict(c))
        if not changed:
            continue
        prompt = []
        injected = False
        for m in row.get("prompt") or []:
            mc = dict(m)
            if mc.get("role") == "user" and not injected:
                mc["content"] = (
                    "[Workspace roots attached: @%s/ (primary), @%s/ (companion). "
                    "Use @<folder>/ path prefixes for all file tools and citations.]\n\n%s"
                    % (ws, companion_label, mc.get("content") or "")
                )
                injected = True
            prompt.append(mc)
        if not injected:
            continue
        chosen_turn = {"role": "assistant", "text": target.get("text") or "", "tool_calls": prefixed_calls}
        rejected_turn = {"role": "assistant", "text": target.get("text") or "", "tool_calls": unprefixed_calls}
        synth_sft.append({
            "schema": SFT_SCHEMA,
            "id": "%s:multiroot:sft" % row["id"],
            "session": row["session"],
            "root_session": row.get("root_session") or row["session"],
            "turn_index": row["turn_index"],
            "surface": row["surface"],
            "prompt": prompt,
            "target": chosen_turn,
            "meta": dict(meta, multi_root=True),
        })
        synth_dpo.append({
            "schema": DPO_SCHEMA,
            "id": "%s:multiroot:dpo" % row["id"],
            "session": row["session"],
            "root_session": row.get("root_session") or row["session"],
            "turn_index": row["turn_index"],
            "source": "shadow_divergence",
            "rejection_code": "multi_root_prefix_missing",
            "rejection_details": {"primary_root": ws, "companion_root": companion_label},
            "surface": row["surface"],
            "prompt": prompt,
            "chosen": chosen_turn,
            "rejected": rejected_turn,
            "meta": dict(meta, multi_root=True),
        })
    return synth_sft, synth_dpo
