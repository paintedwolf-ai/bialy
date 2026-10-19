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

REJECTION_CODE_RE = re.compile(r"(?:^|\n)(?:⟦D⟧)?Code:\s*(\S+)", re.MULTILINE)
STATUS_ERROR_RE = re.compile(r"(?:^|\n)(?:Status:\s*error|(?:⟦D⟧)?Rejected:)", re.MULTILINE)
SCOPED_READ_KEYS = {"symbol", "start_line", "end_line", "offset", "limit"}


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

                for srow in sess_rows:
                    sid = srow["id"]
                    root = root_of(sid)
                    if root not in by_root:
                        continue
                    mentry, spec = by_root[root]
                    meta = spec.get("meta") or mentry.get("meta") or {}
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


def strip_thinking(message):
    """Return an assistant turn with reasoning blocks removed and text + tool_calls normalized."""
    text_parts = []
    tool_calls = []
    for block in message.get("content") or []:
        btype = block.get("type")
        if btype == "reasoning":
            continue
        if btype == "text" and block.get("text"):
            text_parts.append(block["text"].strip())
        elif btype == "tool_calls":
            for i, call in enumerate(block.get("calls") or []):
                args = call.get("arguments", "{}")
                if not isinstance(args, str):
                    args = json.dumps(args, ensure_ascii=False, sort_keys=True)
                tool_calls.append({
                    "id": call.get("id") or ("call_%d" % i),
                    "name": call.get("name", ""),
                    "arguments": args,
                })
    text = "\n\n".join(p for p in text_parts if p) or (message.get("text") or "").strip()
    return {"role": "assistant", "text": text, "tool_calls": tool_calls}


def parse_rejection(result_block):
    """Extract structured rejection Code and paintedwolf.* details from a tool result entry."""
    content = str(result_block.get("content") or "")
    details = result_block.get("details") if isinstance(result_block.get("details"), dict) else {}
    is_error = bool(result_block.get("is_error")) or bool(STATUS_ERROR_RE.search(content))
    code_match = REJECTION_CODE_RE.search(content)
    code = code_match.group(1) if code_match else details.get("code")
    if not is_error and not code:
        return None
    return {
        "tool_call_id": result_block.get("tool_call_id", ""),
        "tool_name": result_block.get("tool_name", ""),
        "code": code or "tool_error",
        "details": details,
        "content": content[:1000],
    }


def _tool_results_in(message):
    out = []
    for block in message.get("content") or []:
        if block.get("type") == "tool_results":
            out.extend(block.get("results") or [])
    return out


def _openai_messages(prefix):
    """Project a transcript message prefix into OpenAI-compatible chat messages with thinking stripped."""
    out = []
    for msg in prefix:
        role = msg.get("role")
        if role == "user":
            out.append({"role": "user", "content": msg.get("text") or ""})
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
                    "name": res.get("tool_name") or "",
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


def synthesized_surface(row, min_score=3, preload_from_judge=True):
    """Compute the offered and preloaded tool/skill surface for a turn."""
    if not row:
        return {"floor": ["read", "edit", "write", "command", "request_tools", "skills_read"],
                "active_tools": ["read", "edit", "write", "command", "request_tools", "skills_read"],
                "loadable": [], "preloaded_guides": [], "required_tools": [], "required_skills": []}
    offered = row.get("offered") or {}
    floor = list(offered.get("floor") or [])
    loadable = list(offered.get("loadable") or [])
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
    workspace = session.get("workspace") or row_meta.get("workspace") or (row or {}).get("project") or "unknown"
    task_id = session.get("task_id") or row_meta.get("task_id") or session.get("root_session") or session.get("session_id")
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
        sid = session.get("session_id")
        sess_rows = by_session.get(sid) or []
        primary_row = sess_rows[0] if sess_rows else None
        meta = _session_meta(session, primary_row)
        messages = session.get("messages") or []
        called_in_session = {
            call.get("name")
            for msg in messages if msg.get("role") == "assistant"
            for block in (msg.get("content") or []) if block.get("type") == "tool_calls"
            for call in (block.get("calls") or []) if call.get("name")
        }
        current_opening = None
        for idx, msg in enumerate(messages):
            if msg.get("role") == "user":
                current_opening = msg.get("id")
                continue
            if msg.get("role") != "assistant":
                continue
            row = by_opening.get((sid, current_opening)) or primary_row
            surface = synthesized_surface(row, min_score=min_score, preload_from_judge=preload_from_judge)
            turn = strip_thinking(msg)
            if not turn["tool_calls"]:
                continue
            next_msg = messages[idx + 1] if idx + 1 < len(messages) else None
            results = _tool_results_in(next_msg) if (next_msg and next_msg.get("role") == "tool") else []
            rejections = [r for r in (parse_rejection(res) for res in results) if r is not None]
            prefix = _openai_messages(messages[:idx])

            if rejections:
                rej = rejections[0]
                code = rej["code"]
                stats["rejection_codes"][code] = stats["rejection_codes"].get(code, 0) + 1
                repaired_turn = None
                if idx + 2 < len(messages) and messages[idx + 2].get("role") == "assistant":
                    candidate = strip_thinking(messages[idx + 2])
                    later_msg = messages[idx + 3] if idx + 3 < len(messages) else None
                    later_results = _tool_results_in(later_msg) if (later_msg and later_msg.get("role") == "tool") else []
                    later_rejs = [r for r in (parse_rejection(res) for res in later_results) if r is not None]
                    if candidate["tool_calls"] and not later_rejs:
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
            if session.get("status") not in ("settled", ""):
                continue
            if teachers and meta["model"] and meta["model"] not in teachers:
                continue
            if surface["required_skills"] and "skills_read" not in called_in_session:
                continue
            if surface["required_tools"] and not (set(surface["required_tools"]) & called_in_session):
                continue
            adapted = _adapt_turn_for_preloaded(turn, surface["active_tools"], preload_from_judge=preload_from_judge)
            if not adapted["tool_calls"]:
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


def classify_divergence(sft_row, teacher_turn, student_turn):
    """Return a structured divergence code when the student's non-thinking turn is inferior to the teacher's."""
    surface = sft_row.get("surface") or {}
    allowed_tools = set(surface.get("active_tools") or []) | set(surface.get("loadable") or [])
    t_calls = teacher_turn.get("tool_calls") or []
    s_calls = student_turn.get("tool_calls") or []
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
    req_tools = set(surface.get("required_tools") or [])
    teacher_req = t_names & req_tools
    if teacher_req and not (s_names & teacher_req):
        if "command" in s_names and "command" not in t_names:
            return "shell_fallback"
        return "missed_loadable_tool"
    if any(_is_scoped_read(c) for c in t_calls) and any(c.get("name") == "read" for c in s_calls) and not any(_is_scoped_read(c) for c in s_calls):
        return "unscoped_read"
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
