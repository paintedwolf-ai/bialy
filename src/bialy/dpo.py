"""SFT and DPO dataset construction for custom small-model training.

Extracts non-thinking tool-calling trajectories (pw-custom-sft/1) and
multi-source preference pairs (pw-custom-dpo/1) from shard SQLite stores or
released transcript archives joined with configured reviewer labels:
  1. Host-rejection prevention pairs at s_t (chosen = teacher/repaired call,
     rejected = call that triggered a host Code: rejection or nudge).
  2. Post-rejection recovery pairs at s_{t+1} (chosen = recovery call after
     reading the structured rejection card, rejected = repeating the bad call).
  3. Dropped request_tools follow-through pairs (after == []).
  4. Shadow-mode divergence pairs at teacher states s_t (student e4b queried
     with thinking=False diverges on loadable tools, skills_read, scoped read,
     or premature stops).

Assistant turns are the exact model text from model_outputs; messages supply role,
kind, grounding, and draft versions.
"""

import concurrent.futures
import copy
import gzip
import hashlib
import json
import re
import sqlite3
import threading
from pathlib import Path

import zstandard

from . import citationmd, provenance
from .dpo_qualification import capture_reasons, qualification, training_eligible

SFT_SCHEMA = "pw-custom-sft/1"
DPO_SCHEMA = "pw-custom-dpo/1"

REJECTION_CODE_RE = re.compile(r"(?:^|\n)Code:\s*([^\s,]+)", re.MULTILINE)
WITHHELD_NOTICE_PREFIX = "MID_TURN_NOTE_"
AGENT_NOTE_KIND = "agent_note"
SCOPED_READ_KEYS = {"ranges", "symbol", "start_line", "end_line", "offset", "limit"}


def is_coordinator_feedback(text):
    return isinstance(text, str) and text.strip().startswith(">>> Tool feedback")


def compact_prompt(messages, max_chars=24000):
    """Enforce a character bound without inventing model observations.

    A training tokenizer must separately measure the complete serialized request
    and response. Oversized requests need host-generated compaction and a newly
    captured/validated response, not a summary substituted into an old prompt.
    """
    total = sum(len(json.dumps(m, ensure_ascii=False)) for m in messages)
    if total > max_chars:
        raise ValueError("request exceeds character budget; recapture host compaction or exclude the row")
    return copy.deepcopy(messages)



def _open_jsonl(path):
    path = Path(path)
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, encoding="utf-8")


def export_transcripts(run_dir, tasks_dir, out_path):
    """Export sessions/transcripts.jsonl(.gz) from one or more pass directories' shard SQLite stores."""
    run_dirs = [Path(p) for p in run_dir] if isinstance(run_dir, (list, tuple)) else [Path(run_dir)]
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    specs = {}
    for path in Path(tasks_dir).glob("*.jsonl"):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                task = json.loads(line)
                specs[task["id"]] = task
    opener = (lambda p: gzip.open(p, "wt", encoding="utf-8")) if out_path.suffix == ".gz" else (lambda p: open(p, "w", encoding="utf-8"))
    sessions_written = 0
    counts = {"model_outputs": 0, "missing_model_outputs": 0}
    with opener(out_path) as out:
        for pass_dir in run_dirs:
            shards_dir = pass_dir / "shards"
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
                conn = sqlite3.connect(db_path.resolve().as_uri() + "?mode=ro", uri=True)
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
                        outputs = _session_model_outputs(conn, shard, sid, counts)
                        messages = []
                        for m in conn.execute(
                            "SELECT id, role, content, kind, origin, visibility, seq, ord, ts, tool_calls_json, tool_result_json, reasoning_json, "
                            "grounding_json, artifact_ids_json, completion_report_json, draft_version_count, draft_status "
                            "FROM messages WHERE session_id = ? ORDER BY seq, ord, rowid",
                            (sid,),
                        ).fetchall():
                            message = dict(m)
                            if message["id"] in outputs:
                                message["model_output"] = outputs[message["id"]]
                            messages.append(message)
                        draft_versions = [dict(d) for d in conn.execute(
                            "SELECT slot_id, version_index, body, outcome_code, created_at "
                            "FROM draft_versions WHERE session_id = ? ORDER BY created_at, version_index",
                            (sid,),
                        ).fetchall()]
                        record = {
                            "pass": pass_dir.name,
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
    if counts["model_outputs"] and counts["missing_model_outputs"] == counts["model_outputs"]:
        raise ValueError("no model output body was retained beside any store; the runner image predates model-content retention")
    return {"sessions": sessions_written, "out": str(out_path), **counts}


def _read_model_output_body(shard, project_id, sha):
    """Read one content-addressed envelope; the digest names its plaintext."""
    path = Path(shard) / "projects" / project_id / "model-content" / sha[:2] / sha[2:]
    with path.open("rb") as stream, zstandard.ZstdDecompressor().stream_reader(stream) as reader:
        plain = reader.read()
    if hashlib.sha256(plain).hexdigest() != sha:
        raise ValueError("model output body %s does not match its digest" % sha)
    return json.loads(plain)


def _session_model_outputs(conn, shard, session_id, counts):
    """Each message's latest settled output. A scripted output has no model author."""
    outputs = {}
    for row in conn.execute(
        "SELECT message_id, project_id, content_blob_sha256, finish_reason, scripted FROM model_outputs "
        "WHERE session_id = ? ORDER BY settled_at, iteration, id",
        (session_id,),
    ).fetchall():
        outputs.pop(row["message_id"], None)
        if row["scripted"]:
            continue
        counts["model_outputs"] += 1
        body = {}
        if row["content_blob_sha256"]:
            try:
                body = _read_model_output_body(shard, row["project_id"], row["content_blob_sha256"])
            except FileNotFoundError:
                counts["missing_model_outputs"] += 1
                continue
        outputs[row["message_id"]] = {
            "content": body.get("content", ""),
            "tool_calls_json": body.get("tool_calls_json"),
            "finish_reason": row["finish_reason"],
        }
    return outputs


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


def has_grounded_citations(message):
    """The host traced citations the model supplied, from row facts alone."""
    grounding = _parse_json_field(message.get("grounding_json") or message.get("grounding"))
    return (
        isinstance(grounding, dict)
        and grounding.get("traced") is True
        and not grounding.get("host_assembled")
        and not grounding.get("evidence_optional")
        and bool(grounding.get("cited_evidence") or grounding.get("cited_urls"))
        and not report_document_defect_code(message)
    )


def closeout_outcome(message):
    """Name a tool-free reply's host outcome; these outcomes stay distinct in every row."""
    grounding = _parse_json_field(message.get("grounding_json") or message.get("grounding"))
    if not isinstance(grounding, dict):
        return None
    if report_document_defect_code(message):
        return "report_defect"
    if is_host_assembled_closeout(message):
        return "host_assembled"
    if grounding.get("evidence_optional"):
        return "evidence_optional"
    return "grounded" if has_grounded_citations(message) else "uncited"


def is_report_phase(message):
    """The host stored this reply as a workflow run's report document."""
    report = _parse_json_field(message.get("completion_report_json") or message.get("completion_report"))
    return isinstance(report, dict) and report.get("scope") == "run"


def model_turn(message):
    """The assistant turn as the model wrote it, reasoning excluded; None if no model output backs the row."""
    output = message.get("model_output")
    if not isinstance(output, dict):
        return None
    tool_calls = []
    raw_calls = output.get("tool_calls_json")
    if isinstance(raw_calls, str):
        try:
            raw_calls = json.loads(raw_calls) if raw_calls else []
        except ValueError:
            raw_calls = []
    for i, call in enumerate(raw_calls or []):
        fn = call.get("function") if isinstance(call.get("function"), dict) else {}
        name = call.get("name") or fn.get("name") or ""
        args = call.get("args") if "args" in call else call.get("arguments", fn.get("arguments", "{}"))
        if not isinstance(args, str):
            args = json.dumps(args, ensure_ascii=False, sort_keys=True)
        tool_calls.append({"id": call.get("id") or ("call_%d" % i), "name": name, "arguments": args})
    return {"role": "assistant", "text": output.get("content") or "", "tool_calls": tool_calls}


def _without_notes(messages):
    """A published note repeats its carrier's text, and no model output backs a note row."""
    return [m for m in messages if m.get("kind") != AGENT_NOTE_KIND]


def parse_rejection(result_block):
    """Extract structured rejection Code and paintedwolf.* details from a tool result entry."""
    content = str(result_block.get("content") or "")
    details = result_block.get("details") if isinstance(result_block.get("details"), dict) else {}
    is_error = result_block.get("is_error") is True
    if not is_error and not details.get("code"):
        return None
    code_match = REJECTION_CODE_RE.search(content)
    code = details.get("code") or (code_match.group(1) if code_match else None)
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
    if message.get("origin") not in (None, "", "host"):
        return None
    raw = message.get("content")
    text = (raw if isinstance(raw, str) else "").strip()
    if not text:
        return None
    code_match = REJECTION_CODE_RE.search(text)
    if text.startswith(">>> Tool feedback"):
        code = code_match.group(1) if code_match else "COORDINATOR_FEEDBACK"
        if code.startswith(WITHHELD_NOTICE_PREFIX):
            return None
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


def _withheld_notice(messages, idx):
    """The code telling the model its carrier's text at idx never reached the user. Its tools still ran."""
    at = idx + 1
    while at < len(messages) and messages[at].get("role") == "tool":
        at += 1
    if at == len(messages) or messages[at].get("role") != "user" or not is_coordinator_feedback(messages[at].get("content")):
        return None
    match = REJECTION_CODE_RE.search(messages[at]["content"])
    code = match.group(1) if match else ""
    return code if code.startswith(WITHHELD_NOTICE_PREFIX) else None


def _is_user_opening(message):
    return (
        message.get("origin") != "host"
        and parse_host_user_rejection(message) is None
        and not is_coordinator_feedback(message.get("content"))
    )


def _tool_results_in(message):
    out = []
    raw_content = message.get("content")
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
        if role in ("system", "developer", "user"):
            raw_c = msg.get("content")
            out.append({"role": role, "content": raw_c if isinstance(raw_c, str) else ""})
        elif role == "assistant":
            raw_c = msg.get("content")
            clean = model_turn(msg) or {"text": raw_c if isinstance(raw_c, str) else "", "tool_calls": []}
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
REPORT_DOCUMENT_MEMBERS = {"headline", "summary", "rating", "findings", "limits", "ask", "set_asides"}

# Rejection codes of the retired pre-Language-v1 command surface. Pairs built on them
# would penalize valid command programs and teach retired schemas.
RETIRED_COMMAND_REJECTION_PREFIXES = (
    "USE_READ_NATIVE",
    "USE_GREP_NATIVE",
    "USE_FIND_NATIVE",
    "USE_LIST_DIR_NATIVE",
    "USE_WRITE_NATIVE",
    "USE_STAT_NATIVE",
    "USE_WC_NATIVE",
    "USE_DIFF_NATIVE",
    "USE_JQ_NATIVE",
    "USE_DELETE_NATIVE",
    "USE_COPY_NATIVE",
    "USE_MOVE_NATIVE",
    "USE_MKDIR_NATIVE",
    "USE_GIT_",
    "COMMAND_NOT_ARGV",
)

# Structured domain tools that also expose in-process built-in commands inside `command`.
COMMAND_BUILTIN_EQUIVALENTS = {
    "survey_repo": ("repo-map", "survey-repo", "survey_repo"),
    "summarize": ("summarize", "outline"),
    "code_rewrite": ("code-rewrite", "code_rewrite"),
    "jq_edit": ("jq-edit", "jq_edit"),
}

SHALLOW_CMD_STAGE_RE = re.compile(
    r"^\s*(?:ls(?:\s+-[a-zA-Z0-9]+)*(?:\s+\S+)?|pwd|find\s+\S+\s+-maxdepth\s+1(?:\s+.*)?)\s*$"
)
SCOPED_CMD_READ_RE = re.compile(
    r"(?:\bsed\s+-n\s+['\"]?\d+,\d+p|\bhead\s+-n\s*\d+|\btail\s+-n\s*[+]?\d+|\boutline\b|\brepo-map\b|\bsurvey-repo\b|\bsummarize\b)"
)
UNSCOPED_CMD_CAT_RE = re.compile(r"^\s*(?:cat|nl)\b")
MUTATING_CMD_RE = re.compile(
    r"(?:>>?|2>>?|&>>?|\|\s*tee\b|<<-?['\"]?\w+"
    r"|\b(?:sed|perl|ruby)\s+(?:-[a-zA-Z]*i|--in-place)\b"
    r"|\b(?:rm|rmdir|cp|mv|mkdir|touch|chmod|chown|tar|unzip|patch|code-rewrite|jq-edit)\b"
    r"|\bgit\s+(?:add|commit|restore|checkout|switch|merge|rebase|reset|clean|apply|cherry-pick|stash\s+(?:save|push|pop|apply|drop))\b"
    r"|\b(?:npm|pnpm|yarn|bun)\s+(?:install|i|add|remove|uninstall)\b"
    r"|\b(?:pip3?\s+(?:install|uninstall)|uv\s+(?:add|remove|pip\s+install)|(?:poetry|pdm)\s+(?:add|remove|install))\b"
    r"|\b(?:cargo\s+(?:add|remove|install)|go\s+(?:get|install|mod\s+tidy)|dotnet\s+(?:add|remove|tool\s+install|new\s+install))\b"
    r"|\b(?:gem\s+(?:install|uninstall)|bundle\s+(?:add|install)|composer\s+(?:require|remove|install|create-project)|cs\s+install)\b"
    r"|\b(?:(?:dart|flutter)\s+pub\s+add|mix\s+(?:archive\.install|escript\.install|deps\.get)|cpanm?\b|luarocks\s+install|opam\s+install|conan\s+install|scarb\s+add|Install-Module\b|Install-PSResource\b)\b"
    r"|\b(?:terraform|tofu)\s+apply\b"
    r"|\b(?:python3?|node)\s+-(?:c|e)\b)"
)
# Test runners, compilers, typecheckers, and linters for every supported language.
EXEC_OR_TEST_CMD_RE = re.compile(
    r"(?:\b(?:pytest|py\.test|tox|nox|ruff|mypy|flake8|pyright|(?:uv|poetry|pdm)\s+run|python3?\s+-m\s+(?:pytest|unittest|mypy|ruff|flake8|doctest|compileall))\b"
    r"|\b(?:(?:npm|pnpm|yarn|bun)\s+(?:test|run|check|lint|build)|(?:npx|pnpx|bunx)\b|vitest|jest|mocha|ava|playwright|cypress|vue-tsc|tsc|eslint|biome|deno\s+(?:test|lint|check|run))\b"
    r"|\b(?:go\s+(?:test|build|vet|run|generate)|golangci-lint|staticcheck)\b"
    r"|\b(?:cargo\s+(?:test|nextest|check|build|clippy|run|bench)|rustc|scarb\s+(?:test|build|cairo-run)|snforge\s+test)\b"
    r"|\b(?:(?:\./)?mvnw?\b|(?:\./)?gradlew?\b|sbt\b|mill\b|lein\s+(?:test|check|compile|run)|(?:clj|clojure)\s+-M|javac|kotlinc|scalac|groovyc|cs\s+launch)\b"
    r"|\b(?:dotnet\s+(?:test|build|run|vstest|format)|msbuild|vstest\.console|Invoke-Pester|(?:pwsh|powershell)\s+-(?:Command|c|File))\b"
    r"|\b(?:bundle\s+exec|rspec|rake|minitest|rubocop|ruby\s+-(?:I|c|e))\b"
    r"|\b(?:(?:vendor/bin/)?(?:phpunit|pest)|composer\s+(?:test|run|check)|phpstan|psalm|php\s+-(?:l|f)|hh_client|hhvm)\b"
    r"|\b(?:ctest|cmake|meson\s+(?:test|compile)|ninja|g?make|scons|(?:bazel|buck2)\s+(?:test|build|run)|conan\s+build|gcc|g\+\+|clang(?:\+\+)?|cppcheck|valgrind)\b"
    r"|\b(?:(?:dart|flutter)\s+(?:test|analyze|compile|run|build)|mix\s+(?:test|compile|credo|dialyzer|run)|elixir\b|swift\s+(?:test|build|run)|xcodebuild|swiftlint)\b"
    r"|\b(?:dune\s+(?:runtest|test|build|exec)|opam\s+exec|ocaml(?:find|c|opt)|prove|perl\s+(?:-[Ic]|Makefile\.PL|Build\.PL|t/)|(?:\./)?Build\s+test|dzil\s+test|perlcritic)\b"
    r"|\b(?:busted|luacheck|luajit?\b|julia\s+(?:--project|-e)|Rscript\b|R\s+(?:CMD\s+check|-e)|sbcl\s+(?:--eval|--load|--script)|ros\s+(?:test|run)|guile|racket|raco\s+test)\b"
    r"|\b(?:forge\s+(?:test|build|script)|(?:hardhat|truffle)\s+(?:test|compile)|slither|circom\b|snarkjs\b|(?:aptos|sui)\s+move\s+(?:test|compile|build|prove))\b"
    r"|\b(?:(?:terraform|tofu)\s+(?:test|validate|plan|fmt)|tflint|docker\s+build|hadolint|buf\s+(?:lint|build|breaking|generate)|protoc|jsonnet(?:fmt)?|promtool\s+(?:test|check)|codeql\s+(?:test|query|database)|sf\s+apex\s+(?:run|get)\s+test|sfdx\s+force:apex:test:run|bats|shellcheck|(?:ba|z)?sh\s+-n|xmllint|yamllint|just\b))"
)
READONLY_ORIENTATION_CMD_RE = re.compile(
    r"^\s*(?:cat|head|tail|nl|sed\s+-n|grep|rg|find|ls|stat|wc|diff|jq|yq|git\s+(?:status|diff|log|show|blame|rev-parse|branch|stash\s+list))\b"
)


def is_retired_command_rejection(code):
    """Return True when a rejection code belongs to a retired pre-Language-v1 USE_*_NATIVE or COMMAND_NOT_ARGV rule."""
    if not code:
        return False
    c = str(code).strip()
    return any(c == p or c.startswith(p) for p in RETIRED_COMMAND_REJECTION_PREFIXES)


def _command_program_and_args(call):
    if call.get("name") != "command":
        return "", {}
    try:
        args = json.loads(call.get("arguments") or "{}")
    except ValueError:
        return "", {}
    if not isinstance(args, dict):
        return "", {}
    cmd = str(args.get("command") or args.get("cmd") or "").strip()
    if not cmd and isinstance(args.get("pipeline"), list):
        cmd = " | ".join(str(s) for s in args["pipeline"] if s)
    return cmd, args


def _is_mutating_command(call):
    """Return True when a `command` invocation mutates files, redirects output, or runs mutating Git/process actions."""
    prog, args = _command_program_and_args(call)
    if not prog and not args:
        return False
    if args.get("stdout_to") or args.get("stderr_to"):
        return True
    return bool(MUTATING_CMD_RE.search(prog))


def _is_exec_or_test_command(call):
    """Return True when a `command` invocation runs a build/test suite, verification check, or background job."""
    prog, args = _command_program_and_args(call)
    if not prog and not args:
        return False
    if args.get("verification") or args.get("background"):
        return True
    return bool(EXEC_OR_TEST_CMD_RE.search(prog))


def _is_shallow_command(call):
    """Return True when a `command` invocation only performs top-level directory listing (`ls`, `pwd`, `find -maxdepth 1`)."""
    prog, _ = _command_program_and_args(call)
    if not prog or _is_mutating_command(call) or _is_exec_or_test_command(call):
        return False
    stages = [s.strip() for s in re.split(r"(?:\|\||&&|[|;\n])", prog) if s.strip()]
    return bool(stages) and all(bool(SHALLOW_CMD_STAGE_RE.match(s)) for s in stages)


def _uses_command_builtin_for(calls, tool_name):
    """Return True when any `command` call invokes the in-process built-in equivalent of `tool_name`."""
    builtins = COMMAND_BUILTIN_EQUIVALENTS.get(tool_name) or ()
    if not builtins:
        return False
    for c in calls:
        prog, _ = _command_program_and_args(c)
        if not prog:
            continue
        for b in builtins:
            if re.search(r"(?:^|[|;&\n]\s*)%s\b" % re.escape(b), prog):
                return True
    return False


def synthesized_surface(row, min_score=3, preload_from_judge=True, observed_tools=None):
    """Compute the offered and preloaded tool/skill surface for a turn."""
    obs = set(observed_tools or ())
    if not row:
        # Preserve legacy schemas only when observed in the session; otherwise default to the Language v1 command floor.
        legacy = obs & {"read", "write", "edit", "list_dir", "grep", "find_by_name"}
        base = sorted({"command", "command_output", "command_stop", "summarize", "update_progress", "request_tools", "skills_read"} | legacy | obs)
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
        "name": row.get("surface") or (row.get("state") or {}).get("surface") or "",
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


def _completed_openings(messages):
    """Return opening user message ids that settled cleanly before any later timeout."""
    completed = set()
    cur = None
    messages = _without_notes(messages)
    for idx, msg in enumerate(messages):
        if msg.get("role") == "user":
            if _is_user_opening(msg):
                cur = msg.get("id")
            continue
        if msg.get("role") != "assistant" or not cur:
            continue
        turn = model_turn(msg)
        if turn is None:
            continue
        next_msg = messages[idx + 1] if idx + 1 < len(messages) else None
        results = _tool_results_in(next_msg) if (next_msg and next_msg.get("role") == "tool") else []
        rejs = [r for r in (parse_rejection(res) for res in results) if r is not None]
        if parse_host_user_rejection(next_msg) is not None:
            rejs.append(parse_host_user_rejection(next_msg))
        if rejs:
            continue
        call_names = {c.get("name") for c in turn["tool_calls"]}
        if "complete_leg" in call_names and _calls_succeeded(turn, results):
            completed.add(cur)
            continue
        if not turn["tool_calls"] and has_grounded_citations(msg):
            completed.add(cur)
    return completed


def _calls_succeeded(turn, results):
    """Require an explicit, matching successful result for every proposed call."""
    calls = turn.get("tool_calls") or []
    by_id = {r.get("tool_call_id"): r for r in results if r.get("tool_call_id")}
    return bool(calls) and all(
        call.get("id") in by_id and by_id[call["id"]].get("is_error") is False
        for call in calls
    )


def _is_scoped_read(call):
    name = call.get("name")
    if name == "read":
        try:
            args = json.loads(call.get("arguments") or "{}")
        except ValueError:
            return False
        return bool(isinstance(args, dict) and (set(args) & SCOPED_READ_KEYS))
    if name == "command":
        prog, _ = _command_program_and_args(call)
        return bool(prog and SCOPED_CMD_READ_RE.search(prog))
    return False


def _is_unscoped_read_call(call):
    name = call.get("name")
    if name == "read":
        return not _is_scoped_read(call)
    if name == "command":
        prog, _ = _command_program_and_args(call)
        return bool(
            prog
            and UNSCOPED_CMD_CAT_RE.search(prog)
            and not SCOPED_CMD_READ_RE.search(prog)
            and not _is_mutating_command(call)
        )
    return False


ROUTINE_ORIENTATION_TOOLS = {"read", "grep", "list_dir", "find_by_name"}
MULTI_ROOT_CMD_TOKEN_RE = re.compile(r"(?:^|\s|['\"])@[A-Za-z0-9_.-]+/")


def _has_multi_root_path(call):
    try:
        args = json.loads(call.get("arguments") or "{}")
    except ValueError:
        return False
    if not isinstance(args, dict):
        return False
    for key in ("path", "file_path", "cwd", "stdout_to", "stderr_to", "stdin_from"):
        val = args.get(key)
        if isinstance(val, str) and val.strip().startswith("@"):
            return True
    prog, _ = _command_program_and_args(call)
    return bool(prog and MULTI_ROOT_CMD_TOKEN_RE.search(prog))


def _is_routine_orientation_turn(turn, idx, steps_since_opening=1):
    """Select native orientation candidates; command effects need typed host facts."""
    if idx <= 2 or steps_since_opening <= 0:
        return False
    calls = turn.get("tool_calls") or []
    if not calls:
        return False
    for call in calls:
        if _is_scoped_read(call) or _has_multi_root_path(call):
            return False
        name = call.get("name")
        if name in ROUTINE_ORIENTATION_TOOLS:
            continue
        return False
    return True


def translate_legacy_turn_to_command(turn):
    """Historical requests cannot establish authority or outcomes on a new surface."""
    raise ValueError("legacy translation is not training evidence; collect a fresh command-surface run")


def extract(
    transcripts_path,
    rows_paths=(),
    out_sft=None,
    out_dpo=None,
    min_score=3,
    preload_from_judge=False,
    teacher_models=None,
    max_routine_reads_per_session=6,
    drop_retired_command_rejections=False,
    translate_legacy_tools=False,
):
    """Extract Stage 1 SFT rows and Stage 2 DPO pairs from transcripts and judged rows."""
    if translate_legacy_tools:
        raise ValueError("legacy trajectories cannot be translated into a different host state; collect a fresh command-surface run")
    by_session, by_opening = load_judged_index(rows_paths)
    teachers = set(teacher_models) if teacher_models else None
    sft_rows = []
    dpo_rows = []
    stats = {
        "sessions": 0,
        "sft_rows": 0,
        "dpo_rows": 0,
        "by_source": {},
        "rejection_codes": {},
        "capped_routine_sft": 0,
        "dropped_retired_rejections": 0,
        "qualified_sft_rows": 0,
        "qualified_dpo_rows": 0,
        "without_model_output": 0,
        "evidence_optional_replies": 0,
        "withheld_carriers": {},
    }

    for session in iter_transcripts(transcripts_path):
        session_sft_start, session_dpo_start = len(sft_rows), len(dpo_rows)
        stats["sessions"] += 1
        sid = session.get("session_id") or session.get("id")
        sess_rows = by_session.get(sid) or []
        primary_row = sess_rows[0] if sess_rows else None
        meta = _session_meta(session, primary_row)
        messages = _without_notes(session.get("messages") or [])
        completed_openings = _completed_openings(messages)
        called_in_session = {
            call.get("name")
            for msg in messages if msg.get("role") == "assistant"
            for call in (model_turn(msg) or {"tool_calls": []})["tool_calls"] if call.get("name")
        }
        draft_by_slot = {}
        for dv in session.get("draft_versions") or []:
            slot = dv.get("slot_id")
            if slot:
                draft_by_slot.setdefault(slot, []).append(dv)
        current_opening = None
        steps_since_opening = 0
        routine_reads_kept = 0
        routine_seen = set()
        for idx, msg in enumerate(messages):
            if msg.get("role") == "user":
                if _is_user_opening(msg):
                    current_opening = msg.get("id")
                    steps_since_opening = 0
                continue
            if msg.get("role") != "assistant":
                continue
            turn = model_turn(msg)
            slot_drafts = draft_by_slot.get(msg.get("id")) or []
            if turn is None:
                stats["without_model_output"] += 1
                if not slot_drafts:
                    continue
            step_offset = steps_since_opening
            steps_since_opening += 1
            row = by_opening.get((sid, current_opening)) or primary_row
            surface = synthesized_surface(row, min_score=min_score, preload_from_judge=preload_from_judge, observed_tools=called_in_session)
            next_msg = messages[idx + 1] if idx + 1 < len(messages) else None
            results = _tool_results_in(next_msg) if (next_msg and next_msg.get("role") == "tool") else []
            rejections = [r for r in (parse_rejection(res) for res in results) if r is not None]
            host_user_rej = parse_host_user_rejection(next_msg)
            if not rejections and host_user_rej is not None:
                rejections = [host_user_rej]
            if drop_retired_command_rejections and rejections:
                retired = [r for r in rejections if is_retired_command_rejection(r.get("code"))]
                if retired:
                    stats["dropped_retired_rejections"] += len(retired)
                    rejections = [r for r in rejections if not is_retired_command_rejection(r.get("code"))]
                    if not rejections:
                        continue
            host_assembled = is_host_assembled_closeout(msg)
            defect_code = report_document_defect_code(msg)
            grounded = has_grounded_citations(msg)
            prefix = _openai_messages(messages[:idx])
            captures = session.get("model_requests") or {}
            capture = captures.get(msg.get("id")) if isinstance(captures, dict) else None
            if isinstance(capture, dict) and isinstance(capture.get("messages"), list):
                prefix = copy.deepcopy(capture["messages"])

            # A retracted draft's body is the rejected attempt's Markdown.
            if slot_drafts:
                for dv_idx, dv in enumerate(slot_drafts):
                    dv_code = dv.get("outcome_code") or "TURN_CLOSEOUT_REJECTED"
                    stats["rejection_codes"][dv_code] = stats["rejection_codes"].get(dv_code, 0) + 1
                    rej_turn = {
                        "role": "assistant",
                        "text": str(dv.get("body") or ""),
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
                        "chosen": None,
                        "candidate_reasons": ["draft_original_request_unavailable"],
                        "rejected": rej_turn,
                        "meta": meta,
                    })
                    stats["by_source"]["host_rejection"] = stats["by_source"].get("host_rejection", 0) + 1
            elif turn is not None and (host_assembled or defect_code) and not rejections:
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

            if turn is None:
                continue
            if not turn["tool_calls"] and not rejections and not grounded:
                if closeout_outcome(msg) == "evidence_optional":
                    stats["evidence_optional_replies"] += 1
                continue

            if rejections:
                rej = rejections[0]
                code = rej["code"]
                stats["rejection_codes"][code] = stats["rejection_codes"].get(code, 0) + 1
                repaired_turn = None
                if idx + 2 < len(messages) and messages[idx + 2].get("role") == "assistant":
                    candidate_msg = messages[idx + 2]
                    candidate = model_turn(candidate_msg)
                    later_msg = messages[idx + 3] if idx + 3 < len(messages) else None
                    later_results = _tool_results_in(later_msg) if (later_msg and later_msg.get("role") == "tool") else []
                    later_rejs = [r for r in (parse_rejection(res) for res in later_results) if r is not None]
                    if parse_host_user_rejection(later_msg) is not None:
                        later_rejs.append(parse_host_user_rejection(later_msg))
                    if candidate is not None and (_calls_succeeded(candidate, later_results) or has_grounded_citations(candidate_msg)) and not later_rejs:
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
                    "chosen": None,
                    "proposed_chosen": repaired_turn,
                    "candidate_reasons": ["recovery_observed_at_different_state"],
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
                        "chosen": None,
                        "proposed_chosen": repaired_turn,
                        "candidate_reasons": ["repeated_action_not_validated_at_recovery_state"],
                        "rejected": turn,
                        "meta": meta,
                    })
                    stats["by_source"]["post_rejection_recovery"] = stats["by_source"].get("post_rejection_recovery", 0) + 1
                continue

            # Clean turn candidate for Stage 1 SFT warmup.
            if host_assembled or defect_code:
                continue
            withheld = _withheld_notice(messages, idx)
            if withheld:
                stats["withheld_carriers"][withheld] = stats["withheld_carriers"].get(withheld, 0) + 1
                continue
            task_status = ((row or {}).get("meta") or {}).get("task_status")
            sess_settled = session.get("status") in ("settled", "idle", "") and task_status in (None, "", "settled")
            opening_settled = (
                bool(current_opening and current_opening in completed_openings)
                and session.get("status") in ("settled", "idle", "", "timeout", "overdue")
                and task_status in (None, "", "settled", "timeout", "overdue")
            )
            if not (sess_settled or opening_settled):
                continue
            if teachers and meta["model"] and meta["model"] not in teachers:
                continue
            if surface["required_skills"] and "skills_read" not in called_in_session:
                continue
            if surface["required_tools"]:
                req_satisfied = bool(set(surface["required_tools"]) & called_in_session) or any(
                    _uses_command_builtin_for(turn["tool_calls"], req_t) for req_t in surface["required_tools"]
                )
                if not req_satisfied:
                    continue
            adapted = turn if capture else _adapt_turn_for_preloaded(turn, surface["active_tools"], preload_from_judge=preload_from_judge)
            prior_tool_msgs = sum(1 for m in prefix if m.get("role") == "tool")
            if not adapted["tool_calls"] and not (grounded and prior_tool_msgs >= 1):
                continue
            if (
                max_routine_reads_per_session is not None
                and max_routine_reads_per_session > 0
                and _is_routine_orientation_turn(adapted, idx, steps_since_opening=step_offset)
            ):
                by_call = {r.get("tool_call_id"): r for r in results}
                signature = json.dumps([
                    [c.get("name"), c.get("arguments"), by_call.get(c.get("id"), {}).get("content")]
                    for c in adapted["tool_calls"]
                ], sort_keys=True)
                previous = messages[idx - 1] if idx else {}
                recovery = parse_host_user_rejection(previous) or any(
                    parse_rejection(r) for r in _tool_results_in(previous)
                )
                if signature in routine_seen and not recovery:
                    if routine_reads_kept >= max_routine_reads_per_session:
                        stats["capped_routine_sft"] += 1
                        continue
                    routine_reads_kept += 1
                routine_seen.add(signature)
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

        for candidate in sft_rows[session_sft_start:] + dpo_rows[session_dpo_start:]:
            message = messages[candidate["turn_index"]]
            candidate["message_id"] = message.get("id")
            candidate["closeout"] = {"outcome": closeout_outcome(message), "report_phase": is_report_phase(message)}
            captures = session.get("model_requests")
            capture = captures.get(candidate["message_id"]) if isinstance(captures, dict) else None
            if isinstance(capture, dict):
                candidate["request_capture"] = copy.deepcopy(capture)
                candidate["prompt"] = copy.deepcopy(capture.get("messages", candidate["prompt"]))
                validations = session.get("training_validations")
                if isinstance(validations, dict) and candidate["schema"] == SFT_SCHEMA:
                    candidate["validation"] = copy.deepcopy(validations.get(candidate["message_id"]))
                tools = capture.get("tools")
                if isinstance(tools, list):
                    active = [
                        t["function"]["name"] for t in tools
                        if isinstance(t, dict) and isinstance(t.get("function"), dict) and t["function"].get("name")
                    ]
                    candidate["surface"] = dict(candidate["surface"], active_tools=active, captured=True)
            candidate["qualification"] = qualification(candidate)

    stats["sft_rows"] = len(sft_rows)
    stats["dpo_rows"] = len(dpo_rows)
    stats["qualified_sft_rows"] = sum(training_eligible(r) for r in sft_rows)
    stats["qualified_dpo_rows"] = sum(training_eligible(r) for r in dpo_rows)
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


def _is_malformed_where(where):
    """A finding's source association is typed: an evidence handle, a place, or both."""
    if not isinstance(where, dict) or not where or set(where) - {"evidence", "path", "line", "end_line", "excerpt"}:
        return True
    if "evidence" in where and not citationmd.is_handle(where["evidence"]):
        return True
    if not where.get("evidence") and not where.get("path"):
        return True
    line, end = where.get("line", 1), where.get("end_line", where.get("line", 1))
    if type(line) is not int or type(end) is not int or line < 1 or end < line:
        return True
    return any(not isinstance(where.get(key, ""), str) for key in ("path", "excerpt"))


def _is_malformed_report(teacher_fence, student_fence):
    """Judge a student's report document fence against what the teacher's document carried."""
    try:
        teacher, student = json.loads(teacher_fence or "null"), json.loads(student_fence or "null")
    except ValueError:
        return True
    if not isinstance(teacher, dict):
        return False
    if not isinstance(student, dict) or set(student) - REPORT_DOCUMENT_MEMBERS:
        return True
    if teacher.get("headline") and not str(student.get("headline") or "").strip():
        return True
    if teacher.get("rating"):
        rating = student.get("rating")
        if not isinstance(rating, dict) or rating.get("level") not in VALID_REPORT_RATINGS or not str(rating.get("why") or "").strip():
            return True
    if teacher.get("findings"):
        findings = student.get("findings")
        if not isinstance(findings, list) or not findings or any(not isinstance(f, dict) for f in findings):
            return True
        claim_ids = {f.get("id") for f in teacher["findings"] if isinstance(f, dict) and f.get("id")}
        for finding in findings:
            if finding.get("disposition") and finding["disposition"] not in VALID_FINDING_DISPOSITIONS:
                return True
            if claim_ids and not finding.get("disposition"):
                return True
            where = finding.get("where", [])
            if not isinstance(where, list) or any(_is_malformed_where(w) for w in where):
                return True
        if not claim_ids <= {f.get("id") for f in findings}:
            return True
    if teacher.get("set_asides") and not isinstance(student.get("set_asides"), list):
        return True
    return bool(teacher.get("ask") and not isinstance(student.get("ask"), dict))


def _closeout_divergence(sft_row, teacher_text, student_text):
    """Judge a student's final reply by inline citation structure, against what the teacher's carried."""
    teacher, student = citationmd.parse(teacher_text), citationmd.parse(student_text)
    source_kinds = (citationmd.CITATION, citationmd.WEB_LINK, citationmd.WORKSPACE_LINK, citationmd.REPAIR)
    if student.issues:
        return "closeout_citation_malformed"
    if student.of(citationmd.REPAIR) and not teacher.of(citationmd.REPAIR):
        return "closeout_citation_malformed"
    for kinds in (source_kinds, (citationmd.VERIFY,), (citationmd.ARTIFACT,)):
        if teacher.of(*kinds) and not student.of(*kinds):
            return "closeout_citation_malformed"
    if (sft_row.get("closeout") or {}).get("report_phase") and _is_malformed_report(teacher.report_fence, student.report_fence):
        return "closeout_report_malformed"
    return None


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
    """Return an analysis hypothesis, never a host outcome or preference label."""
    surface = sft_row.get("surface") or {}
    allowed_tools = set(surface.get("active_tools") or [])
    t_calls = teacher_turn.get("tool_calls") or []
    s_calls = student_turn.get("tool_calls") or []
    t_text = teacher_turn.get("text") or ""
    s_text = student_turn.get("text") or ""
    if not t_calls:
        return None if s_calls else _closeout_divergence(sft_row, t_text, s_text)
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
    surface_name = str(surface.get("name") or "")
    is_readonly_surface = (
        surface_name.startswith("observe_")
        or surface_name.endswith("_readonly")
        or bool(surface.get("command_is_readonly"))
    )
    if (
        is_readonly_surface
        and any(_is_mutating_command(c) for c in s_calls)
        and not any(_is_mutating_command(c) for c in t_calls)
    ):
        return "readonly_command_mutation"
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
        has_edit_or_mutating_cmd = bool(s_names & {"write", "edit", "replace_lines", "code_rewrite", "jq_edit"}) or any(
            c.get("name") == "command"
            and (_is_mutating_command(c) or _is_exec_or_test_command(c) or not _command_program_and_args(c)[0])
            for c in s_calls
        )
        if has_edit_or_mutating_cmd:
            return "missing_progress_before_edit"
        return "unclosed_progress_before_closeout"
    # A rendered mockup, a page capture and a terminal capture are the visual evidence
    # a user asked to see; prose or code in their place is the divergence.
    visual = {"capture_page", "capture_tui", "render_view"}
    if (t_names & visual) and not (s_names & visual):
        return "missed_visual_capture"
    req_tools = set(surface.get("required_tools") or [])
    teacher_req = {t for t in req_tools if t in t_names or _uses_command_builtin_for(t_calls, t)}
    missing_req = {t for t in teacher_req if t not in s_names and not _uses_command_builtin_for(s_calls, t)}
    if missing_req:
        if "command" in s_names and ("command" not in t_names or (missing_req <= {"edit", "replace_lines", "verify"})):
            return "shell_fallback"
        return "missed_loadable_tool"
    if (
        any(_is_scoped_read(c) for c in t_calls)
        and any(_is_unscoped_read_call(c) for c in s_calls)
        and not any(_is_scoped_read(c) for c in s_calls)
    ):
        return "unscoped_read"
    if "fetch_url" in t_names and "fetch_url" not in s_names and s_names <= {"web_search"}:
        return "snippet_only_web_research"
    shallow_only = {"list_dir", "summarize"}
    deep_tools = {
        "read",
        "grep",
        "survey_repo",
        "verify",
        "command",
        "task",
        "fanout_plan",
        "delegate_dispatch",
        "web_search",
        "fetch_url",
        "http_request",
    }

    def _is_shallow_call(call):
        return call.get("name") in shallow_only or (call.get("name") == "command" and _is_shallow_command(call))

    def _is_deep_call(call):
        return call.get("name") in deep_tools and not _is_shallow_call(call)

    if (
        any(_is_deep_call(c) for c in t_calls)
        and s_calls
        and all(_is_shallow_call(c) for c in s_calls)
        and not all(_is_shallow_call(c) for c in t_calls)
    ):
        return "shallow_investigation"
    return None


def _tool_schemas_for(row):
    reasons = capture_reasons(row)
    if reasons:
        raise ValueError("exact request unavailable: %s" % ", ".join(reasons))
    return copy.deepcopy(row["request_capture"]["tools"])


def shadow_signature(sft_path, student_id, seed):
    return {"format": "candidate-shadow/2", "sft_sha256": provenance.file_sha256(sft_path), "student": student_id, "seed": seed}


def shadow(factory, sft_path, out_path, student_id="gemma-4-e4b-it", workers=16, seed=7):
    """Query exact captured requests and retain divergences for independent validation."""
    sft_path, out_path = Path(sft_path), Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    partial = Path(str(out_path) + ".partial.jsonl")
    manifest = Path(str(out_path) + ".manifest.json")
    sig = shadow_signature(sft_path, student_id, seed)
    student = factory.model(student_id)
    sig["model_revision"] = student.revision
    sig["model_hf"] = student.hf
    if manifest.exists():
        try:
            if json.loads(manifest.read_text(encoding="utf-8")).get("signature") != sig:
                partial.unlink(missing_ok=True)
        except ValueError:
            partial.unlink(missing_ok=True)
    else:
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
    replayable = [r for r in sft_rows if not capture_reasons(r)]
    todo = [r for r in replayable if r["id"] not in done_ids]
    chats = factory.chats(student) if todo else []
    lock = threading.Lock()
    counts = {}

    def run_one(index_row):
        idx, row = index_row
        chat = chats[idx % len(chats)]
        row_seed = int(hashlib.sha256(("%s:%s" % (seed, row["id"])).encode()).hexdigest()[:8], 16)
        tools = _tool_schemas_for(row)
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
                "message_id": row.get("message_id"),
                "source": "shadow_divergence",
                "rejection_code": code,
                "rejection_details": {"student": student_id, "teacher": row.get("meta", {}).get("model", "")},
                "surface": row["surface"],
                "prompt": row["prompt"],
                "chosen": None,
                "proposed_chosen": row["target"],
                "rejected": student_turn,
                "meta": row["meta"],
                "request_capture": row["request_capture"],
                "candidate_reasons": ["divergence_is_not_a_validated_preference"],
            }
            pair["qualification"] = qualification(pair)
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
    return {"evaluated": len(replayable), "skipped_missing_capture": len(sft_rows) - len(replayable),
            "pairs": len(kept), "qualified_pairs": sum(training_eligible(r) for r in kept), "by_code": counts}


def repair(factory, dpo_path, out_path, teacher_id="gemma-4-31b-it", workers=16, seed=7):
    """Collect teacher proposals at exact requests; host validation remains separate."""
    dpo_path, out_path = Path(dpo_path), Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(line) for line in dpo_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    need_repair = [(i, r) for i, r in enumerate(rows) if not r.get("chosen") and not capture_reasons(r)]
    proposed = 0
    if need_repair:
        teacher = factory.model(teacher_id)
        chats = factory.chats(teacher)

        def repair_one(item):
            idx, row = item
            chat = chats[idx % len(chats)]
            row_seed = int(hashlib.sha256(("%s:%s" % (seed, row["id"])).encode()).hexdigest()[:8], 16)
            tools = _tool_schemas_for(row)
            turn = chat.turn(row["prompt"], tools=tools, temperature=0.0, seed=row_seed, thinking=False)
            turn["role"] = "assistant"
            if turn == row.get("rejected") or not (turn.get("tool_calls") or turn.get("text")):
                return idx, None
            return idx, turn

        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            for idx, proposal in pool.map(repair_one, need_repair):
                rows[idx]["proposed_chosen"] = proposal
                if proposal:
                    proposed += 1
                    rows[idx]["candidate_reasons"] = sorted(set(rows[idx].get("candidate_reasons") or []) | {"teacher_proposal_requires_host_validation"})

    with open(out_path, "w", encoding="utf-8") as fh:
        for r in rows:
            r["qualification"] = qualification(r)
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    return {"input_rows": len(rows), "queried": len(need_repair), "proposed": proposed,
            "skipped_missing_capture": sum(bool(capture_reasons(r)) for r in rows),
            "completed_pairs": sum(training_eligible(r) for r in rows)}


def synthesize_multi_root_pairs(sft_rows, companion_label="spec-reference", max_pairs=200):
    """Require real attached roots; bare session-root paths are legal."""
    raise ValueError("multi-root training requires real attached-root trajectories; synthetic prefix negatives are invalid")
