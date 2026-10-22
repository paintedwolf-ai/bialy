"""Live run health from durable statuses and capture protocol facts."""

import json
import sqlite3
import subprocess
from collections import Counter
from pathlib import Path

# One read transaction for a consistent status snapshot; never read error prose.
QUERIES = {
    "failed_turns": "SELECT count(*) FROM turns WHERE status = 'failed'",
    "turns": "SELECT count(*) FROM turns WHERE status IN ('failed', 'complete')",
    "codes": "SELECT coalesce(nullif(error_code,''),'submission_failed'),count(*),min(completed_at) "
             "FROM prompt_submissions WHERE status='failed' GROUP BY error_code",
    "scanner_failures": "SELECT count(*) FROM code_scans WHERE status IN ('failed','timed_out')",
    "scanner_completed": "SELECT count(*) FROM code_scans WHERE status='complete'",
    "engine_unavailable": "SELECT count(*) FROM turn_load_receipts WHERE abstained=1 AND reason IN "
                          "('engine unavailable','head not loaded','engine disabled')",
    "spend_usd": "SELECT coalesce(sum(estimated_nano_usd),0)/1000000000.0 FROM llm_calls",
}


class RunFault(RuntimeError):
    def __init__(self, code, shard):
        self.code, self.shard = code, shard
        super().__init__("%s: %s" % (shard, code))


def store_snapshot(path=None, container=None):
    if container:
        sql = "BEGIN; SELECT json_object(" + ",".join("'%s',(%s)" % (key, query if key != "codes" else
              "SELECT json_group_array(json_array(code,n,first_at)) FROM (" + query.replace(
                  "coalesce(nullif(error_code,''),'submission_failed'),count(*),min(completed_at)",
                  "coalesce(nullif(error_code,''),'submission_failed') AS code,count(*) AS n,min(completed_at) AS first_at") + ")")
              for key, query in QUERIES.items()) + "); COMMIT;"
        result = subprocess.run(["docker", "exec", container, "sqlite3", "-readonly", "-cmd", ".timeout 1000", "/cfg/store.db", sql],
                                capture_output=True, text=True, timeout=10)
        if result.returncode:
            raise RunFault("health_store_unavailable", Path(path).name if path else container)
        return json.loads(result.stdout)
    with sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True, timeout=1) as db:
        db.execute("BEGIN")
        return {key: db.execute(query).fetchall() if key == "codes" else db.execute(query).fetchone()[0] for key, query in QUERIES.items()}


def capture_snapshot(shard):
    counts, excluded, modified = Counter(), Counter(), Counter()
    committed, candidates = set(), []
    root = Path(shard) / "training-captures"
    for path in sorted([*root.glob("*.json"), *root.glob("health/*.json")]):
        # Host artifacts are atomically published. A malformed record is a fault.
        record = json.loads(path.read_text(encoding="utf-8"))
        schema = record.get("schema")
        if schema == "pw-training-health/1":
            if record.get("purpose") == "compaction":
                counts["summary_calls"] += 1
                counts["summary_completed"] += bool(record.get("success"))
                counts["summary_length"] += record.get("finish_reason") == "length"
        elif schema == "pw-training-fact/1":
            if record.get("kind") == "turn_committed":
                committed.add((record["session_id"], record["message_id"]))
        elif schema == "pw-training-turn/1":
            counts["captures"] += 1
            modified.update(record.get("modified_fields") or [])
            if record.get("exclusion_reason"):
                excluded[record["exclusion_reason"]] += 1
            elif record.get("settled") and record.get("student_request") and record.get("response"):
                counts["payloads_kept"] += 1
                candidates.append((record["session_id"], record["message_id"]))
        else:
            raise RunFault("health_capture_schema", Path(shard).name)
    counts["committed_payloads"] = len(set(candidates) & committed)
    return dict(counts, exclusion_reasons=dict(excluded), modified_fields=dict(modified))


def snapshot(shard, container=None):
    shard = Path(shard)
    result = {"shard": shard.name, **capture_snapshot(shard)}
    if (shard / "task-fault.json").exists():
        result["task_fault"] = json.loads((shard / "task-fault.json").read_text())
    if container or (shard / "store.db").exists():
        result.update(store_snapshot(shard / "store.db", container))
        result["codes"] = {code: {"count": count, "first_at": first} for code, count, first in result["codes"]}
    else:
        result["observation"] = "awaiting_store"
    return result


def reason(report, settings):
    if report.get("task_fault"):
        return report["task_fault"]["code"]
    if report.get("failed_turns", 0):
        return "turn_failed"
    if report.get("scanner_failures", 0):
        return "scanner_failed"
    if settings.get("require_engine") and report.get("engine_unavailable", 0):
        return "engine_unavailable"
    if report.get("summary_length", 0) >= settings.get("summary_length_limit", 2):
        return "summary_truncated"
    if settings.get("require_captures") and max(report.get("turns", 0), report.get("captures", 0)) >= settings.get("capture_turn_limit", 3) and not report.get("payloads_kept", 0):
        return "no_kept_captures"
    return None


def run_snapshot(run_dir, containers=None):
    containers = containers or {}
    return [snapshot(shard, containers.get(str(shard))) for shard in sorted(Path(run_dir).glob("fleet-*/shards/*"))]
