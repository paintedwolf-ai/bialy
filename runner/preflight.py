"""Runner-local probes and a free OpenAI-compatible capture fixture."""

import json
import os
import re
import shutil
import sqlite3
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.request import Request, urlopen


class ProbeFailure(RuntimeError):
    pass


def api(path, body=None):
    token = Path("/cfg/api.token").read_text().strip()
    request = Request("http://127.0.0.1:8850" + path, data=json.dumps(body).encode() if body is not None else None,
                      headers={"Authorization": "Bearer " + token, "Content-Type": "application/json"})
    with urlopen(request, timeout=120) as response:
        return json.load(response)


def before(project):
    probes = api("/v1/preflight")["probes"]
    if os.environ.get("LYCAON_DECIDE_DISABLED") != "1":
        engine = next((p for p in probes if p["id"] == "decision_engine"), None)
        if not engine or engine["status"] != "ok" or os.environ.get("LYCAON_DECIDE_DEVICE") != "cpu":
            raise ProbeFailure("preflight_engine_unavailable")
    scanner = next((p for p in probes if p["id"] == "scanner_engine"), None)
    if not scanner or scanner["status"] != "ok":
        raise ProbeFailure("preflight_scanner_unavailable")
    Path(project, "fixture.py").write_text("def value():\n    return 1\n")
    project_id = api("/v1/projects", {"name": "preflight", "roots": [{"path": project}]})["id"]
    api("/v1/projects/" + project_id + "/scans", {"scanner_ids": ["lycaon-sast"]})
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        with sqlite3.connect("file:/cfg/store.db?mode=ro", uri=True) as db:
            statuses = [r[0] for r in db.execute("SELECT status FROM code_scans WHERE scanner_id='lycaon-sast'")]
        if "complete" in statuses:
            return
        if any(s in {"failed", "timed_out", "canceled"} for s in statuses):
            raise ProbeFailure("preflight_scanner_dispatch_failed")
        time.sleep(1)
    raise ProbeFailure("preflight_scanner_dispatch_timeout")


def after():
    entries = [json.loads(line) for line in Path("/shard/manifest.jsonl").read_text().splitlines() if line.strip()]
    if not entries or any(e["status"] != "settled" for e in entries):
        raise ProbeFailure("preflight_turn_failed")
    if os.environ["BIALY_PREFLIGHT"] == "fixture":
        records = [json.loads(p.read_text()) for p in Path("/shard/training-captures").glob("*.json")]
        commits = {(r["session_id"], r["message_id"]) for r in records
                   if r.get("schema") == "pw-training-fact/1" and r.get("kind") == "turn_committed"}
        kept = [r for r in records if r.get("schema") == "pw-training-turn/1" and not r.get("exclusion_reason") and
                r.get("settled") and r.get("student_request") and r.get("response") and (r["session_id"], r["message_id"]) in commits]
        if not kept or any(r["student_kind"] != "openai-compatible" or r["student_model"] != os.environ["PW_TRAINING_STUDENT_MODEL"] for r in kept):
            raise ProbeFailure("preflight_capture_not_kept")
        session = entries[-1]["root_session"]
        compact = api("/v1/sessions/" + session + "/compact", {})
        if compact.get("generation", 0) <= 0 or compact["tokens_after"] >= compact["tokens_before"]:
            raise ProbeFailure("preflight_compaction_no_reduction")
        wait_summary(session)
    Path("/shard/preflight.json").write_text(json.dumps({"status": "passed", "phase": os.environ["BIALY_PREFLIGHT"]}) + "\n")


def configure(source, destination):
    # Only the fixture's reduction thresholds change; the serving window and
    # provider request controls remain those of the image being qualified.
    shutil.copytree(Path(source) / "config", Path(destination) / "config")
    path = Path(destination) / "config/packs/painted-wolf/platform/host/compaction.yaml"
    body = path.read_text()
    for key, value in {"budget_trigger_pct": 99, "target_tokens_pct": 10, "keep_recent_messages": 2, "prune_protect_tail_messages": 2}.items():
        body, count = re.subn(r"(?m)^" + key + r":.*$", key + ": " + str(value), body)
        if count != 1:
            raise ProbeFailure("preflight_compaction_config_unknown")
    path.write_text(body)


def wait_summary(session, seconds=120):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        records = [json.loads(p.read_text()) for p in Path("/shard/training-captures/health").glob("*.json")]
        summaries = [r for r in records if r.get("schema") == "pw-training-health/1" and r.get("purpose") == "compaction"]
        if any(not r["success"] or r.get("finish_reason") == "length" for r in summaries):
            raise ProbeFailure("preflight_compaction_failed")
        with sqlite3.connect("file:/cfg/store.db?mode=ro", uri=True) as db:
            row = db.execute("SELECT json_extract(view_json,'$.continuation.summary_through_ord'), "
                             "json_extract(view_json,'$.continuation.through_ord') FROM compaction_views WHERE session_id=?", (session,)).fetchone()
        if summaries and row and row[0] and row[0] >= row[1]:
            return
        time.sleep(0.25)
    raise ProbeFailure("preflight_summary_not_published")


class Fixture(BaseHTTPRequestHandler):
    summary_lock = threading.Lock()
    summary_calls = 0
    def log_message(self, *_args):
        pass

    def do_GET(self):
        body = json.dumps({"object": "list", "data": [{"id": "fixture", "object": "model"}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        # Enough settled history to exercise host compaction, without paid driving.
        text = "\n".join("Fixture observation %d: the marker is stable and the next check can proceed." % i for i in range(80))
        response_format = request.get("response_format") or {}
        if response_format.get("type") == "json_schema" and (response_format.get("json_schema") or {}).get("name") == "compaction_summary":
            with Fixture.summary_lock:
                if Fixture.summary_calls:
                    self.send_response(400)
                    self.end_headers()
                    return
                Fixture.summary_calls += 1
            upstream = Request(os.environ["BIALY_PREFLIGHT_UPSTREAM"], data=json.dumps(request).encode(),
                               headers={"Content-Type": "application/json", "Authorization": self.headers.get("Authorization", "")})
            try:
                with urlopen(upstream, timeout=120) as response:
                    body = response.read()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(body)
            except Exception:
                self.send_response(400)
                self.end_headers()
            return
        response_id = "chatcmpl-" + uuid.uuid4().hex
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream" if request.get("stream") else "application/json")
        self.end_headers()
        if request.get("stream"):
            for delta, reason in (({"role": "assistant", "content": text}, None), ({}, "stop")):
                chunk = {"id": response_id, "object": "chat.completion.chunk", "model": request.get("model", "fixture"),
                         "choices": [{"index": 0, "delta": delta, "finish_reason": reason}]}
                self.wfile.write(("data: " + json.dumps(chunk) + "\n\n").encode())
            self.wfile.write(b"data: [DONE]\n\n")
        else:
            self.wfile.write(json.dumps({"id": response_id, "object": "chat.completion", "model": request.get("model", "fixture"),
                                        "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
                                        "usage": {"prompt_tokens": 32, "completion_tokens": 32, "total_tokens": 64}}).encode())


def retain_store(source=Path('/cfg/store.db'), destination=Path('/shard/preflight-store.db')):
    if not source.is_file():
        return False
    try:
        with sqlite3.connect(source.resolve().as_uri() + '?mode=ro', uri=True) as original:
            with sqlite3.connect(destination) as retained:
                original.backup(retained)
        destination.chmod(0o600)
        return True
    except (OSError, sqlite3.Error):
        return False


def main():
    try:
        if sys.argv[1] == "fixture":
            ThreadingHTTPServer(("127.0.0.1", 8851), Fixture).serve_forever()
        elif sys.argv[1] == "configure":
            configure(os.environ["LYCAON_CONFIG_ROOT"], "/cfg/preflight-overlay")
        elif sys.argv[1] == "before":
            before(sys.argv[2])
        else:
            after()
    except Exception as exc:
        # Provider bodies and credentials never enter the report.
        code = str(exc) if isinstance(exc, ProbeFailure) else "preflight_probe_error"
        Path("/shard/preflight.json").write_text(json.dumps({"status": "failed", "code": code, "error_type": type(exc).__name__, "store_retained": retain_store()}) + "\n")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
