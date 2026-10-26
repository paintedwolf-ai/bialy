"""Transactional stage ownership, immutable inputs, and reconciled outcomes."""
import contextlib
import fcntl
import hashlib
import json
import sqlite3
import time
import uuid
from pathlib import Path


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


class CampaignFault(RuntimeError):
    def __init__(self, code, *, retryable=False, details=None):
        self.code, self.retryable, self.details = code, retryable, details or {}
        super().__init__(code)

    def record(self):
        return {"code": self.code, "retryable": self.retryable, "details": self.details}


class Journal:
    def __init__(self, directory, manifest):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.directory / "campaign.sqlite", timeout=30)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.executescript("""
        CREATE TABLE IF NOT EXISTS manifest(id INTEGER PRIMARY KEY CHECK(id=1), body TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS stages(name TEXT PRIMARY KEY, input_hash TEXT NOT NULL,
          status TEXT NOT NULL, attempt TEXT NOT NULL, tries INTEGER NOT NULL, output TEXT, fault TEXT);
        CREATE TABLE IF NOT EXISTS events(sequence INTEGER PRIMARY KEY, at REAL NOT NULL,
          stage TEXT NOT NULL, attempt TEXT NOT NULL, status TEXT NOT NULL, body TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS usage(id TEXT PRIMARY KEY, body TEXT NOT NULL);
        """)
        try:
            with self.db:
                current = self.db.execute("SELECT body FROM manifest WHERE id=1").fetchone()
                body = canonical(manifest)
                if current is not None and current["body"] != body:
                    raise CampaignFault("campaign_identity_changed")
                self.db.execute("INSERT OR IGNORE INTO manifest VALUES(1,?)", (body,))
        except Exception:
            self.db.close()
            raise

    def close(self):
        self.db.close()

    @contextlib.contextmanager
    def ownership(self):
        with (self.directory / ".owner.lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise CampaignFault("campaign_already_owned") from exc
            try:
                yield
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)

    def stage(self, name):
        row = self.db.execute("SELECT * FROM stages WHERE name=?", (name,)).fetchone()
        if row is None:
            return None
        return {key: json.loads(row[key]) if key in {"output", "fault"} and row[key] else row[key] for key in row.keys()}

    def event(self, name, attempt, status, body):
        self.db.execute("INSERT INTO events(at,stage,attempt,status,body) VALUES(?,?,?,?,?)",
                        (time.time(), name, attempt, status, canonical(body)))

    def begin(self, name, inputs):
        identity = digest(inputs)
        with self.db:
            current = self.stage(name)
            if current and current["input_hash"] != identity:
                raise CampaignFault("stage_inputs_changed", details={"stage": name})
            if current and current["status"] in {"running", "passed", "skipped_dependency"}:
                return current
            if current and (not current["fault"]["retryable"] or current["tries"] >= 3):
                return current
            attempt = uuid.uuid4().hex
            tries = current["tries"] + 1 if current else 1
            self.db.execute("INSERT INTO stages VALUES(?,?, 'running',?,?,NULL,NULL) "
                            "ON CONFLICT(name) DO UPDATE SET status='running',attempt=excluded.attempt,"
                            "tries=excluded.tries,output=NULL,fault=NULL", (name, identity, attempt, tries))
            self.event(name, attempt, "running", inputs)
        return self.stage(name)

    def settle(self, name, attempt, status, *, output=None, fault=None):
        if status not in {"passed", "failed", "skipped_dependency"}:
            raise ValueError("invalid terminal stage status")
        if status == "failed" and fault is None:
            raise ValueError("failed stage requires a typed fault")
        with self.db:
            current = self.stage(name)
            if not current or current["attempt"] != attempt or current["status"] != "running":
                raise CampaignFault("stage_ownership_changed")
            self.db.execute("UPDATE stages SET status=?,output=?,fault=? WHERE name=?",
                            (status, canonical(output), canonical(fault), name))
            self.event(name, attempt, status, {"output": output, "fault": fault})
        return self.stage(name)

    def usage(self, operation_id, record):
        with self.db:
            prior = self.db.execute("SELECT body FROM usage WHERE id=?", (operation_id,)).fetchone()
            body = canonical(record)
            if prior and prior["body"] != body:
                raise CampaignFault("usage_identity_changed")
            self.db.execute("INSERT OR IGNORE INTO usage VALUES(?,?)", (operation_id, body))

    def snapshot(self):
        return {"schema": "pw-training-campaign-status/1",
                "stages": {row[0]: self.stage(row[0]) for row in self.db.execute("SELECT name FROM stages ORDER BY rowid")},
                "usage": {row[0]: json.loads(row[1]) for row in self.db.execute("SELECT id,body FROM usage ORDER BY id")}}
