"""A shard's store snapshot: backed up, verified and digested before it counts as kept.

A shard is recoverable only while `store.db` and `store.json` agree. The record names
the snapshot's digest and size and the attempt that took it (completed, overdue,
aborted, salvage). A snapshot that fails verification never replaces a verified store:
it is kept for diagnosis as `store.<attempt>.unverified.db` and the failure is recorded
in `store-fault.json`. Rows are kept only beside a verified store, so every admitted
row can be exported again from the store it came from.
"""

import hashlib
import json
import os
import shutil
import sqlite3
from contextlib import closing
from pathlib import Path

STORE = "store.db"
RECORD = "store.json"
FAULT = "store-fault.json"
SCHEMA = "bialy-store-snapshot/1"
ATTEMPTS = "attempts"
# Bytes 18 and 19 of a SQLite header are the read and write versions: 1 for rollback
# journal mode, 2 for WAL. A WAL-mode file is not self-contained once copied.
ROLLBACK_HEADER = b"\x01\x01"
CONTAINER_STORE = "/cfg/store.db"
CONTAINER_SHARD = "/shard"


class PreservationFault(RuntimeError):
    """A snapshot that cannot be relied on to export the shard's rows again."""

    def __init__(self, code, detail=""):
        super().__init__("%s%s" % (code, ": " + detail if detail else ""))
        self.code = code
        self.detail = detail


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def verify(path):
    """The digest, size and table count of a readable, self-contained SQLite snapshot."""
    path = Path(path)
    if not path.is_file():
        raise PreservationFault("store_missing")
    for suffix in ("-wal", "-journal"):
        if path.with_name(path.name + suffix).exists():
            raise PreservationFault("store_not_self_contained", "side file " + suffix)
    with open(path, "rb") as fh:
        header = fh.read(100)
    if not header.startswith(b"SQLite format 3\x00"):
        raise PreservationFault("store_unreadable", "not a SQLite database")
    if header[18:20] != ROLLBACK_HEADER:
        raise PreservationFault("store_not_self_contained", "WAL journal mode")
    try:
        with closing(sqlite3.connect(path.resolve().as_uri() + "?mode=ro&immutable=1", uri=True)) as db:
            check = db.execute("PRAGMA quick_check").fetchall()
            tables = db.execute("SELECT count(*) FROM sqlite_master WHERE type = 'table'").fetchone()[0]
    except sqlite3.DatabaseError as exc:
        raise PreservationFault("store_unreadable", type(exc).__name__) from exc
    if check != [("ok",)]:
        raise PreservationFault("store_corrupt", "quick_check failed")
    if not tables:
        raise PreservationFault("store_empty")
    return {"sha256": digest(path), "bytes": path.stat().st_size, "tables": tables}


def write_json(path, value):
    """Write a JSON file so a reader sees the old or the new content, never a part."""
    path = Path(path)
    part = path.with_name(path.name + ".part")
    part.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(part, path)


def record_fault(shard, attempt, fault, candidate=None):
    """Record a failed preservation and keep the candidate snapshot for diagnosis."""
    shard = Path(shard)
    kept = None
    if candidate is not None and Path(candidate).exists():
        kept = shard / ("store.%s.unverified.db" % attempt)
        os.replace(candidate, kept)
    write_json(shard / FAULT, {"code": fault.code, "detail": fault.detail, "attempt": attempt,
                               "unverified": kept.name if kept else None})


def publish(shard, candidate, attempt):
    """Make a verified candidate the shard's store and record it; raises PreservationFault
    and leaves any earlier verified store in place when the candidate does not verify."""
    shard, candidate = Path(shard), Path(candidate)
    try:
        facts = verify(candidate)
    except PreservationFault as fault:
        record_fault(shard, attempt, fault, candidate)
        raise
    os.replace(candidate, shard / STORE)
    write_json(shard / RECORD, dict(schema=SCHEMA, attempt=attempt, **facts))
    (shard / FAULT).unlink(missing_ok=True)
    return facts


def adopt(shard, attempt):
    """Verify and record a store written into the shard by its runner."""
    shard = Path(shard)
    store = shard / STORE
    try:
        facts = verify(store)
    except PreservationFault as fault:
        record_fault(shard, attempt, fault, store if store.exists() else None)
        raise
    write_json(shard / RECORD, dict(schema=SCHEMA, attempt=attempt, **facts))
    (shard / FAULT).unlink(missing_ok=True)
    return facts


def container_backup(sh, container, shard, attempt):
    """Back a running container's live store up into the shard and publish it.

    `sh` runs a command and returns a CompletedProcess. The backup goes through SQLite's
    backup API, so committed WAL contents are included, and is switched to rollback
    journal mode so the snapshot opens without side files."""
    shard = Path(shard)
    part = "store.%s.part" % attempt
    target = "%s/%s" % (CONTAINER_SHARD, part)
    (shard / part).unlink(missing_ok=True)
    for args in (("sqlite3", CONTAINER_STORE, ".backup '%s'" % target), ("sqlite3", target, "PRAGMA journal_mode=DELETE;")):
        result = sh("docker", "exec", container, *args, check=False)
        if result.returncode != 0:
            fault = PreservationFault("store_backup_failed", "exit %d" % result.returncode)
            record_fault(shard, attempt, fault, shard / part)
            raise fault
    return publish(shard, shard / part, attempt)


def recorded(shard):
    """The shard's preservation record, or None when no verified store is recorded."""
    shard = Path(shard)
    record, store = shard / RECORD, shard / STORE
    if not record.is_file() or not store.is_file():
        return None
    value = json.loads(record.read_text(encoding="utf-8"))
    if value.get("schema") != SCHEMA or value.get("bytes") != store.stat().st_size:
        return None
    return value


def check(shard):
    """The recorded store, digested again; raises PreservationFault when it no longer
    matches its record."""
    value = recorded(shard)
    if value is None:
        raise PreservationFault("store_unrecorded")
    if digest(Path(shard) / STORE) != value["sha256"]:
        raise PreservationFault("store_changed", "digest differs from its record")
    return value


def copy_verified(source_shard, dest_shard):
    """Copy a shard's verified store and record into another shard directory. The copy is
    published only after its digest matches, so an interrupted copy leaves no store."""
    value = check(source_shard)
    dest_shard = Path(dest_shard)
    part = dest_shard / (STORE + ".part")
    shutil.copyfile(Path(source_shard) / STORE, part)
    if digest(part) != value["sha256"]:
        part.unlink()
        raise PreservationFault("store_copy_mismatch")
    os.replace(part, dest_shard / STORE)
    write_json(dest_shard / RECORD, value)
    return value


def archive_attempt(shard, names):
    """Move an earlier attempt's store evidence and the named files into attempts/<n>/
    before the shard is driven again, so no attempt overwrites another's evidence."""
    shard = Path(shard)
    evidence = [shard / STORE, shard / RECORD, shard / FAULT, *shard.glob("store.*.unverified.db"), *(shard / n for n in names)]
    present = [p for p in evidence if p.exists()]
    if not present:
        return None
    n = 1
    while (shard / ATTEMPTS / str(n)).exists():
        n += 1
    dest = shard / ATTEMPTS / str(n)
    dest.mkdir(parents=True)
    for path in present:
        os.replace(path, dest / path.name)
    return dest
