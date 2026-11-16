"""Frozen task partitions inherited by every generated descendant and worker.

A root task's partition is the one its ancestry component holds (ancestry.Assignment);
descendants, sessions and workers inherit their root's."""
import sqlite3

from . import ancestry
from .journal import CampaignFault, canonical, digest

FAMILIES = (
    "implementation_investigation", "missing_path_or_executable", "interactive_capture",
    "build_test_repair", "partial_effect_recovery", "selection_miss", "long_history", "structured_closeout",
)


class Lineage:
    def __init__(self, journal):
        self.db = journal.db
        self.db.executescript("""
        CREATE TABLE IF NOT EXISTS tasks(id TEXT PRIMARY KEY, root TEXT NOT NULL, parent TEXT,
          partition TEXT NOT NULL, family TEXT NOT NULL, source_class TEXT NOT NULL,
          spec TEXT NOT NULL, spec_hash TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS lineage_freeze(id INTEGER PRIMARY KEY CHECK(id=1), digest TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS lineage_ancestry(id INTEGER PRIMARY KEY CHECK(id=1), assignment_sha256 TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS task_sessions(epoch TEXT NOT NULL, session_id TEXT NOT NULL,
          task_id TEXT NOT NULL REFERENCES tasks(id), PRIMARY KEY(epoch,session_id));
        """)

    def roots(self, specifications, assignment):
        """Register root tasks with the partitions `assignment` gives them."""
        if self.db.execute("SELECT 1 FROM lineage_freeze").fetchone():
            raise CampaignFault("task_bank_frozen")
        with self.db:
            prior = self.db.execute("SELECT assignment_sha256 FROM lineage_ancestry").fetchone()
            if prior and prior[0] != assignment.sha256:
                raise CampaignFault("ancestry_assignment_changed")
            self.db.execute("INSERT OR IGNORE INTO lineage_ancestry VALUES(1,?)", (assignment.sha256,))
            for task in specifications:
                if "partition" in task:
                    raise CampaignFault("task_declares_partition", details={"task": task["id"]})
                if task["family"] not in FAMILIES:
                    raise CampaignFault("invalid_task_family")
                if task["source_class"] not in {"synthetic", "permitted_repository"}:
                    raise CampaignFault("source_not_permitted")
                if task["source_class"] == "permitted_repository" and not task.get("permission_sha256"):
                    raise CampaignFault("repository_permission_missing")
                try:
                    self.db.execute("INSERT INTO tasks VALUES(?,?,NULL,?,?,?,?,?)",
                                    (task["id"], task["id"], assignment.partition(ancestry.task_node(task["id"])), task["family"],
                                     task["source_class"], canonical(task), digest(task)))
                except sqlite3.IntegrityError as exc:
                    raise CampaignFault("duplicate_task_identity") from exc

    def freeze(self):
        roots = [dict(row) for row in self.db.execute("SELECT * FROM tasks WHERE parent IS NULL ORDER BY id")]
        if not roots:
            raise CampaignFault("empty_task_bank")
        for partition in ancestry.PARTITIONS:
            if {row["family"] for row in roots if row["partition"] == partition} != set(FAMILIES):
                raise CampaignFault("task_family_coverage_missing", details={"partition": partition})
        bound = self.db.execute("SELECT assignment_sha256 FROM lineage_ancestry").fetchone()
        identity = digest({"roots": roots, "ancestry": bound[0] if bound else None})
        with self.db:
            prior = self.db.execute("SELECT digest FROM lineage_freeze").fetchone()
            if prior and prior[0] != identity:
                raise CampaignFault("task_bank_changed")
            self.db.execute("INSERT OR IGNORE INTO lineage_freeze VALUES(1,?)", (identity,))
        return identity

    def task(self, task_id):
        row = self.db.execute("SELECT * FROM tasks WHERE id=?", (task_id,)).fetchone()
        if row is None:
            raise CampaignFault("unregistered_task")
        return dict(row)

    def bind_session(self, epoch, session_id, task_id):
        if not self.db.execute("SELECT 1 FROM lineage_freeze").fetchone():
            raise CampaignFault("task_bank_not_frozen")
        self.task(task_id)
        if not epoch or not session_id:
            raise CampaignFault("invalid_session_identity")
        with self.db:
            prior = self.db.execute("SELECT task_id FROM task_sessions WHERE epoch=? AND session_id=?",
                                    (epoch, session_id)).fetchone()
            if prior and prior[0] != task_id:
                raise CampaignFault("session_task_changed")
            self.db.execute("INSERT OR IGNORE INTO task_sessions VALUES(?,?,?)", (epoch, session_id, task_id))

    def session_task(self, epoch, session_id):
        row = self.db.execute("SELECT task_id FROM task_sessions WHERE epoch=? AND session_id=?",
                              (epoch, session_id)).fetchone()
        if row is None:
            raise CampaignFault("unregistered_session")
        return self.task(row[0])

    def descendant(self, task_id, parent_id, specification):
        if not self.db.execute("SELECT 1 FROM lineage_freeze").fetchone():
            raise CampaignFault("task_bank_not_frozen")
        parent = self.task(parent_id)
        with self.db:
            try:
                self.db.execute("INSERT INTO tasks VALUES(?,?,?,?,?,?,?,?)", (
                    task_id, parent["root"], parent_id, parent["partition"], parent["family"],
                    parent["source_class"], canonical(specification), digest(specification)))
            except sqlite3.IntegrityError as exc:
                raise CampaignFault("duplicate_task_identity") from exc
        return self.task(task_id)

    def admit(self, task_id, partition, spec_hash):
        task = self.task(task_id)
        if task["partition"] != partition or task["spec_hash"] != spec_hash:
            raise CampaignFault("task_lineage_mismatch")
        return task
