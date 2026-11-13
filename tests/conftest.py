import json
from pathlib import Path

import pytest

from bialy import config
from bialy.campaign.scenarios import load as load_scenarios

FIXTURES = Path(__file__).parent / "fixtures"
# The scenario catalog's focus IDs, which every lineage covers.
FOCUSES = load_scenarios().focuses


@pytest.fixture(scope="session")
def factory():
    return config.load()


def row(repo, group, n, host="coordinator", judge_model="glm-5.3-flash", **labels):
    """A minimal pw-decide-row/1 row, as decide export writes it plus factory metadata."""
    base = {"tools": [], "requests": [], "requested_names": [], "requested_groups": [], "skills": [], "kind": "inspect", "guides": {}}
    base.update(labels)
    return {
        "schema": "pw-decide-row/1", "receipt": n, "session": "s%d" % n, "root_session": "s%d" % n, "opening_message_id": "m%d" % n,
        "host": host, "surface": "implement_investigate", "catalog_revision": "rev", "project": repo, "model": "qwen3.6-35b-a3b",
        "partial": False, "lang": "en", "engine": {"state": "abstained", "reason": "engine disabled", "preloaded": [], "omitted": []},
        "offered": {"floor": ["read"], "loadable": ["edit", "command"], "guides": []},
        "state": {"host": host, "user": "request %d" % n, "surface": "implement_investigate"},
        "labels": base, "judge": {"model": judge_model},
        "meta": {"workspace": repo, "prompt_group": group, "task_id": "t%d" % n},
    }


# Each split file's rows as (workspace, prompt group, task number, partition).
SPLIT_LAYOUT = {"train.jsonl": [("flask", "flask:a", 1, "train"), ("flask", "flask:a", 2, "train"), ("gin", "gin:b", 3, "train")],
                "val.jsonl": [("gin", "gin:c", 4, "validation")],
                "development.jsonl": [("httpx", "httpx:e", 6, "development")],
                "holdout.jsonl": [("zod", "zod:d", 5, "system_test")]}


def layout_ancestry(path):
    """The assignment the split fixture was made by, each task placed explicitly."""
    from bialy.campaign import ancestry

    graph = ancestry.Graph()
    for rows in SPLIT_LAYOUT.values():
        for _, _, n, partition in rows:
            graph.add(ancestry.task_node("t%d" % n), "task", source_class="synthetic", holdout=partition)
    return ancestry.allocate(graph, 7, {"train": 1.0}).save(path)


@pytest.fixture
def split_dir(tmp_path):
    """Split files that match SPLIT_LAYOUT: zod is held out."""
    out = tmp_path / "split"
    out.mkdir()
    layout = {name: [(workspace, group, n) for workspace, group, n, _ in rows] for name, rows in SPLIT_LAYOUT.items()}
    for name, rows in layout.items():
        (out / name).write_text("".join(json.dumps(row(*r)) + "\n" for r in rows))
    return out


@pytest.fixture
def driven_file(tmp_path):
    """tasks-driven.jsonl for the split fixture's six rows."""
    path = tmp_path / "tasks-driven.jsonl"
    path.write_text("".join(json.dumps({"id": "t%d" % n, "prompt": "request %d" % n, "outcome": {"status": "settled", "run": "r"}}) + "\n"
                            for n in range(1, 7)))
    return path


@pytest.fixture
def corpus_file(tmp_path):
    path = tmp_path / "corpus.json"
    path.write_text(json.dumps({"catalog_revision": "rev", "skills": [{"name": "commit", "card": "Commit in logical groups."}],
                                "tools": [{"name": "edit", "card": "Edit a file."}]}))
    return path


@pytest.fixture(autouse=True)
def anchors(tmp_path, monkeypatch):
    """Anchors a test's releases record land in the test's own directory."""
    from bialy import anchor

    monkeypatch.setattr(anchor, "ANCHORS", tmp_path / "releases")
    return tmp_path / "releases"


@pytest.fixture(autouse=True)
def license_texts(monkeypatch):
    """Release builds read license texts from seed clones the tests do not have."""
    from bialy import repos

    monkeypatch.setattr(repos, "license_text", lambda factory, repo, name: ("%s %s\n" % (repo.name, name)).encode())


def ancestry_node(task_id):
    from bialy.campaign import ancestry
    return ancestry.task_node(task_id)


def review_assignment():
    """One assignment for every task the review tests register: an id's prefix names
    its partition, and any other id is system_test."""
    from bialy.campaign import ancestry

    ids = ['held-out', 't1', 't2', 't3', 't4'] + ['train-' + name for name in ('a', 'b', 'c', 'spot', 'split', 'spot_check')]
    ids += [partition + family for partition in ancestry.PARTITIONS for family in FOCUSES]
    graph = ancestry.Graph()
    for task_id in ids:
        partition = next((p for p in ancestry.PARTITIONS if task_id.startswith(p)), 'system_test')
        graph.add(ancestry.task_node(task_id), 'task', source_class='synthetic', holdout=partition)
    return ancestry.allocate(graph, 7, {'train': 1.0})


class ReviewCampaign:
    """A campaign journal with recorded trials, as the review gates and command read them."""
    def __init__(self, journal, review, lineage):
        self.journal, self.review, self.lineage = journal, review, lineage

    def task(self, task_id, family='build_test_repair', partition='system_test'):
        """Register one task from the review bank's single assignment (review_assignment)."""
        assignment = review_assignment()
        if assignment.partition(ancestry_node(task_id)) != partition:
            raise AssertionError('review task %s belongs in %s' % (task_id, assignment.partition(ancestry_node(task_id))))
        self.lineage.roots([{'id': task_id, 'focus': family, 'source_class': 'synthetic',
                             'prompt': 'Add a --json flag to the list command and verify it'}], assignment)
        return json.loads(self.journal.db.execute('SELECT spec FROM tasks WHERE id=?', (task_id,)).fetchone()[0])

    messages = [
        {'role': 'user', 'content': 'Add a --json flag to the list command and verify it'},
        {'role': 'assistant', 'content': 'Running the tests.',
         'tool_calls': [{'id': 'call-1', 'name': 'command', 'args': {'argv': ['go', 'test', './...']}}]},
        {'role': 'tool', 'content': '', 'tool_result': {'tool_call_id': 'call-1', 'content': '\n'.join(
            'FAIL line %d' % index for index in range(20))}},
        {'role': 'assistant', 'content': 'Done, the --json flag works.'}]

    def trial(self, name, reviews=('pass', 'pass'), deterministic=True, exit_status=0, record=True):
        from host_bundles import operation, write_episode

        from bialy.campaign.grading import DIMENSIONS
        directory = self.journal.directory / 'trials' / name
        directory.mkdir(parents=True)
        write_episode(directory / 'host-bundle', sessions=[{'id': 'root', 'messages': self.messages}],
                      operations=[operation('invocation-1', call='call-1', tool='command')])
        passed = deterministic and exit_status == 0 and all(value == 'pass' for value in reviews)
        output = {'success': passed, 'exit_status': exit_status, 'directory': str(directory),
                  'human_interventions': 0, 'unauthorized_effects': 0, 'effective_thinking': False,
                  'unsupported_completion_claims': int(any(value != 'pass' for value in reviews)),
                  'receipt': {'checks': [{'id': 'tests', 'passed': deterministic}], 'evidence_ids': ['invocation-1', 'tests']},
                  'verdict': {'deterministic': deterministic, 'episode_sha256': 'episode-' + name},
                  'reviews': [{'judge_id': 'judge-%d' % index, 'reason': 'reviewer %d reason' % index,
                               'dimensions': {key: value for key in DIMENSIONS}} for index, value in enumerate(reviews)]}
        if record:
            stage = self.journal.begin(name, {})
            self.journal.settle(name, stage['attempt'], 'passed', output=output)
        return output

    def verdict(self, value='pass'):
        from bialy.campaign.grading import DIMENSIONS
        return dict.fromkeys(DIMENSIONS, value), dict.fromkeys(DIMENSIONS, ['tests'])


@pytest.fixture
def review_campaign(tmp_path):
    from bialy.campaign.human_review import Review
    from bialy.campaign.journal import Journal
    from bialy.campaign.lineage import Lineage
    journal = Journal(tmp_path / 'campaign', {'schema': 'fixture', 'configuration': {'human_review': True}})
    campaign = ReviewCampaign(journal, Review(journal.directory), Lineage(journal, FOCUSES))
    yield campaign
    journal.close()


def register_bank(lineage, tasks):
    """Register root tasks whose partitions a test fixes: each task's `partition` key is
    taken out of its specification and placed by an explicit holdout in its own ancestry
    component. Returns the assignment."""
    from bialy.campaign import ancestry

    graph = ancestry.Graph()
    for task in tasks:
        graph.add(ancestry.task_node(task["id"]), "task", source_class="synthetic", holdout=task.pop("partition"))
    assignment = ancestry.allocate(graph, 7, {"train": 1.0})
    lineage.roots(tasks, assignment)
    return assignment


def sqlite_store(path, roots=(), wal=False):
    """A real SQLite store holding one exported row per root session, as a sidecar's
    store would hold the sessions a shard drove."""
    import sqlite3

    path = Path(path)
    with sqlite3.connect(path) as db:
        db.execute("PRAGMA journal_mode=%s" % ("WAL" if wal else "DELETE"))
        db.execute("CREATE TABLE IF NOT EXISTS turns(root_session TEXT, receipt INTEGER)")
        db.executemany("INSERT INTO turns VALUES(?, ?)", [(root, n) for n, root in enumerate(roots)])
    return path


class FakeDocker:
    """Answers the `docker exec <container> sqlite3 ...` commands the fleet uses to
    snapshot a container's store, with /cfg/store.db at `live` and /shard at `shard`.
    `fail` names the step that exits nonzero: "backup" or "journal"."""

    def __init__(self, live, shard, fail=None):
        self.live, self.shard, self.fail, self.calls = Path(live), Path(shard), fail, []

    def host(self, container_path):
        if container_path == "/cfg/store.db":
            return self.live
        return self.shard / container_path.removeprefix("/shard/")

    def __call__(self, *args, check=True, **kw):
        import sqlite3
        import subprocess as sp

        self.calls.append(args)
        if args[:2] != ("docker", "exec") or args[3] != "sqlite3":
            return sp.CompletedProcess(args, 0, "", "")
        source, command = self.host(args[4]), args[5]
        if command.startswith(".backup"):
            if self.fail == "backup":
                return sp.CompletedProcess(args, 1, "", "disk I/O error")
            target = self.host(command.split("'")[1])
            with sqlite3.connect(source) as src, sqlite3.connect(target) as dst:
                src.backup(dst)
            return sp.CompletedProcess(args, 0, "", "")
        if self.fail == "journal":
            return sp.CompletedProcess(args, 1, "", "locked")
        with sqlite3.connect(source) as db:
            db.execute(command.rstrip(";"))
        return sp.CompletedProcess(args, 0, "", "")


EXPORTER = '''#!/usr/bin/env python3
import json, sqlite3, sys
args = sys.argv[1:]
def opt(name):
    return args[args.index(name) + 1] if name in args else None
roots = set(open(opt("--roots")).read().split()) if opt("--roots") else None
db = sqlite3.connect("file:%s?mode=ro" % opt("--db"), uri=True)
rows = [{"root_session": r, "receipt": n} for r, n in db.execute("SELECT root_session, receipt FROM turns ORDER BY receipt")
        if roots is None or r in roots]
for flag, kind in (("--out", None), ("--retention", "retention"), ("--requests", "requests")):
    if opt(flag):
        with open(opt(flag), "w") as fh:
            for row in rows:
                fh.write(json.dumps(dict(row, kind=kind) if kind else row) + "\\n")
'''


def exporter(path):
    """A stand-in for lycaon-debug decide export that reads the turns table of the store
    it is given, so an export is a function of the store alone."""
    path = Path(path)
    path.write_text(EXPORTER)
    path.chmod(0o755)
    return path
