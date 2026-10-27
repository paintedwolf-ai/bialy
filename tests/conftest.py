import json

import pytest

from bialy import config

FIXTURES = __import__("pathlib").Path(__file__).parent / "fixtures"


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


@pytest.fixture
def split_dir(tmp_path):
    """train/val/holdout files that split correctly: zod is held out."""
    out = tmp_path / "split"
    out.mkdir()
    layout = {"train.jsonl": [("flask", "flask:a", 1), ("flask", "flask:a", 2), ("gin", "gin:b", 3)],
              "val.jsonl": [("gin", "gin:c", 4)],
              "holdout.jsonl": [("zod", "zod:d", 5)]}
    for name, rows in layout.items():
        (out / name).write_text("".join(json.dumps(row(*r)) + "\n" for r in rows))
    return out


@pytest.fixture
def driven_file(tmp_path):
    """tasks-driven.jsonl for the split fixture's five rows."""
    path = tmp_path / "tasks-driven.jsonl"
    path.write_text("".join(json.dumps({"id": "t%d" % n, "prompt": "request %d" % n, "outcome": {"status": "settled", "run": "r"}}) + "\n"
                            for n in range(1, 6)))
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


class ReviewCampaign:
    """A campaign journal with recorded trials, as the review gates and command read them."""
    def __init__(self, journal, review, lineage):
        self.journal, self.review, self.lineage = journal, review, lineage

    def task(self, task_id, family='build_test_repair', partition='system_test'):
        self.lineage.roots([{'id': task_id, 'partition': partition, 'family': family, 'source_class': 'synthetic',
                             'prompt': 'Add a --json flag to the list command and verify it'}])
        return json.loads(self.journal.db.execute('SELECT spec FROM tasks WHERE id=?', (task_id,)).fetchone()[0])

    def trial(self, name, reviews=('pass', 'pass'), deterministic=True, exit_status=0, record=True):
        from bialy.campaign.grading import DIMENSIONS
        directory = self.journal.directory / 'trials' / name
        directory.mkdir(parents=True)
        facts = {'version': 1, 'root_session_id': 'root', 'sessions': [{'id': 'root', 'parent_id': '', 'messages': [
            {'role': 'user', 'content': 'Add a --json flag to the list command and verify it'},
            {'role': 'assistant', 'content': 'Running the tests.',
             'tool_calls': [{'id': 'call-1', 'name': 'command', 'args': {'argv': ['go', 'test', './...']}}]},
            {'role': 'tool', 'content': '', 'tool_result': {'tool_call_id': 'call-1', 'content': '\n'.join(
                'FAIL line %d' % index for index in range(20))}},
            {'role': 'assistant', 'content': 'Done, the --json flag works.'}],
            'invocations': [{'id': 'invocation-1', 'tool': 'command', 'tool_call_id': 'call-1', 'status': 'completed'}]}]}
        (directory / 'host-evidence.json').write_text(json.dumps(facts))
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
    campaign = ReviewCampaign(journal, Review(journal.directory), Lineage(journal))
    yield campaign
    journal.close()
