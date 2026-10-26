import json
from types import SimpleNamespace

import pytest

from bialy.campaign.grading import DIMENSIONS
from bialy.campaign.journal import CampaignFault, Journal
from bialy.campaign.review import Reviewers


def controls():
    return [{'id': name, 'payload': {'evidence_ids': ['check'], 'fixture': name},
             'expected': dict.fromkeys(DIMENSIONS, name)} for name in ('pass', 'fail')]


class Factory:
    def __init__(self, bad=()):
        self.bad, self.calls = bad, []
    def model(self, name):
        return SimpleNamespace(id=name, revision='revision', family=name, roles=['judge'])
    def chats(self, model):
        def turn(messages, **kwargs):
            payload = json.loads(messages[-1]['content'])
            self.calls.append((model.id, payload))
            verdict = 'pass' if model.id in self.bad else payload.get('fixture', 'pass')
            return {'text': json.dumps({'dimensions': dict.fromkeys(DIMENSIONS, verdict), 'evidence_ids': ['check'], 'reason': 'fixture'})}
        return [SimpleNamespace(turn=turn)]


def test_controls_are_persisted_and_unqualified_judges_never_review(tmp_path):
    journal = Journal(tmp_path, {})
    factory = Factory(bad={'unreliable'})
    judges = Reviewers(journal, factory, ['teacher', 'unreliable', 'first', 'second'])
    report = judges.qualify(controls())
    assert not report['unreliable']['passed']
    calls = len(factory.calls)
    assert judges.qualify(controls()) == report and len(factory.calls) == calls
    reviews = judges.review({'teacher_family': 'teacher', 'evidence_ids': ['check']}, [])
    assert [item['judge_id'] for item in reviews] == ['first', 'second']
    assert all(item['controls_passed'] is True and item['controls_sha256'] for item in reviews)
    journal.close()


def test_two_distinct_qualified_reviewers_are_required(tmp_path):
    journal = Journal(tmp_path, {})
    judges = Reviewers(journal, Factory(), ['teacher', 'other'])
    judges.qualify(controls())
    with pytest.raises(CampaignFault, match='insufficient_qualified'):
        judges.review({'teacher_family': 'teacher', 'evidence_ids': ['check']}, [])
    journal.close()


def test_changed_controls_do_not_reuse_prior_qualification(tmp_path):
    journal = Journal(tmp_path, {})
    judges = Reviewers(journal, Factory(), ['first', 'second'])
    judges.qualify(controls())
    changed = controls()
    changed[0]['payload']['fixture_detail'] = 'different'
    with pytest.raises(CampaignFault, match='controls_changed'):
        judges.qualify(changed)
    journal.close()
