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
    assert [item['judge_id'] for item in reviews] == ['teacher', 'first', 'second']
    assert all(item['controls_passed'] is True and item['controls_sha256'] for item in reviews)
    journal.close()


def test_one_qualified_same_family_reviewer_is_sufficient(tmp_path):
    journal = Journal(tmp_path, {})
    judges = Reviewers(journal, Factory(), ['teacher'])
    judges.qualify(controls())
    reviews = judges.review({'teacher_family': 'teacher', 'evidence_ids': ['check']}, [])
    assert len(reviews) == 1 and reviews[0]['judge_family'] == 'teacher'
    journal.close()


def test_no_qualified_reviewer_fails_without_inventing_a_review(tmp_path):
    journal = Journal(tmp_path, {})
    judges = Reviewers(journal, Factory(bad={'teacher'}), ['teacher'])
    judges.qualify(controls())
    with pytest.raises(CampaignFault, match='no_qualified_episode_reviewer'):
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


def test_multiple_slots_can_assign_the_same_qualified_model(tmp_path):
    journal = Journal(tmp_path, {})
    judges = Reviewers(journal, Factory(), ['teacher', 'teacher'])
    judges.qualify(controls())
    reviews = judges.review({'teacher_family': 'teacher', 'evidence_ids': ['check']}, [])
    assert [review['reviewer_slot'] for review in reviews] == ['0', '1']
    assert [review['judge_id'] for review in reviews] == ['teacher', 'teacher']
    journal.close()


def test_episode_review_reuses_completed_slots_without_another_model_call(tmp_path):
    journal = Journal(tmp_path, {})
    factory = Factory()
    reviewers = Reviewers(journal, factory, ['first', 'second'])
    reviewers.qualify(controls())
    receipt = {'evidence_ids': ['check']}
    result = reviewers.review(receipt, [])
    calls = len(factory.calls)
    assert reviewers.review(receipt, []) == result
    assert len(factory.calls) == calls
    journal.close()


def test_unknown_review_result_is_not_blindly_repeated(tmp_path, monkeypatch):
    journal = Journal(tmp_path, {})
    factory = Factory()
    reviewers = Reviewers(journal, factory, ['first'])
    reviewers.qualify(controls())
    def disconnected(*args):
        raise OSError('lost provider response')
    monkeypatch.setattr(reviewers, 'ask', disconnected)
    with pytest.raises(OSError):
        reviewers.review({'evidence_ids': ['check']}, [])
    monkeypatch.setattr(reviewers, 'ask', lambda *args: pytest.fail('review was repeated'))
    with pytest.raises(CampaignFault, match='episode_review_outcome_unknown'):
        reviewers.review({'evidence_ids': ['check']}, [])
    journal.close()
