import json

import pytest
from conftest import FOCUSES, register_bank

from bialy.campaign.ancestry import PARTITIONS
from bialy.campaign.grading import wilson
from bialy.campaign.journal import CampaignFault, Journal
from bialy.campaign.lineage import Lineage
from bialy.campaign.teacher_qualification import TeacherQualification, policy


class Driver:
    """Teacher `strong` succeeds on every attempt, `weak` on the second repetition only."""
    def __init__(self):
        self.calls = []
    def execute(self, name, inputs, attempt):
        self.calls.append(inputs)
        success = inputs['actor']['id'] == 'strong' or inputs['repetition'] > 0
        return {'success': success, 'task': inputs['task']['id'], 'verdict': {'episode_sha256': name}}
    def validate(self, name, inputs, output):
        assert output['task'] == inputs['task']['id']


INPUTS = {'host': {'conservative_reference': {}, 'teacher_capture_qualified': {'strong': True, 'weak': True}}}


def bank(tmp_path):
    journal = Journal(tmp_path, {})
    lineage = Lineage(journal, FOCUSES)
    register_bank(lineage, [{'id': f'{partition}-{focus}-{index}', 'partition': partition, 'focus': focus, 'source_class': 'synthetic'}
                            for partition in PARTITIONS for focus in FOCUSES for index in range(2)])
    lineage.freeze()
    return journal, lineage


def measurement(focus, teacher='teacher', qualified=1, attempts=8, capture=True):
    return {'focus': focus, 'teacher_id': teacher, 'qualified': qualified, 'attempts': attempts, 'capture_qualified': capture}


def test_routes_rank_teachers_with_a_verified_success_by_their_floor():
    measurements = [measurement(focus, 'weak', 4, 8) for focus in FOCUSES] + [measurement(focus, 'strong', 8, 8) for focus in FOCUSES]
    result = policy(measurements, FOCUSES)
    for routes in result['focuses'].values():
        assert [route['teacher_id'] for route in routes] == ['strong', 'weak']
        assert routes[1]['floor'] == wilson(4, 8)[0]
    measurements = [measurement(focus) for focus in FOCUSES]
    measurements[0]['qualified'] = 0
    measurements[1]['capture_qualified'] = False
    with pytest.raises(CampaignFault, match='teachers_unqualified') as failure:
        policy(measurements, FOCUSES)
    assert failure.value.details['focuses'] == [FOCUSES[0], FOCUSES[1]]
    with pytest.raises(CampaignFault, match='duplicate_teacher_measurement'):
        policy([measurement(focus) for focus in FOCUSES] + [measurement(FOCUSES[0])], FOCUSES)
    with pytest.raises(CampaignFault, match='invalid_teacher_measurement'):
        policy([dict(measurement(FOCUSES[0]), attempts='8')], FOCUSES)


def test_the_policy_comes_from_development_tasks_and_is_written_once(tmp_path):
    journal, lineage = bank(tmp_path)
    driver = Driver()
    stage = TeacherQualification(journal, lineage, driver, [{'id': 'strong'}, {'id': 'weak'}], minimum_tasks=2)
    result = stage.execute(INPUTS, 'run')
    assert {call['task']['id'].split('-', 1)[0] for call in driver.calls} == {'development'}
    assert len(driver.calls) == 2 * len(FOCUSES) * 2 * 2
    written = json.loads((journal.directory / 'teacher-policy.json').read_text())
    assert written['focuses'] == result['focuses'] and written['lineage_sha256'] == lineage.freeze()
    stage.validate(INPUTS, result)
    assert stage.reconcile(INPUTS, 'run') == result and len(driver.calls) == 2 * len(FOCUSES) * 2 * 2
    (journal.directory / 'teacher-policy.json').write_text(json.dumps(dict(written, focuses={})))
    with pytest.raises(CampaignFault, match='teacher_policy_changed'):
        stage.validate(INPUTS, result)
    journal.close()


def test_the_policy_is_never_first_written_after_a_student_result(tmp_path):
    journal, lineage = bank(tmp_path)
    stage = journal.begin('evaluate_trained_e4b', {})
    journal.settle('evaluate_trained_e4b', stage['attempt'], 'passed', output={})
    with pytest.raises(CampaignFault, match='teacher_policy_after_student_results'):
        TeacherQualification(journal, lineage, Driver(), [{'id': 'strong'}], minimum_tasks=2).execute(INPUTS, 'run')
    journal.close()


def test_an_unfinished_episode_stops_qualification_typed(tmp_path):
    journal, lineage = bank(tmp_path)
    driver = Driver()
    def broken(name, inputs, attempt):
        driver.calls.append(inputs)
        raise CampaignFault('evidence_unavailable')
    driver.execute = broken
    stage = TeacherQualification(journal, lineage, driver, [{'id': 'strong'}], minimum_tasks=1, repetitions=1)
    with pytest.raises(CampaignFault, match='teacher_episode_incomplete') as failure:
        stage.execute(INPUTS, 'run')
    assert failure.value.details['fault']['code'] == 'evidence_unavailable' and len(driver.calls) == 1
    with pytest.raises(CampaignFault, match='teacher_development_bank_incomplete'):
        TeacherQualification(journal, lineage, Driver(), [{'id': 'strong'}], minimum_tasks=3).execute(INPUTS, 'run')
    with pytest.raises(CampaignFault, match='sampling_invalid'):
        TeacherQualification(journal, lineage, Driver(), [{'id': 'strong'}], minimum_tasks=0)
    journal.close()
