import copy

import pytest

from bialy.campaign.evaluation import Evaluation, EvaluationPlan
from bialy.campaign.grading import DIMENSIONS
from bialy.campaign.journal import CampaignFault, Journal
from bialy.campaign.lineage import FAMILIES, PARTITIONS, Lineage


class Driver:
    def __init__(self):
        self.calls = []
    def execute(self, name, inputs, attempt):
        self.calls.append(inputs)
        if inputs['repetition'] == 0:
            raise CampaignFault('fixture_failure')
        return {'success': True, 'human_interventions': 0, 'unauthorized_effects': 0,
                'unsupported_completion_claims': 0, 'effective_thinking': False, 'exit_status': 0,
                'verdict': {'deterministic': True, 'episode_sha256': inputs['task']['id'] + str(inputs['repetition'])},
                'reviews': [{'dimensions': dict.fromkeys(DIMENSIONS, 'pass')}]}
    def validate(self, name, inputs, output):
        assert output['effective_thinking'] is False


def test_every_planned_attempt_is_retained_and_resume_does_not_replay(tmp_path):
    journal = Journal(tmp_path, {})
    lineage = Lineage(journal)
    lineage.roots([{'id': f'{partition}-{family}-{index}', 'partition': partition,
                    'family': family, 'source_class': 'synthetic'}
                   for partition in PARTITIONS for family in FAMILIES for index in range(3)])
    lineage.freeze()
    planner = EvaluationPlan(journal, lineage, repetitions=2)
    plan = planner.execute({'measurements': {'accepted': 32}}, 'plan')
    assert plan['planned_attempts_per_arm'] == 48
    driver = Driver()
    evaluation = Evaluation(journal, lineage, driver, 'e4b', 'base', {'kind': 'student'}, sleep=lambda _: None)
    inputs = {'plan': plan}
    result = evaluation.execute(inputs, 'evaluation')
    evaluation.validate(inputs, result)
    assert len(result['attempts']) == 48
    assert sum(row['success'] for row in result['attempts']) == 24
    assert evaluation.reconcile(inputs, 'evaluation') == result
    assert len(driver.calls) == 48
    assert all(call['actor']['kind'] == 'student' for call in driver.calls)
    changed = copy.deepcopy(result)
    changed['attempts'][0]['success'] = True
    with pytest.raises(CampaignFault, match='evaluation_failure_changed'):
        evaluation.validate(inputs, changed)
    omitted = {**result, 'attempts': result['attempts'][1:]}
    with pytest.raises(CampaignFault, match='evaluation_attempts_changed'):
        evaluation.validate(inputs, omitted)
    journal.close()
