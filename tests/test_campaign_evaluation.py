import copy

import pytest
from conftest import FOCUSES, register_bank

from bialy.campaign import evaluation_policy
from bialy.campaign.ancestry import PARTITIONS
from bialy.campaign.evaluation import Evaluation, EvaluationPlan
from bialy.campaign.grading import DIMENSIONS
from bialy.campaign.journal import CampaignFault, Journal
from bialy.campaign.lineage import Lineage


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
    lineage = Lineage(journal, FOCUSES)
    register_bank(lineage, [{'id': f'{partition}-{family}-{index}', 'partition': partition,
                    'focus': family, 'source_class': 'synthetic'}
                   for partition in PARTITIONS for family in FOCUSES for index in range(3)])
    lineage.freeze()
    policy = evaluation_policy.load()
    policy = {**policy, 'design': {**policy['design'], 'attempts_per_task': 2}}
    planner = EvaluationPlan(journal, lineage, policy)
    driver = Driver()
    evaluation = Evaluation(journal, lineage, driver, 'e4b', 'base', {'kind': 'student'}, sleep=lambda _: None)
    unplanned = {'tasks': [], 'lineage_sha256': lineage.freeze(), 'policy_sha256': 'none'}
    with pytest.raises(CampaignFault, match='evaluation_policy_unfrozen'):
        evaluation.execute({'plan': unplanned}, 'early')
    plan = planner.execute({'measurements': {'accepted': 32}, 'selector': {'qualified': True, 'mode': 'explicit_tool_loading'}}, 'plan')
    planner.validate({'measurements': {'accepted': 32}, 'selector': {'qualified': True, 'mode': 'explicit_tool_loading'}}, plan)
    assert plan['selector'] == {'policy': None, 'fallback_rate': 0}
    assert {task['partition'] for task in plan['tasks']} == {'system_test', 'development'}
    assert plan['planned_attempts_per_arm'] == 96
    inputs = {'plan': plan}
    result = evaluation.execute(inputs, 'evaluation')
    evaluation.validate(inputs, result)
    assert len(result['attempts']) == 96
    assert sum(row['success'] for row in result['attempts']) == 48
    assert evaluation.reconcile(inputs, 'evaluation') == result
    assert len(driver.calls) == 96
    assert all(call['actor']['kind'] == 'student' for call in driver.calls)
    changed = copy.deepcopy(result)
    changed['attempts'][0]['success'] = True
    with pytest.raises(CampaignFault, match='evaluation_failure_changed'):
        evaluation.validate(inputs, changed)
    omitted = {**result, 'attempts': result['attempts'][1:]}
    with pytest.raises(CampaignFault, match='evaluation_attempts_changed'):
        evaluation.validate(inputs, omitted)
    journal.close()
