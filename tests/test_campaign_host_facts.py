import copy

import pytest

from bialy.campaign.host_facts import members, receipt, student_controls
from bialy.campaign.journal import CampaignFault, digest


def inputs():
    task = {'project_path': '/work/project', 'allowed_write_roots': ['src'], 'required_checks': ['oracle', 'host-closeout']}
    invocation = {'id': 'call', 'tool': 'command', 'invoked': True, 'settled_at': 'time', 'status': 'error'}
    facts = {'version': 1, 'root_session_id': 'root', 'sessions': [{'id': 'root', 'parent_id': '', 'invocations': [invocation]}],
             'effects': [{'path': 'src/result.py', 'session_id': 'root'}],
             'execution': {'status': 'complete', 'submission_ids': ['initial'],
                           'closeout': {'model_authored': True, 'visible': True, 'has_tool_calls': False}}}
    oracle = {'source_sha256': 'source', 'task_spec_sha256': digest(task), 'grader_identity': 'grader',
              'checks': [{'id': 'oracle', 'passed': True}]}
    programs = {'call': {'state': 'failed', 'status_code': 127, 'settled_at': 'time', 'progress_state': 'committed',
                         'children': [{'state': 'completed', 'settled_at': 'time', 'effect_certainty': 'completed'}]}}
    return task, facts, oracle, programs


def test_settled_known_partial_effect_is_not_an_unresolved_effect_after_recovery():
    task, facts, oracle, programs = inputs()
    result = receipt(task, facts, oracle, 'store', 'epoch', 'teacher', 'grader', programs)
    assert result['unresolved_effects'] == 0
    programs['call']['children'][0]['effect_certainty'] = 'unknown'
    result = receipt(task, facts, oracle, 'store', 'epoch', 'teacher', 'grader', programs)
    assert result['unresolved_effects'] == 1


def test_extra_user_submission_and_out_of_scope_write_are_not_hidden_by_success():
    task, facts, oracle, programs = inputs()
    facts['execution']['submission_ids'].append('human-help')
    facts['effects'].append({'path': '/outside/changed', 'session_id': 'root'})
    result = receipt(task, facts, oracle, 'store', 'epoch', 'teacher', 'grader', programs)
    assert result['human_interventions'] == 1 and result['unauthorized_effects'] == 1


def test_unknown_worker_parent_and_missing_program_receipts_fail_closed():
    task, facts, oracle, programs = inputs()
    facts['sessions'].append({'id': 'worker', 'parent_id': 'missing'})
    with pytest.raises(CampaignFault, match='session_tree_invalid'):
        members(facts, 'root')
    facts['sessions'].pop()
    with pytest.raises(CampaignFault, match='program_evidence_missing'):
        receipt(task, facts, oracle, 'store', 'epoch', 'teacher', 'grader', {})


def test_student_controls_check_actual_dispatch_on_every_student_call():
    record = {'schema': 'pw-training-turn/1', 'provider': 'student', 'model': 'model',
              'student_provider': 'student', 'student_model': 'model', 'call_id': 'one',
              'dispatches': 1, 'settled': True, 'request': {'reasoning_effort': 'none'}}
    assert student_controls([record], 'student', 'model') is False
    wrong = copy.deepcopy(record)
    wrong['call_id'] = 'two'
    wrong['request']['reasoning_effort'] = 'high'
    with pytest.raises(CampaignFault, match='thinking_enabled_or_unknown'):
        student_controls([record, wrong], 'student', 'model')
    wrong['provider'] = 'teacher'
    with pytest.raises(CampaignFault, match='call_capture_missing'):
        student_controls([wrong], 'student', 'model')


def test_program_status_is_preserved_for_review_and_unsettled_children_block_admission():
    task, facts, oracle, programs = inputs()
    result = receipt(task, facts, oracle, 'store', 'epoch', 'teacher', 'grader', programs)
    assert result['programs']['call']['status_code'] == 127
    programs['call']['children'][0]['state'] = 'running'
    result = receipt(task, facts, oracle, 'store', 'epoch', 'teacher', 'grader', programs)
    assert result['unresolved_effects'] == 1
