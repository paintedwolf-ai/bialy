import pytest
from host_bundles import operation, write_episode

from bialy.campaign.host_facts import members, receipt, student_controls
from bialy.campaign.journal import CampaignFault, digest
from bialy.host_bundle import Episode

EXECUTION = {'kind': 'execution', 'value': {'status': 'complete', 'submission_ids': ['initial'],
                                            'closeout': {'model_authored': True, 'visible': True, 'has_tool_calls': False}}}


def inputs():
    task = {'project_path': '/work/project', 'allowed_write_roots': ['src'], 'required_checks': ['oracle', 'host-closeout']}
    oracle = {'source_sha256': 'source', 'task_spec_sha256': digest(task), 'grader_identity': 'grader',
              'checks': [{'id': 'oracle', 'passed': True}]}
    # A failed program whose settled child changed a file: a known partial effect.
    operations = [operation('call', call='c1', tool='command', outcome='failure', effects='known'),
                  operation('call-op0', parent='call', tool='tee', effects='known')]
    return task, oracle, operations


def episode(tmp_path, operations, observations=(EXECUTION,), effects=({'path': 'src/result.py', 'session_id': 'root', 'op': 'write'},),
            sessions=None, name='bundle'):
    return Episode(write_episode(tmp_path / name, sessions=sessions, operations=operations, observations=observations, effects=effects))


def test_a_settled_known_partial_effect_is_not_unresolved(tmp_path):
    task, oracle, operations = inputs()
    result = receipt(task, episode(tmp_path, operations), oracle, 'store', 'epoch', 'teacher', 'grader')
    assert result['unresolved_effects'] == 0 and result['evidence_ids'] == ['call', 'host-closeout', 'oracle']
    operations[1]['effect_state'] = 'unknown'
    result = receipt(task, episode(tmp_path, operations, name='unknown'), oracle, 'store', 'epoch', 'teacher', 'grader')
    assert result['unresolved_effects'] == 1


@pytest.mark.parametrize('change,unresolved', [
    ({}, 0), ({'settlement': 'unsettled'}, 1), ({'effect_state': 'unknown'}, 1), ({'invoked': False, 'effect_state': 'unknown'}, 0)])
def test_unresolved_work_is_an_invoked_operation_the_host_could_not_settle(tmp_path, change, unresolved):
    task, oracle, _ = inputs()
    result = receipt(task, episode(tmp_path, [{**operation('call', call='c1'), **change}]), oracle, 'store', 'epoch', 'teacher', 'grader')
    assert result['unresolved_effects'] == unresolved


def test_extra_user_submission_and_out_of_scope_write_are_not_hidden_by_success(tmp_path):
    task, oracle, operations = inputs()
    execution = {'kind': 'execution', 'value': {**EXECUTION['value'], 'submission_ids': ['initial', 'human-help']}}
    effects = [{'path': 'src/result.py', 'session_id': 'root', 'op': 'write'}, {'path': '/outside/changed', 'session_id': 'root', 'op': 'write'},
               {'path': 'src/other.py', 'session_id': 'stranger', 'op': 'write'}]
    result = receipt(task, episode(tmp_path, operations, (execution,), effects), oracle, 'store', 'epoch', 'teacher', 'grader')
    assert result['human_interventions'] == 2 and result['unauthorized_effects'] == 1
    assert result['host_bundle_sha256'] and result['contract_sha256']


def test_missing_submissions_oracle_binding_and_closeout_are_visible(tmp_path):
    task, oracle, operations = inputs()
    with pytest.raises(CampaignFault, match='submission_evidence_missing'):
        receipt(task, episode(tmp_path, operations, observations=()), oracle, 'store', 'epoch', 'teacher', 'grader')
    with pytest.raises(CampaignFault, match='oracle_binding_mismatch'):
        receipt(task, episode(tmp_path, operations, name='oracle'), oracle, 'store', 'epoch', 'teacher', 'other')
    unfinished = {'kind': 'execution', 'value': {**EXECUTION['value'], 'status': 'failed'}}
    result = receipt(task, episode(tmp_path, operations, (unfinished,), name='unfinished'), oracle, 'store', 'epoch', 'teacher', 'grader')
    assert [check['passed'] for check in result['checks'] if check['id'] == 'host-closeout'] == [False]


def test_the_session_tree_must_be_rooted_and_closed(tmp_path):
    good = episode(tmp_path, [], sessions=[{'id': 'root'}, {'id': 'worker', 'parent_id': 'root'}])
    assert set(members(good, 'root')) == {'root', 'worker'}
    with pytest.raises(CampaignFault, match='host_identity_mismatch'):
        members(good, 'other')
    broken = episode(tmp_path, [], sessions=[{'id': 'root'}, {'id': 'worker', 'parent_id': 'missing'}], name='broken')
    with pytest.raises(CampaignFault, match='session_tree_invalid'):
        members(broken, 'root')
    duplicate = episode(tmp_path, [], sessions=[{'id': 'root'}, {'id': 'root'}], name='duplicate')
    with pytest.raises(CampaignFault, match='sessions_invalid'):
        members(duplicate, 'root')


def test_student_controls_check_actual_dispatch_on_every_student_call(tmp_path):
    record = {'schema': 'pw-training-turn/1', 'provider': 'student', 'model': 'model', 'student_provider': 'student',
              'student_model': 'model', 'call_id': 'one', 'dispatches': 1, 'settled': True, 'request': {'reasoning_effort': 'none'}}
    single = Episode(write_episode(tmp_path / 'one', captures=[record]))
    assert student_controls(single, 'student', 'model') is False
    wrong = dict(record, call_id='two', request={'reasoning_effort': 'high'})
    both = Episode(write_episode(tmp_path / 'two', captures=[record, wrong]))
    with pytest.raises(CampaignFault, match='thinking_enabled_or_unknown'):
        student_controls(both, 'student', 'model')
    with pytest.raises(CampaignFault, match='capture_missing_or_duplicate'):
        student_controls(single, 'teacher', 'model')
