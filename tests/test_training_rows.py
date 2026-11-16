import pytest
from host_bundles import operation, write_episode

from bialy import training_rows
from bialy.host_bundle import Episode

TASK = {'diagnostic': {'expectation': 'reproduced_failure', 'witness': 'diagnostic'}}
WITNESSED = {'checks': [{'id': 'grader', 'passed': True}, {'id': 'diagnostic', 'passed': True}]}
UNWITNESSED = {'checks': [{'id': 'grader', 'passed': True}, {'id': 'diagnostic', 'passed': False}]}
GROUNDED = {'traced': True, 'cited_evidence': [{'ref': 'op'}]}


def call(**changes):
    facts = {'call_id': 'c1', 'tool': 'read', 'offered': True, 'recorded': True, 'operation_ids': ['op'], 'invoked': True,
             'settlement': 'settled', 'semantic_outcome': 'success', 'effect_state': 'known', 'failure_code': '',
             'source_verdict': '', 'source_revision': '', 'replays_unknown_effects': False}
    return {**facts, **changes}


def candidate(message='m1', settled='2026-11-12T10:00:00Z', calls=None, committed=False, session='root'):
    target = {'role': 'assistant', 'text': '' if calls else 'Done.', 'tool_calls': [{'id': c['call_id'], 'name': c['tool'], 'arguments': {}} for c in calls or []]}
    return {'schema': 'pw-training-candidate/1', 'session': session, 'message_id': message, 'call_id': 'request-' + message,
            'prompt': [{'role': 'system', 'content': 'host'}], 'target': target,
            'request_capture': {'sha256': 'a' * 64, 'tools': []}, 'response_sha256': 'b' * 64,
            'execution': {'receipt_id': 'out-' + message, 'settled_at': settled, 'calls': calls or [], 'committed': committed},
            'source': {'bundle_sha256': 'c' * 64, 'capture_sha256': 'd' * 64}}


def episode(tmp_path, operations=(), grounding=None, name='bundle'):
    messages = [{'id': message, 'role': 'assistant', 'grounding': value} for message, value in (grounding or {}).items()]
    return Episode(write_episode(tmp_path / name, sessions=[{'id': 'root', 'messages': messages}], operations=operations))


def admission(tmp_path, candidates, operations=(), grounding=None, oracle=UNWITNESSED, task=TASK):
    return [row['admission'] for row in training_rows.admit(candidates, episode(tmp_path, operations, grounding), task, oracle)]


def test_a_successful_step_is_ordinary_and_keeps_only_its_own_facts(tmp_path):
    item = candidate(calls=[call()])
    [row] = training_rows.admit([item], episode(tmp_path), TASK, UNWITNESSED)
    assert row['admission'] == {'eligibility': 'eligible', 'category': 'ordinary', 'semantic_outcome': 'success'}
    assert set(row) == set(item) | {'admission'} and row['execution'] == item['execution']
    assert training_rows.is_eligible(row) and not training_rows.training_ready(row)
    assert training_rows.training_ready({**row, 'quality': {'verdict': 'pass'}})


@pytest.mark.parametrize('change,reason', [
    ({'offered': False}, 'unoffered_call'), ({'recorded': False}, 'unrecorded_call'),
    ({'settlement': 'unsettled'}, 'unsettled_operation'), ({'effect_state': 'unknown'}, 'unknown_effects'),
    ({'replays_unknown_effects': True}, 'replay_after_unknown_effects'),
    ({'invoked': False, 'semantic_outcome': 'failure', 'effect_state': 'none'}, 'authorization_refusal'),
    ({'semantic_outcome': 'unknown'}, 'unknown_outcome')])
def test_each_call_level_exclusion_keeps_its_reason(tmp_path, change, reason):
    [result] = admission(tmp_path, [candidate(calls=[call(**change)])])
    assert result['eligibility'] == 'excluded' and result['reason'] == reason


def test_a_host_excluded_capture_is_recorded_with_the_hosts_reason(tmp_path):
    stub = {'schema': 'pw-training-candidate/1', 'session': 'root', 'message_id': 'm1', 'call_id': 'r',
            'host_exclusion': 'host_modified_response', 'source': {'bundle_sha256': 'c' * 64, 'capture_sha256': 'd' * 64}}
    [row] = training_rows.admit([stub], episode(tmp_path), TASK, UNWITNESSED)
    assert row['admission'] == {'eligibility': 'excluded', 'reason': 'host_excluded', 'semantic_outcome': None,
                                'detail': 'host_modified_response'}
    assert 'target' not in row


def test_a_diagnostic_failure_needs_the_tasks_witness_and_the_original_source(tmp_path):
    failing = call(semantic_outcome='failure', source_verdict='failed', source_revision='r0')
    operations = [dict(operation('op', call='c1'), source_revision='r0', source_verdict='failed'),
                  dict(operation('edit', call='c2'), source_revision='r1', started_at='2026-11-12T10:00:05Z')]
    [unwitnessed] = admission(tmp_path, [candidate(calls=[failing])], operations, oracle=UNWITNESSED)
    assert unwitnessed == {'eligibility': 'excluded', 'reason': 'failure_without_witness', 'semantic_outcome': 'failure'}
    [diagnostic] = admission(tmp_path / 'w', [candidate(calls=[failing])], operations, oracle=WITNESSED)
    assert diagnostic == {'eligibility': 'eligible', 'category': 'diagnostic', 'semantic_outcome': 'failure', 'witness': 'diagnostic'}
    later = call(semantic_outcome='failure', source_verdict='failed', source_revision='r1')
    [elsewhere] = admission(tmp_path / 'l', [candidate(calls=[later])], operations, oracle=WITNESSED)
    assert elsewhere['reason'] == 'failure_without_witness'
    [undeclared] = admission(tmp_path / 'u', [candidate(calls=[failing])], operations, oracle=WITNESSED, task={})
    assert undeclared['reason'] == 'failure_without_witness'


def test_a_success_after_a_failure_is_a_recovery_that_does_not_change_the_failure(tmp_path):
    failed = candidate('m1', '2026-11-12T10:00:00Z', [call(semantic_outcome='failure')])
    fixed = candidate('m2', '2026-11-12T10:01:00Z', [call(call_id='c2')])
    other = candidate('m3', '2026-11-12T10:02:00Z', [call(call_id='c3')], session='worker')
    results = admission(tmp_path, [fixed, other, failed])
    assert [result.get('category') or result['reason'] for result in results] == ['failure_without_witness', 'recovery', 'ordinary']
    assert results[0]['semantic_outcome'] == 'failure'
    [alone] = admission(tmp_path / 'alone', [failed])
    assert alone == results[0]


@pytest.mark.parametrize('grounding,result', [
    (GROUNDED, 'closeout'), ({'evidence_optional': True}, 'closeout'),
    ({'traced': True, 'host_assembled': True, 'cited_evidence': [{'ref': 'op'}]}, 'host_replaced_closeout'),
    ({'traced': True}, 'uncited_closeout'), (None, 'host_replaced_closeout')])
def test_a_committed_closeout_is_admitted_only_with_its_own_grounding(tmp_path, grounding, result):
    [found] = admission(tmp_path, [candidate(committed=True)], grounding={'m1': grounding})
    assert (found.get('category') or found['reason']) == result


def test_an_uncommitted_reply_and_a_stale_verification_are_excluded(tmp_path):
    [uncommitted] = admission(tmp_path, [candidate(committed=False)], grounding={'m1': GROUNDED})
    assert uncommitted['reason'] == 'uncommitted_reply'
    verified = dict(operation('check', call='c1'), source_revision='r1', source_verdict='passed', started_at='2026-11-12T09:59:00Z')
    edited = dict(operation('edit', call='c2'), source_revision='r2', started_at='2026-11-12T09:59:30Z')
    [stale] = admission(tmp_path / 's', [candidate(committed=True)], [verified, edited], grounding={'m1': GROUNDED})
    assert stale['reason'] == 'stale_verification'
    [fresh] = admission(tmp_path / 'f', [candidate(committed=True)], [edited, dict(verified, source_revision='r2',
                        started_at='2026-11-12T09:59:45Z')], grounding={'m1': GROUNDED})
    assert fresh['category'] == 'closeout'
