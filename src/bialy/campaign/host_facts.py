"""Derive episode checks from one verified host episode bundle."""
from pathlib import PurePosixPath

from .journal import CampaignFault, digest


def student_controls(episode, provider, model):
    calls = [record for record in episode.captures if record.get('schema') == 'pw-training-turn/1'
             and record.get('provider') == provider and record.get('model') == model
             and record.get('student_provider') == provider and record.get('student_model') == model]
    if not calls or len({record['call_id'] for record in calls}) != len(calls):
        raise CampaignFault('student_call_capture_missing_or_duplicate')
    for record in calls:
        request = episode.capture_request(record) or {}
        if (record.get('exclusion_reason') or record.get('dispatches') != 1 or not record.get('settled')
                or request.get('reasoning_effort') != 'none'):
            raise CampaignFault('student_thinking_enabled_or_unknown')
    return False


def members(episode, root):
    if episode.root_session != root:
        raise CampaignFault('episode_host_identity_mismatch')
    sessions = episode.sessions
    if not sessions:
        raise CampaignFault('episode_sessions_missing')
    by_id = {session['id']: session for session in sessions}
    if len(by_id) != len(sessions) or root not in by_id:
        raise CampaignFault('episode_sessions_invalid')
    for session in sessions:
        visited, current = set(), session
        while current['id'] != root:
            if current['id'] in visited or current.get('parent_id') not in by_id:
                raise CampaignFault('episode_session_tree_invalid')
            visited.add(current['id'])
            current = by_id[current['parent_id']]
    return by_id


def unresolved(operation, children):
    """An invoked operation whose work or effects the host could not settle."""
    family = [operation, *children]
    return operation['invoked'] and any(item['settlement'] != 'settled' or item['effect_state'] == 'unknown' for item in family)


def receipt(task, episode, oracle, store_sha256, epoch, teacher_family, grader_identity):
    root = episode.root_session
    sessions = members(episode, root)
    execution = episode.execution or {}
    final = execution.get('closeout') or {}
    source = oracle.get('source_sha256')
    if (not source or oracle.get('task_spec_sha256') != digest(task)
            or oracle.get('grader_identity') != grader_identity):
        raise CampaignFault('episode_oracle_binding_mismatch')
    checks = [{'id': check['id'], 'passed': check.get('passed') is True, 'source_sha256': source}
              for check in oracle['checks']]
    checks.append({'id': 'host-closeout', 'source_sha256': source,
                   'passed': execution.get('status') == 'complete' and final.get('model_authored') is True
                             and final.get('visible') is True and not final.get('has_tool_calls')})
    expected_submissions = 1 + len(task.get('follow_ups', []))
    submissions = execution.get('submission_ids')
    if not isinstance(submissions, list):
        raise CampaignFault('episode_submission_evidence_missing')
    interventions = abs(len(submissions) - expected_submissions)
    project = PurePosixPath(task['project_path'])
    allowed = [PurePosixPath(path) for path in task['allowed_write_roots']]
    unauthorized = 0
    for effect in episode.effects:
        path = PurePosixPath(effect['path'])
        if path.is_absolute():
            try:
                path = path.relative_to(project)
            except ValueError:
                unauthorized += 1
                continue
        if '..' in path.parts or not any(path == base or base in path.parents for base in allowed):
            unauthorized += 1
        if effect['session_id'] not in sessions:
            interventions += 1
    calls = episode.calls()
    return {'task_spec_sha256': digest(task), 'grader_identity': grader_identity, 'complete': True,
            'epoch': epoch, 'root_session': root, 'teacher_family': teacher_family,
            'store_sha256': store_sha256, 'final_source_sha256': source, 'checks': checks,
            'human_interventions': interventions, 'unauthorized_effects': unauthorized,
            'unresolved_effects': sum(unresolved(operation, children) for operation, children in calls),
            'host_bundle_sha256': episode.manifest_sha256, 'contract_sha256': episode.contract_sha256,
            'evidence_ids': sorted({operation['id'] for operation, _ in calls} | {check['id'] for check in checks})}
