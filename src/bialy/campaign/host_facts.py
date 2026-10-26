"""Derive episode checks from typed, closed host evidence."""
import json
from pathlib import PurePosixPath

from .journal import CampaignFault, digest


def student_controls(captures, provider, model):
    calls = [record for record in captures if record.get('schema') == 'pw-training-turn/1'
             and record.get('provider') == provider and record.get('model') == model
             and record.get('student_provider') == provider and record.get('student_model') == model]
    if not calls or len({record['call_id'] for record in calls}) != len(calls):
        raise CampaignFault('student_call_capture_missing_or_duplicate')
    for record in calls:
        request = record.get('request') or {}
        if (record.get('exclusion_reason') or record.get('dispatches') != 1 or not record.get('settled')
                or request.get('reasoning_effort') != 'none'):
            raise CampaignFault('student_thinking_enabled_or_unknown')
    return False


def members(facts, root):
    if facts.get('version') != 1 or facts.get('root_session_id') != root:
        raise CampaignFault('episode_host_identity_mismatch')
    sessions = facts.get('sessions')
    if not isinstance(sessions, list) or not sessions:
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


def receipt(task, facts, oracle, store_sha256, epoch, teacher_family, grader_identity, programs, tool_surface='command-v1'):
    if tool_surface not in {'legacy', 'command-v1'}:
        raise CampaignFault('host_tool_surface_unknown')
    root = facts.get('root_session_id')
    sessions = members(facts, root)
    execution = facts.get('execution') or {}
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
    for effect in facts.get('effects', []):
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
    unresolved = 0
    legacy_commands = {}
    invocation_ids = set()
    for session in sessions.values():
        for invocation in session.get('invocations', []):
            invocation_ids.add(invocation['id'])
            if invocation['invoked'] and (not invocation.get('settled_at') or invocation['status'] not in {'completed', 'error', 'rejected', 'interrupted'}):
                unresolved += 1
            if invocation['tool'] == 'command' and invocation['invoked']:
                if tool_surface == 'legacy':
                    verdict = invocation.get('source_verdict')
                    legacy_commands[invocation['id']] = verdict
                    unresolved += int(verdict not in {'passed', 'failed'})
                    continue
                program = programs.get(invocation['id'])
                if program is None:
                    raise CampaignFault('command_program_evidence_missing')
                children = program['children']
                if any(child['effect_certainty'] not in {'none', 'completed', 'attempted', 'unknown'} for child in children):
                    raise CampaignFault('command_effect_evidence_invalid')
                unresolved += int(
                    program['state'] not in {'completed', 'rejected', 'failed', 'interrupted'} or not program.get('settled_at')
                    or program['progress_state'] == 'unknown'
                    or any(child['state'] not in {'completed', 'refused', 'failed', 'interrupted'}
                           or not child.get('settled_at') or child['effect_certainty'] in {'attempted', 'unknown'} for child in children))
    return {'task_spec_sha256': digest(task), 'grader_identity': grader_identity, 'complete': True,
            'epoch': epoch, 'root_session': root, 'teacher_family': teacher_family,
            'store_sha256': store_sha256, 'final_source_sha256': source, 'checks': checks,
            'human_interventions': interventions, 'unauthorized_effects': unauthorized, 'unresolved_effects': unresolved,
            'host_evidence_sha256': digest(facts), 'programs_sha256': digest(programs), 'programs': programs,
            'tool_surface': tool_surface, 'legacy_commands': legacy_commands,
            'evidence_ids': sorted(invocation_ids | {check['id'] for check in checks})}


def program_evidence(database, facts):
    identifiers = [item['id'] for session in facts['sessions'] for item in session.get('invocations', [])
                   if item['tool'] == 'command' and item['invoked']]
    programs = {}
    for identifier in identifiers:
        row = database.execute('SELECT id,state,status_code,settled_at,progress_state FROM invocation_programs WHERE parent_receipt_id=?', (identifier,)).fetchone()
        if row is None:
            raise CampaignFault('command_program_evidence_missing')
        children = [dict(item) for item in database.execute(
            'SELECT id,state,status_code,effect_certainty,settled_at FROM invocation_program_executions WHERE program_id=? ORDER BY ordinal', (row['id'],))]
        programs[identifier] = {**dict(row), 'children': children}
    return json.loads(json.dumps(programs))
