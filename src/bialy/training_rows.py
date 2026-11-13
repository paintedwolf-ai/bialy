"""Training-row admission: the one sequence from host candidates to `pw-training-row/1`.

A candidate states what the host recorded for one turn. Admission decides whether the
turn may supervise a student and as what: an ordinary step, a diagnostic failure the task
asked for, a recovery after an earlier failure, or a closeout. Every exclusion keeps its
reason, and the turn's semantic outcome is recorded as the host found it, never rewritten
to admit a row. A row carries only what was known when its turn was taken; later turns
and the final grade never enter it, except the diagnostic witness the task declared
before the episode ran.
"""
from datetime import datetime

SCHEMA = 'pw-training-row/1'
CATEGORIES = ('ordinary', 'diagnostic', 'recovery', 'closeout')
REASONS = ('host_excluded', 'unoffered_call', 'unrecorded_call', 'unsettled_operation', 'unknown_effects',
           'replay_after_unknown_effects', 'authorization_refusal', 'failure_without_witness', 'unknown_outcome',
           'uncommitted_reply', 'host_replaced_closeout', 'uncited_closeout', 'stale_verification')


def instant(value):
    head, _, rest = value.rstrip('Z').partition('.')
    return datetime.fromisoformat(head + '.' + (rest + '000000')[:6] + '+00:00')


def turn_outcome(calls):
    outcomes = {call.get('semantic_outcome', 'unknown') for call in calls}
    return 'failure' if 'failure' in outcomes else 'unknown' if 'unknown' in outcomes else 'success'


def call_reason(call):
    """Why one call's host record cannot supervise a student, or None."""
    if not call['offered']:
        return 'unoffered_call'
    if not call['recorded']:
        return 'unrecorded_call'
    if call['settlement'] != 'settled':
        return 'unsettled_operation'
    if call['effect_state'] == 'unknown':
        return 'unknown_effects'
    if call['replays_unknown_effects']:
        return 'replay_after_unknown_effects'
    if not call['invoked'] and call['semantic_outcome'] == 'failure':
        return 'authorization_refusal'
    return None


def closeout_reason(grounding):
    """A tool-free reply's grounding as the host traced it."""
    if not isinstance(grounding, dict) or grounding.get('host_assembled'):
        return 'host_replaced_closeout'
    if grounding.get('evidence_optional'):
        return None
    if grounding.get('traced') and (grounding.get('cited_evidence') or grounding.get('cited_urls')):
        return None
    return 'uncited_closeout'


class Episode:
    """The host facts admission reads from one episode, up to any moment in it."""

    def __init__(self, episode):
        self.operations = sorted(episode.operations, key=lambda item: instant(item['started_at']))
        self.grounding = {message.get('id'): message.get('grounding') for session in episode.sessions
                          for message in episode.messages(session)}
        revisions = [item['source_revision'] for item in self.operations if item.get('source_revision')]
        self.initial_revision = revisions[0] if revisions else None

    def before(self, moment):
        return [item for item in self.operations if instant(item['started_at']) <= moment]

    def stale_verification(self, moment):
        """Whether verification passed earlier but not on the source as it stood at `moment`."""
        known = self.before(moment)
        revisions = [item['source_revision'] for item in known if item.get('source_revision')]
        passed = {item['source_revision'] for item in known if item.get('source_verdict') == 'passed'}
        return bool(passed) and bool(revisions) and revisions[-1] not in passed

    def reproduced(self, calls):
        """Whether a call failed verification of the original source."""
        return any(call.get('source_verdict') == 'failed' and call.get('source_revision') == self.initial_revision
                   for call in calls) and self.initial_revision is not None


def witnessed(task, oracle):
    expectation = task.get('diagnostic')
    return bool(expectation) and any(check['id'] == expectation['witness'] and check.get('passed') is True
                                     for check in oracle['checks'])


def admit(candidates, episode, task, oracle):
    """Admission records for one episode's candidates, in the order their turns settled."""
    facts = Episode(episode)
    witness = witnessed(task, oracle)
    built = [item for item in candidates if not item.get('host_exclusion')]
    order = sorted(built, key=lambda item: instant(item['execution']['settled_at']))
    failed, records = set(), []
    for item in candidates:
        if item.get('host_exclusion'):
            records.append(excluded(item, 'host_excluded', None, detail=item['host_exclusion']))
    for item in order:
        calls, session = item['execution']['calls'], item['session']
        outcome = turn_outcome(calls)
        moment = instant(item['execution']['settled_at'])
        if calls:
            reason = next((found for found in map(call_reason, calls) if found), None)
            if reason is None and outcome == 'failure':
                category = 'diagnostic' if witness and facts.reproduced(calls) else None
                reason = None if category else 'failure_without_witness'
            elif reason is None and outcome == 'unknown':
                category, reason = None, 'unknown_outcome'
            else:
                category = 'recovery' if session in failed else 'ordinary'
        else:
            category = 'closeout'
            reason = ('uncommitted_reply' if not item['execution']['committed']
                      else closeout_reason(facts.grounding.get(item['message_id']))
                      or ('stale_verification' if facts.stale_verification(moment) else None))
        if outcome == 'failure':
            failed.add(session)
        records.append(excluded(item, reason, outcome) if reason else eligible(item, category, outcome, witness, task))
    return records


def eligible(item, category, outcome, witness, task):
    admission = {'eligibility': 'eligible', 'category': category, 'semantic_outcome': outcome}
    if category == 'diagnostic':
        admission['witness'] = task['diagnostic']['witness']
    return {**item, 'schema': SCHEMA, 'admission': admission}


def excluded(item, reason, outcome, detail=None):
    admission = {'eligibility': 'excluded', 'reason': reason, 'semantic_outcome': outcome}
    if detail:
        admission['detail'] = detail
    return {'schema': SCHEMA, 'session': item['session'], 'message_id': item['message_id'],
            'call_id': item.get('call_id', ''), 'source': item['source'], 'admission': admission}


def is_eligible(row):
    return row.get('schema') == SCHEMA and row.get('admission', {}).get('eligibility') == 'eligible'


def training_ready(row):
    """An eligible row whose review passed: what a dataset may hold."""
    return is_eligible(row) and (row.get('quality') or {}).get('verdict') == 'pass'
