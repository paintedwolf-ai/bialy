"""Where a remote job or a rental check stands, from its directory and one provider listing."""
import json
import subprocess
import time
from pathlib import Path

from .progress import rental, seconds
from .reaper import provider
from .resources import ResourceError, Resources


def read(path, default):
    return json.loads(path.read_text()) if path.exists() else default


def status(directory, clock=time.time):
    """The stage, the machine, time and cost so far, the replicated checkpoints, and
    whether the account and the ledger hold anything. Nothing is changed."""
    directory, now = Path(directory), clock()
    configuration = json.loads((directory / 'configuration.json').read_text())
    outcome, entries = read(directory / 'outcome.json', {}), read(directory / 'stages.json', [])
    attempts = rental(read(directory / 'rental.json', {'attempts': []})['attempts'], now)
    replicated = directory / 'checkpoints'
    report = {'directory': str(directory), 'student': outcome.get('student'), 'schedule': outcome.get('schedule'),
              'status': outcome.get('status'), 'failure': outcome.get('failure'),
              'stage': entries[-1]['stage'] if entries and entries[-1]['ended_at'] is None else None,
              'stage_seconds': seconds(entries, now),
              'elapsed_seconds': round((entries[-1]['ended_at'] or now) - entries[0]['started_at'], 1) if entries else 0,
              'rental': attempts, 'cost': round(sum(attempt['cost'] or 0 for attempt in attempts), 4),
              'checkpoints': sorted(item.name for item in replicated.iterdir()) if replicated.is_dir() else [],
              'cleanup': read(directory / 'cleanup.json', {}).get('status')}
    provision = configuration.get('provisioning')
    if provision is None:
        return report
    account = provider(provision)
    report['ledger'] = Resources(provision['registry'], account).pending()
    report['ledger_empty'] = not report['ledger']
    try:
        labelled = account.labelled()
    except (ResourceError, ConnectionError, OSError, subprocess.TimeoutExpired) as exc:
        return {**report, 'account': None, 'account_empty': None, 'instance': None, 'provider_error': str(exc)}
    held = next((attempt['token'] for attempt in reversed(attempts) if attempt['ended_at'] is None), None)
    report['instance'] = next(iter(labelled.get(held, [])), None)
    report['account'] = [{'token': token, 'id': instance['id'], 'state': instance['state'], 'price_per_hour': instance['price_per_hour']}
                         for token, found in sorted(labelled.items()) for instance in found]
    report['account_empty'] = not report['account']
    return report
