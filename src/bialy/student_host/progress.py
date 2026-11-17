"""The stages a remote job has been through, with the wall time of each."""
import json
import sys
import time
from pathlib import Path

from .resources import write


class Stages:
    """`stages.json` in the job directory: one entry each time the job enters a stage. Each
    transition is also a line on the controller's log."""

    def __init__(self, directory, clock=time.time, log=None):
        self.directory, self.clock = Path(directory), clock
        self.log = log or (lambda line: print(line, file=sys.stderr, flush=True))

    def entries(self):
        path = self.directory / 'stages.json'
        return json.loads(path.read_text()) if path.exists() else []

    def enter(self, name):
        entries, now = self.entries(), self.clock()
        if entries and entries[-1]['ended_at'] is None:
            if entries[-1]['stage'] == name:
                return
            entries[-1]['ended_at'] = now
        entries.append({'stage': name, 'started_at': now, 'ended_at': None})
        write(self.directory / 'stages.json', entries)
        self.log('%s %s stage=%s' % (time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(now)), self.directory.name, name))

    def close(self):
        entries = self.entries()
        if entries and entries[-1]['ended_at'] is None:
            entries[-1]['ended_at'] = self.clock()
            write(self.directory / 'stages.json', entries)


def seconds(entries, now):
    """Wall seconds spent in each stage, in the order first entered; an open stage runs to `now`."""
    totals = {}
    for entry in entries:
        totals[entry['stage']] = totals.get(entry['stage'], 0) + (entry['ended_at'] or now) - entry['started_at']
    return {name: round(value, 1) for name, value in totals.items()}


def rental(attempts, now):
    """Each leased machine with the hours it was held and what that cost at its hourly price."""
    out = []
    for attempt in attempts:
        hours = ((attempt['ended_at'] or now) - attempt['started_at']) / 3600 if attempt['instance'] else 0
        price = attempt['price_per_hour']
        out.append({**attempt, 'hours': round(hours, 4), 'cost': None if price is None else round(hours * price, 4)})
    return out
