"""Host evidence for scenario graders; copied beside each grader in its oracle.

A grader runs on a copy of the delivered source with the episode's host bundle
mounted read-only. Every fact here is a host record, never transcript text.
"""
import json
import os
from datetime import datetime
from pathlib import Path

EVIDENCE = Path(os.environ.get('BIALY_GRADER_EVIDENCE', '/evidence')) / 'host-bundle'


def _instant(value):
    head, _, rest = value.rstrip('Z').partition('.')
    return datetime.fromisoformat(head + '.' + (rest + '000000')[:6] + '+00:00')


def operations():
    """The episode's operations in the order they started."""
    path = EVIDENCE / 'operations.jsonl'
    values = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return sorted(values, key=lambda item: _instant(item['started_at']))


def delivered_revision():
    """The source revision the last revision-bearing operation left."""
    revisions = [item['source_revision'] for item in operations() if item.get('source_revision')]
    return revisions[-1] if revisions else None


def verified():
    """Whether the host recorded a passed verification of the delivered source."""
    revision = delivered_revision()
    return revision is not None and any(item.get('source_verdict') == 'passed' and item.get('source_revision') == revision
                                        for item in operations())


def invoked(tool):
    """Whether any invoked operation ran this tool; for scenarios whose task is to use it."""
    return any(item.get('tool') == tool and item['invoked'] for item in operations())


def reproduced_failure():
    """Whether the host recorded a failed verification of the original source before it changed."""
    revisions = [item for item in operations() if item.get('source_revision')]
    if not revisions:
        return False
    initial = revisions[0]['source_revision']
    return delivered_revision() != initial and any(
        item.get('source_verdict') == 'failed' and item['source_revision'] == initial for item in revisions)
