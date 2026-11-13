"""What a student's sealed rows exercise, counted from structured host and campaign facts."""
import json
from collections import Counter

from .journal import CampaignFault

STRATA = ('p0', 'planned_fallback', 'runtime_fallback', 'conservative')
PARTITIONS = ('train', 'validation', 'system_test')


def compacted(episode):
    """The episode carries a host-written compaction checkpoint in any session."""
    return any(part.get('source') == 'compaction' and part.get('origin') == 'host'
               for session in episode.sessions for message in episode.messages(session)
               for part in message.get('content_parts') or [])


def stratum(collected, row):
    """A row's selector stratum: its episode's draw, or a runtime fallback where P0 abstained
    and the host offered tools as its fallback."""
    if collected not in ('p0', 'planned_fallback', 'conservative'):
        raise CampaignFault('row_stratum_unknown', details={'stratum': collected})
    tools = ((row.get('prompt_surface') or {}).get('tools')) or []
    if collected == 'p0' and any(tool.get('source') == 'fallback' for tool in tools):
        return 'runtime_fallback'
    return collected


def distribution(values):
    if not values:
        return {'count': 0, 'total': 0, 'min': None, 'p50': None, 'p90': None, 'max': None}
    ordered = sorted(values)
    def at(share):
        return ordered[min(len(ordered) - 1, int(share * len(ordered)))]
    return {'count': len(ordered), 'total': sum(ordered), 'min': ordered[0], 'p50': at(0.5), 'p90': at(0.9), 'max': ordered[-1]}


def ledger(entries, exclusions, faults):
    """One student's ledger. Each entry is a sealed row with its partition, task, episode,
    stratum, token count and whether its session is a worker's and its episode compacted."""
    out = {'independent': {}, 'categories': {}, 'tokens': {}}
    for partition in PARTITIONS:
        rows = [entry for entry in entries if entry['partition'] == partition]
        out['independent'][partition] = {
            'groups': len({entry['task']['root'] for entry in rows}),
            'repositories': len({json.loads(entry['task']['spec'])['workspace'] for entry in rows}),
            'episodes': len({entry['episode'] for entry in rows}), 'encodings': len(rows)}
        out['categories'][partition] = dict(sorted(Counter(entry['row']['admission']['category'] for entry in rows).items()))
        out['tokens'][partition] = distribution([entry['tokens'] for entry in rows])
    strata = Counter(entry['stratum'] for entry in entries)
    out['strata'] = {name: strata[name] for name in STRATA}
    out['exercise'] = {'compacted_episodes': len({entry['episode'] for entry in entries if entry['compacted']}),
                       'worker_rows': sum(entry['worker'] for entry in entries),
                       'worker_episodes': len({entry['episode'] for entry in entries if entry['worker']})}
    out['exclusions'] = dict(sorted(exclusions.items()))
    out['infrastructure_faults'] = dict(sorted(faults.items()))
    return out


def deficits(entries, minimum, focuses):
    """Per partition, how many independent episodes and which focuses a student is short of."""
    out = {}
    for partition in PARTITIONS:
        rows = [entry for entry in entries if entry['partition'] == partition]
        short = minimum[partition] - len({entry['task']['root'] for entry in rows})
        missing = sorted(set(focuses) - {entry['task']['focus'] for entry in rows})
        if short > 0 or missing:
            out[partition] = {'episodes_short': max(0, short), 'focuses_missing': missing}
    return out
