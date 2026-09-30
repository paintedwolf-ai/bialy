"""Merge re-judged worker inputs without moving rows across their original splits.

Usage: merge_worker_requests.py SESSION_DIR REPAIRED_JSONL OUTPUT_DIR
Preserves unrelated labels and records every replacement by session/receipt.
"""
import copy
import hashlib
import json
import sys
from pathlib import Path


def read(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def key(row):
    return str(row['session']) + ':' + str(row['receipt'])


def merge(original, repaired):
    out = copy.deepcopy(original)
    for field in ('session', 'receipt', 'opening_message_id', 'root_session', 'host', 'partial'):
        if original[field] != repaired[field]:
            raise ValueError('worker identity differs: ' + field)
    if original['host'] != 'worker' or original['partial']:
        raise ValueError('only complete worker requests can be repaired')
    for field in ('requests', 'guides'):
        if original['labels'].get(field) != repaired['labels'].get(field):
            raise ValueError('unrelated labels differ: ' + field)
    old_second = original['labels'].get('second_scores', {})
    new_second = repaired['labels'].get('second_scores', {})
    if old_second.get('requests') != new_second.get('requests'):
        raise ValueError('unrelated second-judge requests differ')
    out['state']['user'] = repaired['state']['user']
    for field in ('skill_scores', 'tool_scores', 'second_scores'):
        out['labels'].pop(field, None)
        if field in repaired['labels']:
            out['labels'][field] = copy.deepcopy(repaired['labels'][field])
    out['judge'] = copy.deepcopy(repaired['judge'])
    out.setdefault('meta', {})['worker_request_recovery'] = repaired['meta']['worker_request_recovery']
    return out


def main():
    source, repairs, output = map(Path, sys.argv[1:])
    repaired = read(repairs)
    by_key = {key(r): r for r in repaired}
    if len(by_key) != len(repaired):
        raise ValueError('duplicate worker repair identity')
    if output.exists():
        raise ValueError('output exists; choose a fresh directory')
    output.mkdir(parents=True)
    used, report = set(), {'repairs_sha256': hashlib.sha256(repairs.read_bytes()).hexdigest(), 'splits': {}}
    for split in ('train', 'val', 'holdout'):
        path = source / (split + '.jsonl')
        rows, replaced = read(path), []
        for i, row in enumerate(rows):
            ident = key(row)
            if ident in by_key:
                if ident in used:
                    raise ValueError('worker identity crosses splits: ' + ident)
                rows[i] = merge(row, by_key[ident])
                used.add(ident)
                replaced.append(ident)
        (output / path.name).write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
        report['splits'][split] = {'rows': len(rows), 'replaced': replaced, 'source_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    if set(by_key) - used:
        raise ValueError('repairs contain unknown rows')
    (output / 'recovery.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({s: len(v['replaced']) for s, v in report['splits'].items()}))


if __name__ == '__main__':
    main()
