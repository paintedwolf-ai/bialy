"""Save free-text skill rankings against a full or project-specific catalog.

These cases are diagnostics, not a prevalence-weighted acceptance set or training data.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

from rank_eval import head_hashes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('trainer', 'corpus', 'cases', 'engine', 'out'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--roster', help='JSON list of loaded skill names')
    parser.add_argument('--head-file', action='append', required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(Path(args.trainer) / 'scripts/decide'))
    from replay_eval import Engine
    corpus = json.loads(Path(args.corpus).read_text())
    names = set(json.loads(Path(args.roster).read_text())) if args.roster else None
    skills = sorted((s for s in corpus['skills'] if names is None or s['name'] in names), key=lambda s: s['name'])
    if not skills or names is not None and names != {s['name'] for s in skills}:
        raise ValueError('nonempty roster entirely represented in the corpus is required')
    cases = json.loads(Path(args.cases).read_text())
    if not cases or any(not c.get('need', '').strip() for c in cases):
        raise ValueError('nonempty diagnostic queries are required')
    signature = {'heads': head_hashes(args.head_file), 'sha256': {
        key: hashlib.sha256(Path(getattr(args, key)).read_bytes()).hexdigest()
        for key in ('corpus', 'cases', 'engine', 'roster') if getattr(args, key)}}
    engine = Engine(args.engine)
    try:
        with Path(args.out).open('x') as output:
            output.write(json.dumps({'signature': signature, 'engine': engine.hello, 'roster': [s['name'] for s in skills]}) + '\n')
            for case in cases:
                scores = engine.call({'method': 'rank', 'head': 'unit-rank', 'task': case['need'],
                                      'candidates': [s['card'] for s in skills]})['scores']
                ranked = sorted(zip([s['name'] for s in skills], scores, strict=True), key=lambda pair: -pair[1])
                output.write(json.dumps({'case': case['case'], 'need': case['need'], 'ranked': ranked}) + '\n')
                output.flush()
    finally:
        engine.close()


if __name__ == '__main__':
    main()
