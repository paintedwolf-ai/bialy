"""Compare request grouping policies on identical saved model predictions.

This measures returned schemas and observed-tool coverage. It is a counterfactual
replay, not an end-to-end run; live resource controls and future requests are not
simulated. Catalog families here affect schema loading, never permission grants.
"""
import argparse
import hashlib
import json
from pathlib import Path

import yaml


def expand(selected, catalog, policy, allowed):
    loaded = set(selected)
    if policy != 'none':
        for members in catalog.get('families', {}).values():
            if set(members) & selected:
                loaded.update(members)
    if policy == 'all':
        for name in selected:
            loaded.update(catalog.get('request_companions', {}).get(name, []))
    if policy not in ('none', 'families', 'all'):
        raise ValueError('unknown expansion policy: ' + policy)
    return loaded & set(allowed)


def compare(predictions, offered, catalog, threshold, maximum, nearest):
    details = []
    totals = {p: {'requests': 0, 'schemas': 0, 'added': 0, 'called_needs': 0,
                  'any_called': 0, 'all_called': 0, 'called_tools': 0, 'covered_tools': 0}
              for p in ('none', 'families', 'all')}
    for row in predictions:
        surface = offered[row['key']]
        floor = set(surface.get('floor', []))
        allowed = set(surface['loadable']) | floor
        for need in row['requests']:
            scores = need['scores']
            order = sorted(scores, key=lambda name: (-scores[name], name))
            ranked = {n for n in order[:maximum] if scores[n] >= threshold} or set(order[:nearest])
            selected = set(need.get('exact', [])) | ranked
            called = set(need.get('called', [])) - floor
            sets = {}
            for policy, total in totals.items():
                loaded = expand(selected, catalog, policy, allowed) - floor
                sets[policy] = sorted(loaded)
                total['requests'] += 1
                total['schemas'] += len(loaded)
                total['added'] += len(loaded - selected)
                if called:
                    total['called_needs'] += 1
                    total['any_called'] += bool(loaded & called)
                    total['all_called'] += called <= loaded
                    total['called_tools'] += len(called)
                    total['covered_tools'] += len(loaded & called)
            details.append({'key': row['key'], 'need': need.get('need'), 'selected': sorted(selected),
                            'called': sorted(called), 'sets': sets})
    for total in totals.values():
        total['schemas_per_request'] = total['schemas'] / total['requests'] if total['requests'] else None
        for metric in ('any_called', 'all_called'):
            total[metric + '_rate'] = total[metric] / total['called_needs'] if total['called_needs'] else None
    return {'summary': totals, 'details': details}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('predictions', 'rows', 'catalog', 'out'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--threshold', type=float, default=1.95)
    parser.add_argument('--maximum', type=int, default=4)
    parser.add_argument('--nearest', type=int, default=1)
    args = parser.parse_args()
    def read(path):
        return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]

    offered = {f"{r['session']}:{r['receipt']}": r['offered'] for r in read(args.rows)}
    result = compare(read(args.predictions), offered, yaml.safe_load(Path(args.catalog).read_text()),
                     args.threshold, args.maximum, args.nearest)
    result['sha256'] = {k: hashlib.sha256(Path(getattr(args, k)).read_bytes()).hexdigest()
                        for k in ('predictions', 'rows', 'catalog')}
    result['policy'] = {'threshold': args.threshold, 'maximum': args.maximum, 'nearest': args.nearest}
    with Path(args.out).open('x') as out:
        json.dump(result, out, indent=2)
    print(json.dumps(result['summary'], indent=2))


if __name__ == '__main__':
    main()
