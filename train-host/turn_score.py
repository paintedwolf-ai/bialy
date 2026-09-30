"""Score saved preload predictions on both consensus and observed-use labels.

Select thresholds only on validation. Supply --threshold for a frozen-policy
acceptance report; the file name does not establish which split was used.
"""
import argparse
import collections
import hashlib
import json
import sys
from pathlib import Path


def score(predictions, examples, targets, threshold, known_tools=None):
    count = collections.Counter()
    per_tool = collections.defaultdict(collections.Counter)
    for prediction in predictions:
        row = examples[prediction['key']]
        probabilities = prediction['answers']['tools']['probabilities']
        expected = targets(row)
        if known_tools is not None:
            excluded = {name: value for name, value in expected.items() if name not in known_tools}
            count['out_of_catalog_labels'] += len(excluded)
            count['out_of_catalog_positives'] += sum(value == 1 for value in excluded.values())
            expected = {name: value for name, value in expected.items() if name in known_tools}
        selected = {name for name, value in probabilities.items() if value >= threshold}
        positive = {name for name, value in expected.items() if value == 1}
        count['turns'] += 1
        count['loads'] += len(selected)
        count['empty'] += not selected
        count['needed_turns'] += bool(positive)
        count['any_needed_loaded'] += bool(selected & positive)
        count['all_needed_loaded'] += bool(positive) and positive <= selected
        for name, value in expected.items():
            if name not in probabilities:
                raise ValueError('prediction missing offered tool: ' + name)
            if value is None:
                count['unknown_loaded'] += name in selected
                continue
            metric = ('tp' if value else 'fp') if name in selected else ('fn' if value else 'tn')
            count[metric] += 1
            per_tool[name][metric] += 1
    tp, fp, fn = (count[k] for k in ('tp', 'fp', 'fn'))
    return {'threshold': threshold, 'counts': dict(count),
            'precision': tp / (tp + fp) if tp + fp else None,
            'recall': tp / (tp + fn) if tp + fn else None,
            'f0.5': 1.25 * tp / (1.25 * tp + .25 * fn + fp) if tp + fp + fn else 0,
            'loads_per_turn': count['loads'] / count['turns'] if count['turns'] else None,
            'empty_turn_rate': count['empty'] / count['turns'] if count['turns'] else None,
            'per_tool': {name: dict(value) for name, value in sorted(per_tool.items())}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('trainer', 'predictions', 'rows', 'corpus', 'out'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--threshold', type=float, action='append')
    args = parser.parse_args()
    sys.path.insert(0, str(Path(args.trainer) / 'scripts/decide'))
    from corpus import Corpus
    from rows import load, tool_targets
    corpus = Corpus.load(args.corpus)
    examples = {f"{r['session']}:{r['receipt']}": r for r in load(args.rows) if not r['partial']}
    predictions = [json.loads(s) for s in Path(args.predictions).read_text().splitlines() if s.strip()]
    if not predictions or {p['key'] for p in predictions} != examples.keys() or len(predictions) != len(examples):
        raise ValueError('predictions must cover every non-partial row exactly once')
    thresholds = args.threshold or [.5, .6, .7, .75, .8, .85, .89, .9, .95, .97, .99]
    report = {'sha256': {name: hashlib.sha256(Path(getattr(args, name)).read_bytes()).hexdigest()
                         for name in ('predictions', 'rows', 'corpus')}, 'truth': {}}
    for rule in ('consensus', 'called'):
        report['truth'][rule] = [score(predictions, examples, lambda row, rule=rule: tool_targets(row, rule), t, corpus.tool_options) for t in thresholds]
    with Path(args.out).open('x') as out:
        json.dump(report, out, indent=2)
    print(json.dumps({r: [{k: v for k, v in m.items() if k != 'per_tool'} for m in values]
                      for r, values in report['truth'].items()}, indent=2))


if __name__ == '__main__':
    main()
