"""Audit omissions including unlabeled units, and propose a supported allowlist.

Usage: guide_audit.py TRAIN_JSONL PREDICTIONS_JSONL OUT_JSON
Uses the installed B2 thresholds (.38 probability, .5 confidence). A unit must
have both label classes in training and validation, at least five examples of
each on validation, and at least five omissions at 97% precision with 98% needed-guide retention
to qualify.
Labels are observational proxies; this does not establish instruction usefulness.
"""
import collections
import json
import sys
from pathlib import Path


def audit(train, predictions, threshold=.38, confidence=.5):
    training = collections.defaultdict(collections.Counter)
    for row in train:
        for name, value in row['labels']['guides'].items():
            if value is not None:
                training[name]['positive' if value else 'negative'] += 1
    stats = collections.defaultdict(collections.Counter)
    for row in predictions:
        for name, value in row['answers'].get('guides', {}).get('probabilities', {}).items():
            needed = row['labels']['guides'].get(name)
            omitted = value < threshold and max(value, 1 - value) >= confidence
            stats[name]['scored'] += 1
            stats[name]['omitted'] += omitted
            if needed is None:
                stats[name]['unknown'] += 1
                stats[name]['omitted_unknown'] += omitted
            else:
                stats[name]['positive' if needed else 'negative'] += 1
                stats[name]['omitted_needed' if needed else 'omitted_unneeded'] += omitted
    allowed = []
    for name, s in stats.items():
        known_omissions = s['omitted_needed'] + s['omitted_unneeded']
        if (training[name]['positive'] > 0 and training[name]['negative'] > 0
                and s['positive'] >= 5 and s['negative'] >= 5 and not s['omitted_unknown']
                and known_omissions >= 5 and s['omitted_unneeded'] / known_omissions >= .97
                and 1 - s['omitted_needed'] / s['positive'] >= .98):
            allowed.append(name)
    return {'omittable': sorted(allowed), 'omit_below': threshold, 'confidence_floor': confidence,
            'training': {k: dict(v) for k, v in sorted(training.items())},
            'validation': {k: dict(v) for k, v in sorted(stats.items())},
            'limitation': 'Observed tool use supplies guide labels; unlabeled guidance cannot be certified for omission.'}


def main():
    train, predictions, output = sys.argv[1:]
    def read(path):
        return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]

    training, scored = read(train), read(predictions)
    candidates = [audit(training, scored, step / 100) for step in range(1, 39)]
    result = max(candidates, key=lambda r: sum(r['validation'][n]['omitted'] for n in r['omittable']))
    if not result['omittable']:
        result['omit_below'] = 0
    result['original_threshold_audit'] = audit(training, scored)
    result['selection_rule'] = 'Maximize supported omissions on validation; per-unit precision >= .97 and needed retention >= .98.'
    Path(output).write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
