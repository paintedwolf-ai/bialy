"""Select hard zero-score skill pairs from training-only rank predictions.

Preserve all nonzero judged pairs and the original labels of selected zeros.
The source row hash must match the prediction manifest. The output is a derived
training sample, never a replacement for the archived full judge labels.
"""
import argparse
import copy
import hashlib
import json
import random
from pathlib import Path


def key(row):
    return f"{row['session']}:{row['receipt']}"


def sample(row, prediction, hard=8, random_zeros=4, seed=11):
    out = copy.deepcopy(row)
    if row['partial']:
        return out
    first = row['labels']['skill_scores']
    second = row['labels']['second_scores']['skills']
    if set(first) != set(second) or set(first) != set(prediction['skills']['scores']):
        raise ValueError('complete paired labels and predictions are required: ' + key(row))
    if any(sorted((value, second[name])) != prediction['skills']['labels'][name] for name, value in first.items()):
        raise ValueError('prediction labels differ from source: ' + key(row))
    levels = {name: (value + second[name]) // 2 for name, value in first.items()}
    zeros = sorted((name for name in first if levels[name] == 0),
                   key=lambda name: (-prediction['skills']['scores'][name], name))
    hard_names = zeros[:hard]
    rng = random.Random(str(seed) + ':' + key(row))
    random_names = rng.sample(zeros[hard:], min(random_zeros, len(zeros[hard:])))
    keep = {name for name, level in levels.items() if level > 0} | set(hard_names + random_names)
    out['labels']['skill_scores'] = {name: value for name, value in first.items() if name in keep}
    out['labels']['second_scores']['skills'] = {name: value for name, value in second.items() if name in keep}
    out['meta'] = {**out.get('meta', {}), 'skill_pair_sampling': {'hard': hard_names, 'random': random_names, 'seed': seed}}
    return out


def main():
    ap = argparse.ArgumentParser()
    for name in ['rows', 'predictions', 'out']:
        ap.add_argument('--' + name, required=True)
    args = ap.parse_args()
    source = Path(args.rows)
    meta = json.loads(Path(args.predictions + '.meta.json').read_text())
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if meta['rows'] != digest or not meta.get('heads', {}).get('unit-rank'):
        raise ValueError('prediction manifest does not bind these rows and a rank head')
    rows = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
    predictions = [json.loads(line) for line in Path(args.predictions).read_text().splitlines() if line.strip()]
    lookup = {row['key']: row for row in predictions}
    if len(lookup) != len(predictions) or len({key(row) for row in rows}) != len(rows) or set(lookup) != {key(row) for row in rows}:
        raise ValueError('prediction coverage or row identities differ')
    output = Path(args.out)
    with output.open('x') as fh:
        for row in rows:
            fh.write(json.dumps(sample(row, lookup[key(row)]), ensure_ascii=False) + '\n')
    output.with_suffix(output.suffix + '.provenance.json').write_text(json.dumps({
        'source_sha256': digest, 'predictions_sha256': hashlib.sha256(Path(args.predictions).read_bytes()).hexdigest(),
        'output_sha256': hashlib.sha256(output.read_bytes()).hexdigest(), 'mining_head_sha256': meta['heads']['unit-rank'],
        'rows': len(rows), 'hard_zeros': 8, 'random_zeros': 4, 'seed': 11,
    }, indent=2) + '\n')


if __name__ == '__main__':
    main()
