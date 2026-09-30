"""Compare saved turn tool-load decisions when guide omission and kind veto are off."""
import argparse
import json
from pathlib import Path

from rank_parity import rows


def compare(left, right, threshold):
    if not left or left.keys() != right.keys():
        raise ValueError('nonempty matching turn coverage is required')
    decisions = flips = far = 0
    maximum = 0.0
    for key in left:
        a = left[key]['answers']['tools']['probabilities']
        b = right[key]['answers']['tools']['probabilities']
        if a.keys() != b.keys():
            raise ValueError('tool candidate sets differ')
        for name, value in a.items():
            maximum = max(maximum, abs(value - b[name]))
            decisions += 1
            if (value >= threshold) != (b[name] >= threshold):
                flips += 1
                far += max(abs(value - threshold), abs(b[name] - threshold)) > .05
    return {'rows': len(left), 'decisions': decisions, 'max_probability_gap': maximum,
            'threshold': threshold, 'flips': flips, 'far_flips': far,
            'passed': decisions > 0 and flips / decisions <= .01 and far == 0,
            'scope': 'Tool loads only; valid for the release with guides kept and kind veto disabled.'}


def main():
    ap = argparse.ArgumentParser()
    for name in ['cuda', 'mlx', 'out']:
        ap.add_argument(name)
    ap.add_argument('--threshold', type=float, required=True)
    args = ap.parse_args()
    result = compare(rows(args.cuda), rows(args.mlx), args.threshold)
    Path(args.out).write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))
    if not result['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
