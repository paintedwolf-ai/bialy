"""Compare saved CUDA/MLX predictions for the same rows and head.

Usage: rank_parity.py CUDA_JSONL MLX_JSONL PRELOAD_AT OUT_JSON [ROSTER_JSON]
"""
import argparse
import json
from pathlib import Path


def rows(path):
    data = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    result = {r['key']: r for r in data}
    if len(result) != len(data):
        raise ValueError('duplicate prediction keys')
    return result


def compare(left, right, threshold, roster=None, request=(2.35, 4, 1), list_at=1.0):
    if left.keys() != right.keys():
        raise ValueError('prediction row sets differ')
    maximum = flips = decisions = far_flips = top6 = scored = 0
    visible_top6 = visible_top8 = preload_changes = hidden_far_flips = 0
    request_maximum = request_decisions = request_flips = request_far = 0
    for key in left:
        ar, br = left[key].get('requests', []), right[key].get('requests', [])
        if len(ar) != len(br):
            raise ValueError('request coverage differs')
        for x, y in zip(ar, br, strict=True):
            x, y = x['scores'], y['scores']
            if not x or x.keys() != y.keys():
                raise ValueError('request candidate sets differ or empty')
            bar, cap, nearest = request
            ox = sorted(x, key=lambda n: (-x[n], n))
            oy = sorted(y, key=lambda n: (-y[n], n))
            px = {n for n in ox[:cap] if x[n] >= bar} or set(ox[:nearest])
            py = {n for n in oy[:cap] if y[n] >= bar} or set(oy[:nearest])
            for n in x:
                request_maximum = max(request_maximum, abs(x[n] - y[n]))
                request_decisions += 1
                if (x[n] >= bar) != (y[n] >= bar):
                    request_flips += 1
                    request_far += max(abs(x[n] - bar), abs(y[n] - bar)) > .1
            request_decisions += 1
            if px != py:
                request_flips += 1
                # A boundary swap is close only if both score gaps are small.
                changed = px ^ py
                spread = max(max(x[n] for n in changed) - min(x[n] for n in changed),
                             max(y[n] for n in changed) - min(y[n] for n in changed))
                near_bar = all(max(abs(x[n] - bar), abs(y[n] - bar)) <= .1 for n in changed)
                request_far += not near_bar and spread > .1
        a, b = left[key]['skills'], right[key]['skills']
        if (a is None) != (b is None):
            raise ValueError('skill coverage differs')
        if a is None:
            continue
        a, b = a['scores'], b['scores']
        if a.keys() != b.keys():
            raise ValueError('candidate sets differ')
        names = set(a) if roster is None else set(a) & set(roster)
        if not names:
            raise ValueError('no shared roster candidates')
        oa = sorted(names, key=lambda n: (-a[n], n))
        ob = sorted(names, key=lambda n: (-b[n], n))
        scored += 1
        top6 += set(oa[:6]) != set(ob[:6])
        visible_top6 += {n for n in oa[:6] if a[n] >= list_at} != {n for n in ob[:6] if b[n] >= list_at}
        visible_top8 += {n for n in oa[:8] if a[n] >= list_at} != {n for n in ob[:8] if b[n] >= list_at}
        for name in names:
            maximum = max(maximum, abs(a[name] - b[name]))
            for bar in [list_at, threshold]:
                decisions += 1
                if (a[name] >= bar) != (b[name] >= bar):
                    flips += 1
                    far = max(abs(a[name] - bar), abs(b[name] - bar)) > .1
                    far_flips += far
                    hidden_far_flips += far and name not in oa[:8] and name not in ob[:8]
        # Preload identity is also a decision, not merely its score.
        pa = oa[0] if a[oa[0]] >= threshold else None
        pb = ob[0] if b[ob[0]] >= threshold else None
        decisions += 1
        if pa != pb:
            preload_changes += 1
            flips += 1
            if pa and pb:
                far_flips += abs(a[pa] - a[pb]) > .1 or abs(b[pa] - b[pb]) > .1
            else:
                n = pa or pb
                far_flips += max(abs(a[n] - threshold), abs(b[n] - threshold)) > .1
    return {'rows': len(left), 'skill_rows': scored, 'max_score_gap': maximum,
            'top6_set_changes': top6, 'visible_top6_set_changes': visible_top6, 'visible_top8_set_changes': visible_top8,
            'preload_identity_changes': preload_changes, 'hidden_far_threshold_flips': hidden_far_flips, 'list_at': list_at, 'threshold_decisions': decisions, 'threshold_flips': flips,
            'far_flips': far_flips, 'request_max_score_gap': request_maximum,
            'request_decisions': request_decisions, 'request_flips': request_flips,
            'request_far_flips': request_far,
            'passed': decisions > 0 and flips / decisions <= .01 and far_flips == 0
            and (request_decisions == 0 or request_flips / request_decisions <= .01) and request_far == 0}


def main():
    ap = argparse.ArgumentParser()
    for name in ['cuda', 'mlx', 'preload', 'out']:
        ap.add_argument(name)
    ap.add_argument('roster', nargs='?')
    ap.add_argument('--selection', help='selection report with frozen request thresholds')
    args = ap.parse_args()
    roster = json.loads(Path(args.roster).read_text()) if args.roster else None
    if isinstance(roster, dict):
        roster = roster['roster']
    request = (2.35, 4, 1)
    list_at = 1.0
    if args.selection:
        selected = json.loads(Path(args.selection).read_text())
        list_at = selected.get('list_at', 1.0)
        chosen = selected['request']
        request = tuple(chosen[k] for k in ['load_at', 'max_loads', 'nearest_loads'])
    result = compare(rows(args.cuda), rows(args.mlx), float(args.preload), roster, request, list_at)
    Path(args.out).write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))
    if not result['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
