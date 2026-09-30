"""Save reusable rank predictions and measure discovery without rerunning inference.

The dump is resumable and refuses changed inputs or engine identity. Skill names
and labels never enter the task text; every catalog card is scored. Scoring a
roster subset filters these independent card scores after inference.
"""
import argparse
import collections
import hashlib
import json
import re
import sys
from pathlib import Path


def read(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]


def paired(row, unit):
    labels = row['labels']
    first = labels.get('skill_scores', {}) if unit == 'skills' else labels['requests'][unit].get('scores', {})
    second = labels.get('second_scores', {})
    second = second.get('skills', {}) if unit == 'skills' else ((second.get('requests') or []) + [{}] * (unit + 1))[unit]
    return {k: (min(v, second[k]), max(v, second[k])) for k, v in (first or {}).items() if k in (second or {})}


def dump(args):
    sys.path.insert(0, str(Path(args.trainer) / 'scripts/decide'))
    from corpus import Corpus
    from replay_eval import Engine
    corpus = Corpus.load(args.corpus)
    skills, tools = corpus.skill_cards(), corpus.tool_cards()
    heads = head_hashes(args.head_file)
    engine = Engine(args.engine)
    signature = {'rows': hashlib.sha256(Path(args.rows).read_bytes()).hexdigest(),
                 'corpus': hashlib.sha256(Path(args.corpus).read_bytes()).hexdigest(), 'engine': engine.hello,
                 'launcher': hashlib.sha256(Path(args.engine).read_bytes()).hexdigest(),
                 'heads': heads, 'requests_only': args.requests_only}
    # hello id is a protocol sequence, not an artifact identity.
    signature['engine'].pop('id', None)
    meta = Path(args.out + '.meta.json')
    if meta.exists() and json.loads(meta.read_text()) != signature:
        engine.close()
        raise ValueError('prediction signature changed; use a new output path')
    if not meta.exists():
        if Path(args.out).exists():
            engine.close()
            raise ValueError('predictions exist without a signature')
        meta.write_text(json.dumps(signature, indent=2) + '\n')
    done = {x['key'] for x in read(args.out)} if Path(args.out).exists() else set()
    try:
        with open(args.out, 'a') as fh:
            for row in read(args.rows):
                key = f"{row['session']}:{row['receipt']}"
                if key in done:
                    continue
                result = {'key': key, 'meta': row.get('meta', {}), 'host': row['host'], 'skills': None, 'requests': []}
                if not row['partial'] and not args.requests_only:
                    names = sorted(skills)
                    response = engine.call({'method': 'rank', 'head': 'unit-rank', 'task': row['state']['user'],
                                            'candidates': [skills[n] for n in names]})
                    result['skills'] = {'scores': dict(zip(names, response['scores'], strict=True)), 'labels': paired(row, 'skills')}
                for i, need in enumerate(row['labels'].get('requests', [])):
                    offered = set(row['offered']['loadable']) & set(tools)
                    exact = set(re.findall(r'[a-z][a-z0-9_]*', need['need'].lower())) & offered
                    names = sorted(offered - exact)
                    labels = paired(row, i)
                    wanted = (set(need.get('after', [])) | {k for k, p in labels.items() if p[0] >= 3}) & set(names)
                    if not names:
                        result['requests'].append({'scores': {}, 'wanted': [], 'exact': sorted(exact),
                                                   'called': sorted(set(need.get('after', [])) & offered)})
                        continue
                    response = engine.call({'method': 'rank', 'head': 'unit-rank', 'task': need['need'],
                                            'candidates': [tools[n] for n in names]})
                    result['requests'].append({'scores': dict(zip(names, response['scores'], strict=True)), 'wanted': sorted(wanted), 'exact': sorted(exact),
                                               'called': sorted(set(need.get('after', [])) & offered), 'labels': labels, 'need': need['need']})
                fh.write(json.dumps(result) + '\n')
                fh.flush()
                if len(done) % 50 == 0:
                    print(f'{len(done)} rows', flush=True)
                done.add(key)
    finally:
        engine.close()


def head_hashes(entries):
    if not entries:
        raise ValueError('rank dumps require --head-file NAME=PATH for each loaded head')
    hashes = {}
    for entry in entries:
        name, sep, path = entry.partition('=')
        if not sep or not name or name in hashes:
            raise ValueError('invalid or duplicate head identity: ' + name)
        hashes[name] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    return hashes


def request_metrics(rows, threshold=2.35, maximum=4, nearest=1):
    n = top = served = hit = loaded_count = wanted_count = 0
    for row in rows:
        for need in row['requests']:
            order = sorted(need['scores'], key=lambda s: (-need['scores'][s], s))
            wanted = set(need['wanted'])
            if not wanted or not order:
                continue
            loaded = {s for s in order[:maximum] if need['scores'][s] >= threshold} or set(order[:nearest])
            n += 1
            top += order[0] in wanted
            served += bool(loaded & wanted)
            hit += len(loaded & wanted)
            loaded_count += len(loaded)
            wanted_count += len(wanted)
    return {'needs': n, 'top1': top / n if n else None, 'served': served / n if n else None,
            'precision': hit / loaded_count if loaded_count else None,
            'f1': 2 * hit / (loaded_count + wanted_count) if loaded_count + wanted_count else 0,
            'loads_per_need': loaded_count / n if n else None}


def request_surface_metrics(rows, threshold=2.35, maximum=4, nearest=1):
    """Include exact names and requests with no remaining positive label.

    Called coverage is observed behavior, not a claim that every uncalled tool
    was unnecessary. Precision uses only cards on which both judges scored.
    """
    counts = collections.Counter()
    for row in rows:
        for need in row['requests']:
            scores = need['scores']
            order = sorted(scores, key=lambda name: (-scores[name], name))
            ranked = {name for name in order[:maximum] if scores[name] >= threshold} or set(order[:nearest])
            exact = set(need.get('exact', []))
            loaded = exact | ranked
            called = set(need.get('called', []))
            counts['requests'] += 1
            counts['exact_requests'] += bool(exact)
            counts['selected_tools'] += len(loaded)
            counts['ranked_tools'] += len(ranked)
            if called:
                counts['called_requests'] += 1
                counts['any_called_served'] += bool(loaded & called)
                counts['all_called_served'] += called <= loaded
                counts['called_tools'] += len(called)
                counts['called_tools_served'] += len(called & loaded)
            for name in loaded:
                pair = need.get('labels', {}).get(name)
                if pair is not None:
                    counts['judged_selected'] += 1
                    counts['judged_relevant_selected'] += pair[0] >= 3
    result = dict(counts)
    for key in ('any_called_served', 'all_called_served'):
        result[key + '_rate'] = counts[key] / counts['called_requests'] if counts['called_requests'] else None
    result['selected_per_request'] = counts['selected_tools'] / counts['requests'] if counts['requests'] else None
    result['judged_precision'] = counts['judged_relevant_selected'] / counts['judged_selected'] if counts['judged_selected'] else None
    return result


def metrics(rows, roster, positives, preload=3.49, list_at=1.0):
    counts = collections.Counter()
    per = collections.defaultdict(collections.Counter)
    for row in rows:
        sk = row['skills']
        if sk is None:
            continue
        order = sorted(set(sk['scores']) & set(roster), key=lambda s: (-sk['scores'][s], s))
        if not order:
            continue
        labels = sk['labels']
        relevant = {s for s in order if s in labels and labels[s][0] >= 3}
        counts['rows'] += 1
        # A negative row has complete, agreed non-relevance, not missing scores.
        no_skill = all(s in labels and labels[s][1] < 3 for s in order)
        counts['no_skill_rows'] += no_skill
        generated_none = row.get('meta', {}).get('kind') == 'none'
        counts['generated_none_rows'] += generated_none
        counts['generated_none_agreed_negative'] += generated_none and no_skill
        if sk['scores'][order[0]] >= preload:
            counts['preloads'] += 1
            counts['preloads_unscored'] += order[0] not in labels
            counts['preloads_correct'] += order[0] in relevant
            pair = labels.get(order[0])
            if pair is not None and pair[0] < 3:
                category = 'disputed' if pair[1] >= 3 else ('possible' if pair[1] == 2 else 'unlikely')
                counts['preloads_' + category] += 1
            counts['no_skill_preloads'] += no_skill
            counts['generated_none_preloads'] += generated_none
        if not relevant:
            continue
        counts['relevant_rows'] += 1
        counts['relevant_preloaded'] += sk['scores'][order[0]] >= preload and order[0] in relevant
        for k in (6, 8):
            visible = {s for s in order[:k] if sk['scores'][s] >= list_at}
            counts[f'any_raw@{k}'] += bool(set(order[:k]) & relevant)
            counts[f'any_visible@{k}'] += bool(visible & relevant)
            counts[f'all_visible@{k}'] += relevant <= visible
            for s in relevant:
                per[s][f'visible@{k}'] += s in visible
        for s in relevant:
            per[s]['support'] += 1
    out = dict(counts)
    for key in list(out):
        if '@' in key:
            out[key] /= counts['relevant_rows']
    out['preload_precision'] = counts['preloads_correct'] / counts['preloads'] if counts['preloads'] else None
    out['no_skill_preload_rate'] = counts['no_skill_preloads'] / counts['no_skill_rows'] if counts['no_skill_rows'] else None
    out['per_skill'] = {s: dict(v) for s, v in sorted(per.items())}
    out['bands'] = {}
    for band, low, high in [('rare', 0, 10), ('mid', 10, 50), ('common', 50, 10**9)]:
        members = [v for s, v in per.items() if low <= positives[s] < high]
        support = sum(v['support'] for v in members)
        rates = [v['visible@6'] / v['support'] for v in members if v['support'] >= 3]
        out['bands'][band] = {'support': support, 'visible@6': sum(v['visible@6'] for v in members) / support if support else None,
                              'macro_support3': sum(rates) / len(rates) if rates else None}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['dump', 'score'])
    ap.add_argument('--rows', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--trainer')
    ap.add_argument('--corpus')
    ap.add_argument('--engine')
    ap.add_argument('--head-file', action='append', help='NAME=PATH; fingerprint each immutable loaded head')
    ap.add_argument('--roster')
    ap.add_argument('--train')
    ap.add_argument('--preload', type=float, default=3.49)
    ap.add_argument('--list-at', type=float, default=1.0)
    ap.add_argument('--calibrate', action='store_true')
    ap.add_argument('--requests-only', action='store_true', help='dump tool requests without skill inference')
    ap.add_argument('--thresholds', help='frozen selection report; never recalibrate acceptance')
    args = ap.parse_args()
    if args.mode == 'dump':
        dump(args)
        return
    rows = read(args.rows)
    roster = json.loads(Path(args.roster).read_text()) if args.roster else sorted(next(r['skills']['scores'] for r in rows if r['skills']))
    if isinstance(roster, dict):
        roster = roster['roster']
    if not roster or not all(isinstance(n, str) for n in roster):
        raise ValueError('roster must contain skill names')
    positives = collections.Counter()
    for row in read(args.train):
        if not row['partial']:
            positives.update(s for s, pair in paired(row, 'skills').items() if pair[0] >= 3)
    threshold = args.preload
    list_at = args.list_at
    request = (2.35, 4, 1)
    if args.thresholds:
        if args.calibrate:
            ap.error('--thresholds and --calibrate are exclusive')
        selected = json.loads(Path(args.thresholds).read_text())
        threshold = selected['preload_at']
        list_at = selected.get('list_at', 1.0)
        request = tuple(selected['request'][k] for k in ['load_at', 'max_loads', 'nearest_loads'])
    if args.calibrate:
        eligible = []
        for step in range(100, 401):
            m = metrics(rows, roster, positives, step / 100, list_at)
            if m['preload_precision'] is not None and m['preload_precision'] >= .9 and (m['no_skill_preload_rate'] or 0) <= .05:
                eligible.append((m['preloads_correct'], step / 100))
        threshold = max(eligible, key=lambda p: (p[0], -p[1]))[1] if eligible else 4.01
        best = -1
        for maximum in range(1, 7):
            for nearest in range(1, maximum + 1):
                for step in range(50, 401, 5):
                    candidate = (step / 100, maximum, nearest)
                    value = request_metrics(rows, *candidate)['f1']
                    if value > best:
                        best, request = value, candidate
    report = {'preload_at': threshold, 'list_at': list_at, 'request': dict(zip(['load_at', 'max_loads', 'nearest_loads'], request, strict=True)),
              'skills': metrics(rows, roster, positives, threshold, list_at), 'requests': request_metrics(rows, *request)}
    strata = collections.defaultdict(list)
    for row in rows:
        source = 'generated' if row.get('meta', {}).get('source') == 'skillreq' else 'session'
        strata[source + ':' + row.get('host', 'unknown')].append(row)
    report['strata'] = {name: {'skills': metrics(items, roster, positives, threshold, list_at),
                               'requests': request_metrics(items, *request)} for name, items in sorted(strata.items())}
    Path(args.out).write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({**{k: v for k, v in report.items() if k != 'strata'}, 'skills': {k: v for k, v in report['skills'].items() if k != 'per_skill'}}, indent=2))


if __name__ == '__main__':
    main()
