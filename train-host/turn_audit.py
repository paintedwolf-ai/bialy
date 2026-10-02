"""Uncached turn inference checks for repeatability, batching, and roster stability."""
import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path

from rank_eval import head_hashes


def variants(request, extra_name, extra_description):
    """Each variant owns its input; none modifies the reference request."""
    repeated = copy.deepcopy(request)
    solo = copy.deepcopy(request)
    solo['questions'] = {'tools': solo['questions']['tools']}
    grown = copy.deepcopy(request)
    options = grown['questions']['tools']['options']
    if extra_name in options:
        raise ValueError('extra tool already exists in the reference roster')
    options[extra_name] = extra_description
    return [('repeat', repeated), ('tools_only', solo), ('added_tool', grown)]


def gaps(reference, measured):
    if not reference or not reference.keys() <= measured.keys():
        raise ValueError('missing reference tool probabilities')
    return {name: abs(score - measured[name]) for name, score in reference.items()}


def receipt_requests(receipts, corpus):
    for receipt in receipts:
        if receipt['trigger'] != 'turn':
            continue
        decision = json.loads(receipt['decisions_json'])['turn']
        questions = {}
        for kind, key in [('tool', 'tools'), ('guide', 'guides')]:
            if key == 'guides' and corpus.spec['tools'].get('independent'):
                continue
            spec = corpus.spec[key]
            options = {c['id']: ' '.join(c['description'].split()[:spec['option_words']]).rstrip(',;:')
                       for c in decision['candidates'] if c['kind'] == kind}
            if options:
                questions[key] = {'type': 'multi', 'instructions': spec['question'],
                                  'options': dict(sorted(options.items()))}
                if spec.get('independent'):
                    questions[key]['independent'] = True
        if not corpus.spec['tools'].get('independent'):
            questions['kind'] = corpus.kind_question()
        yield str(receipt['id']), {'method': 'decide', 'head': 'turn-load',
                                   'state': json.loads(receipt['state_json']), 'questions': questions}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ['trainer', 'corpus', 'engine', 'out']:
        ap.add_argument('--' + name, required=True)
    inputs = ap.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--rows')
    inputs.add_argument('--receipts')
    ap.add_argument('--head-file', action='append', required=True)
    ap.add_argument('--extra-name', default='audit_extra_tool')
    ap.add_argument('--extra-description', default='An unrelated additional catalog tool')
    ap.add_argument('--tolerance', type=float, default=.01)
    ap.add_argument('--limit', type=int, default=12)
    args = ap.parse_args()
    sys.path.insert(0, str(Path(args.trainer) / 'scripts/bialy'))
    from corpus import Corpus
    from replay_eval import Engine

    corpus = Corpus.load(args.corpus)
    if args.receipts:
        requests = list(receipt_requests(json.loads(Path(args.receipts).read_text()), corpus))
    else:
        rows = [json.loads(line) for line in Path(args.rows).read_text().splitlines() if line.strip()]
        requests = [(f"{r['session']}:{r['receipt']}", {'method': 'decide', 'head': 'turn-load',
                     'state': r['state'], 'questions': corpus.row_questions(r)}) for r in rows if not r['partial']]
    requests = [(key, req) for key, req in requests if 'tools' in req['questions']][:args.limit]
    if not requests:
        raise ValueError('no tool questions to compare')
    output = Path(args.out)
    signature = {'heads': head_hashes(args.head_file), 'sha256': {
        name: hashlib.sha256(Path(path).read_bytes()).hexdigest()
        for name, path in [('corpus', args.corpus), ('inputs', args.receipts or args.rows), ('launcher', args.engine)]}}
    maxima = {'repeat': 0., 'tools_only': 0., 'added_tool': 0.}
    engine = Engine(args.engine)
    try:
        with output.open('x') as fh:
            fh.write(json.dumps({'signature': signature, 'hello': engine.hello}) + '\n')
            for key, request in requests:
                # ask bypasses the evaluator's memoization; every check reaches the engine.
                baseline = engine.ask(copy.deepcopy(request))
                reference = baseline['answers']['tools']['probabilities']
                fh.write(json.dumps({'key': key, 'variant': 'baseline', 'request': request, 'response': baseline}) + '\n')
                for name, variant in variants(request, args.extra_name, args.extra_description):
                    response = engine.ask(variant)
                    delta = gaps(reference, response['answers']['tools']['probabilities'])
                    maxima[name] = max(maxima[name], max(delta.values()))
                    fh.write(json.dumps({'key': key, 'variant': name, 'response': response, 'gaps': delta}) + '\n')
                    fh.flush()
    finally:
        engine.close()
    result = {'rows': len(requests), 'max_gaps': maxima, 'tolerance': args.tolerance,
              'passed': all(value <= args.tolerance for value in maxima.values())}
    Path(str(output) + '.summary.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))
    if not result['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
