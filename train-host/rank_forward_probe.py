"""Compare rank scores from the trainer's forward with a native-engine launcher.

This is a diagnostic, separate from CUDA/MLX decision parity. It records the
actual score gaps rather than declaring sparse comparison a release pass.
"""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    for name in ['trainer', 'head', 'engine', 'corpus', 'rows', 'out']:
        ap.add_argument('--' + name, required=True)
    ap.add_argument('--limit', type=int, default=6)
    args = ap.parse_args()
    sys.path.insert(0, str(Path(args.trainer) / 'scripts/bialy'))
    import laya
    import torch
    from corpus import Corpus
    from laya.common import QTYPES, build_sequence
    from replay_eval import Engine
    from safetensors.torch import load_file
    from train import RANK_QUESTION, forward_head, precompute

    torch.set_num_threads(4)
    device = os.environ.get('LYCAON_DECIDE_DEVICE', 'cuda')
    agent = laya.load('convaiinnovations/laya-multilingual', device=device)
    model = agent.model
    tensors = load_file(args.head, device=device)
    for name in ['head', 'scorer', 'type_emb']:
        state = {key[len(name) + 1:]: value for key, value in tensors.items() if key.startswith(name + '.')}
        if state:
            getattr(model, name).load_state_dict(state)
    model.eval()
    cards = Corpus.load(args.corpus).skill_cards()
    names = sorted(cards)
    rows = [json.loads(line) for line in Path(args.rows).read_text().splitlines() if line.strip()]
    rows = [row for row in rows if not row['partial']][:args.limit]
    if not rows:
        raise ValueError('no complete rows to compare')
    engine = Engine(args.engine)
    results = []
    try:
        for row in rows:
            items = []
            for name in names:
                text = 'Task: %s\n\nCandidate:\n%s' % (row['state']['user'], cards[name])
                ids, markers = build_sequence(agent.tok, text, RANK_QUESTION, max_len=int(agent.cfg['max_len']))
                items.append({'ids': ids, 'markers': markers, 'qtype': QTYPES['score'], 'label': 0})
            native = engine.call({'method': 'rank', 'head': 'unit-rank', 'task': row['state']['user'],
                                  'candidates': [cards[name] for name in names]})['scores']
            trainer = []
            for item in precompute(agent, items, device):
                with torch.no_grad():
                    logits = forward_head(model, item['h'][None].to(device), item['att'][None].to(device),
                                          item['marker_pos'][None].to(device), item['marker_mask'][None].to(device),
                                          torch.tensor([QTYPES['score']], device=device))[0]
                    trainer.append(float((logits.softmax(-1) * torch.arange(len(logits), device=device)).sum()))
            results.append({'key': f"{row['session']}:{row['receipt']}",
                            'scores': {name: {'trainer': a, 'native': b} for name, a, b in zip(names, trainer, native, strict=True)}})
    finally:
        engine.close()
    gaps = [abs(pair['trainer'] - pair['native']) for row in results for pair in row['scores'].values()]
    report = {'head_sha256': hashlib.sha256(Path(args.head).read_bytes()).hexdigest(),
              'comparisons': len(gaps), 'max_gap': max(gaps), 'rows': results}
    Path(args.out).write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({key: value for key, value in report.items() if key != 'rows'}))


if __name__ == '__main__':
    main()
