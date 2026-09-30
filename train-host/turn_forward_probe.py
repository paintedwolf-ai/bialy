"""Save the trainer's own tool probabilities for an independent runtime check."""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('trainer', 'corpus', 'rows', 'head', 'out'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--model', default='convaiinnovations/laya-multilingual')
    parser.add_argument('--limit', type=int, default=6, help='complete rows to score; 0 scores every complete row')
    args = parser.parse_args()
    sys.path.insert(0, str(Path(args.trainer) / 'scripts/decide'))
    import laya
    import torch
    from corpus import Corpus
    from headfile import read_metadata
    from parity_probe import torch_probs
    from rows import load
    from safetensors.torch import load_file

    torch.set_num_threads(8)
    device = os.environ.get('LYCAON_DECIDE_DEVICE', 'cpu')
    corpus = Corpus.load(args.corpus)
    agent = laya.load(args.model, device=device)
    meta = read_metadata(args.head)[0]
    context = int(agent.cfg.get('max_len', 512))
    budget = min(int(corpus.state_spec.get('head_tokens', 512)), context - 64)
    encoding = 'independent' if corpus.spec['tools'].get('independent') else 'joint'
    if (meta.get('max_len'), meta.get('head_max_len'), meta.get('tool_encoding', 'joint')) != (str(context), str(budget), encoding):
        raise ValueError('head encoding metadata does not match inference')
    tensors = load_file(args.head, device=device)
    for name in ('head', 'scorer', 'type_emb'):
        getattr(agent.model, name).load_state_dict({k[len(name) + 1:]: v for k, v in tensors.items() if k.startswith(name + '.')})
    agent.model.eval()
    if args.limit < 0:
        raise ValueError('limit must be nonnegative')
    examples = [r for r in load(args.rows) if not r['partial']]
    if args.limit:
        examples = examples[:args.limit]
    if not examples:
        raise ValueError('no complete rows')
    signature = {name: hashlib.sha256(Path(getattr(args, name)).read_bytes()).hexdigest() for name in ('head', 'corpus', 'rows')}
    with Path(args.out).open('x') as out:
        for row in examples:
            probabilities = torch_probs(agent, agent.model, corpus, row, device)
            json.dump({'key': f"{row['session']}:{row['receipt']}",
                       'answers': {'tools': {'probabilities': probabilities}}}, out)
            out.write('\n')
            out.flush()
    Path(args.out + '.meta.json').write_text(json.dumps({'sha256': signature, 'encoding': meta, 'device': device}, indent=2))


if __name__ == '__main__':
    main()
