"""Measure each candidate row as the student's pinned template encodes it.

Writes one result per input line: its token count, or the reason the host's encoder
refused it. An oversize row is refused whole; nothing is sliced.
"""
import argparse
import json
import sys
from pathlib import Path


def measure(rows, encode):
    out = []
    for row in rows:
        try:
            out.append({'tokens': len(encode(row)['input_ids'])})
        except ValueError as exc:
            reason = getattr(exc, 'reason', None)
            if reason is None:
                raise
            out.append({'refused': 'sequence_over_limit' if reason == 'oversize' else reason,
                        'tokens': getattr(exc, 'tokens', None)})
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('checkout', 'model', 'rows', 'out'):
        parser.add_argument('--' + name, required=True, type=Path)
    parser.add_argument('--max-tokens', type=int, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.checkout / 'scripts/bialy/custom_model'))
    from data import encode
    from rendering import configure_training_renderer
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, trust_remote_code=False)
    configure_training_renderer(tokenizer, args.model)
    rows = [json.loads(line) for line in args.rows.read_text().splitlines() if line.strip()]
    results = measure(rows, lambda row: encode(tokenizer, row, row['target'], args.max_tokens))
    args.out.write_text(json.dumps(results) + '\n')


if __name__ == '__main__':
    main()
