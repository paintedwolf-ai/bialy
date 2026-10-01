"""Save raw turn-load predictions without skill inference or calibration.

Usage: turn_dump.py TRAINER CORPUS ROWS ENGINE OUT NAME=HEAD_PATH...
"""
import hashlib
import json
import sys
from pathlib import Path

from rank_eval import head_hashes


def main():
    trainer, corpus_path, rows_path, launcher, output, *heads = sys.argv[1:]
    hashes = head_hashes(heads)
    sys.path.insert(0, str(Path(trainer) / 'scripts/bialy'))
    from corpus import Corpus
    from replay_eval import Engine

    corpus, engine = Corpus.load(corpus_path), Engine(launcher)
    rows = [json.loads(line) for line in Path(rows_path).read_text().splitlines() if line.strip()]
    signature = {name: hashlib.sha256(Path(path).read_bytes()).hexdigest()
                 for name, path in [('corpus', corpus_path), ('rows', rows_path), ('launcher', launcher)]}
    signature['heads'] = hashes
    signature['engine'] = engine.hello
    signature['engine'].pop('id', None)
    meta = Path(output + '.meta.json')
    if meta.exists() and json.loads(meta.read_text()) != signature:
        raise ValueError('prediction signature changed')
    if not meta.exists():
        if Path(output).exists():
            raise ValueError('predictions without signature')
        meta.write_text(json.dumps(signature, indent=2) + '\n')
    done = {json.loads(s)['key'] for s in Path(output).read_text().splitlines()} if Path(output).exists() else set()
    try:
        with open(output, 'a') as fh:
            for row in rows:
                key = f"{row['session']}:{row['receipt']}"
                if key in done or row['partial']:
                    continue
                response = engine.call({'method': 'decide', 'head': 'turn-load', 'state': row['state'],
                                        'questions': corpus.row_questions(row)})
                fh.write(json.dumps({'key': key, 'host': row['host'], 'labels': row['labels'], 'answers': response['answers']}) + '\n')
                fh.flush()
                done.add(key)
                if len(done) % 50 == 0:
                    print(len(done), 'rows', flush=True)
    finally:
        engine.close()


if __name__ == '__main__':
    main()
