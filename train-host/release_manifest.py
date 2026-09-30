"""Bind selected weights to the exact preload vocabulary used for evaluation."""
import argparse
import hashlib
import json
import struct
from pathlib import Path


def header(path):
    with path.open('rb') as source:
        size = struct.unpack('<Q', source.read(8))[0]
        if size > 1024 * 1024:
            raise ValueError('oversized head header')
        return json.loads(source.read(size))['__metadata__']


def build(corpus, heads, release, revision):
    selected, metadata = {}, {}
    for name in ('turn-load', 'guide-load', 'unit-rank', 'code-rank'):
        path = heads / (name + '.safetensors')
        if name == 'guide-load' and not path.exists():
            continue
        metadata[name] = header(path)
        selected[name] = {'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'label': metadata[name]['label']}
    turn = metadata['turn-load']
    if any(m['model'] != turn['model'] for m in metadata.values()):
        raise ValueError('selected heads use different models')
    if turn['corpus'] != corpus['catalog_revision']:
        raise ValueError('turn head and training corpus differ')
    spec = corpus['questions']['tools']
    encoding = 'independent' if spec.get('independent') else 'joint'
    if turn.get('tool_encoding', 'joint') != encoding:
        raise ValueError('turn head and corpus encodings differ')
    options = spec.get('options') or {tool['name']: tool['option'] for tool in corpus['tools']}
    return {'version': 1, 'release': release, 'backbone': {'model': turn['model'], 'revision': revision},
            'heads': selected, 'preload': {'catalog_revision': corpus['catalog_revision'], 'encoding': encoding,
                                         'option_words': spec['option_words'], 'max_len': int(turn['max_len']),
                                         'head_tokens': int(turn['head_max_len']), 'options': dict(sorted(options.items()))}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('corpus', 'heads', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--release', required=True)
    parser.add_argument('--revision', default='e4e9ddf21a7b1903b7acffd8814ad4307bf63a67')
    args = parser.parse_args()
    result = build(json.loads(args.corpus.read_text()), args.heads, args.release, args.revision)
    with args.out.open('x') as output:
        output.write(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
