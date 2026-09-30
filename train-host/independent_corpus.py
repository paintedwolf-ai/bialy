"""Prepare the exported corpus for one-tool-per-row training and serving."""
import argparse
import copy
import json
from pathlib import Path


def prepare(corpus):
    out = copy.deepcopy(corpus)
    out['questions']['tools'].pop('options', None)
    out['questions']['tools'].update(independent=True, option_words=60)
    out['questions']['guides'].update(omittable=[], omit_below=0)
    out['questions']['kind']['veto_tools'] = False
    for tool in out['tools']:
        tool['option'] = ' '.join(tool['description'].split()[:60]).rstrip(',;:')
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('corpus')
    ap.add_argument('out')
    args = ap.parse_args()
    result = prepare(json.loads(Path(args.corpus).read_text()))
    with Path(args.out).open('x') as fh:
        fh.write(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
