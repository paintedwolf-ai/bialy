"""Build evaluation-only tool-event variants from recorded called tool identities.

Rows do not preserve call order, so every observed loadable tool is a separate
sensitivity probe. Original request labels are retained; these are not newly
judged event labels and never enter training.
"""
import copy
import json
import sys
import uuid
from pathlib import Path


def variants(rows, max_chars=900):
    for row in rows:
        if row['partial']:
            continue
        for tool in sorted(set(row['labels'].get('tools', [])) & set(row['offered']['loadable'])):
            out = copy.deepcopy(row)
            key = f"{row['session']}:{row['receipt']}:{tool}"
            out['session'] = str(uuid.uuid5(uuid.NAMESPACE_URL, 'pw-decide-tool-event:' + key))
            user = row['state']['user'].strip()
            if max_chars > 0:
                user = user[:max_chars].strip()
            out['state']['user'] = (user + f'\n\nThe agent just called `{tool}`.').strip()
            out['labels']['requests'] = []
            out['meta'] = {**out.get('meta', {}), 'source': 'tool-event-probe', 'original_session': row['session'],
                           'original_receipt': row['receipt'], 'event_tool': tool}
            yield out


def main():
    source, output = map(Path, sys.argv[1:])
    rows = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
    with output.open('x') as fh:
        for row in variants(rows):
            fh.write(json.dumps(row, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    main()
