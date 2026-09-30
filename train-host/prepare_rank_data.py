"""Prepare shared selection data and family-separated rank training inputs.

python prepare_rank_data.py SESSIONS AUGMENTATION OUT
SESSIONS contains train.jsonl and val.jsonl under the frozen rubric. AUGMENTATION
contains generated train/eval families. The raw files are never modified.
"""
import json
import sys
from pathlib import Path


def read(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]


def text_key(row):
    return row['state']['user'].strip().casefold()


def write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))


def main():
    sessions, augmentation, output = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    train, val, generated = read(sessions / 'train.jsonl'), read(sessions / 'val.jsonl'), read(augmentation)
    split_by_family = {}
    for row in generated:
        family, split = row['meta']['family'], row['meta']['split']
        if family in split_by_family and split_by_family[family] != split:
            raise ValueError('family crosses splits: ' + family)
        split_by_family[family] = split
    training_texts = {text_key(r) for r in train}
    clean_val = [r for r in val if text_key(r) not in training_texts]
    selection_texts = {text_key(r) for r in clean_val + [r for r in generated if r['meta']['split'] == 'eval']}
    excluded = {r['meta']['family'] for r in generated if r['meta']['split'] == 'train' and text_key(r) in selection_texts}
    clean_generated = [r for r in generated if r['meta']['family'] not in excluded]
    write(output / 'judged/train.jsonl', train)
    write(output / 'judged/val.jsonl', clean_val)
    write(output / 'skillreq/train.judged.jsonl', clean_generated)
    write(output / 'selection.jsonl', clean_val + [r for r in clean_generated if r['meta']['split'] == 'eval'])
    report = {'session_train': len(train), 'session_val': len(clean_val), 'excluded_session_val_duplicates': len(val) - len(clean_val),
              'excluded_generated_families': sorted(excluded), 'generated_rows': len(clean_generated),
              'selection_rows': len(clean_val) + sum(r['meta']['split'] == 'eval' for r in clean_generated)}
    (output / 'preparation.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
