"""Whole generated families stay out of training when a request leaks to selection."""
import json
import subprocess
import sys
from pathlib import Path


def test_preparation_excludes_duplicate_family_and_session_val(tmp_path):
    def row(text, family=None, split=None):
        return {'state': {'user': text}, 'meta': {'family': family, 'split': split}}

    def write(path, rows):
        path.write_text(''.join(json.dumps(r) + '\n' for r in rows))

    sessions = tmp_path / 'sessions'
    sessions.mkdir()
    write(sessions / 'train.jsonl', [row('original')])
    write(sessions / 'val.jsonl', [row(' ORIGINAL '), row('selection')])
    generated = tmp_path / 'generated.jsonl'
    write(generated, [row('duplicate', 'a', 'train'), row('sibling', 'a', 'train'),
                      row('DUPLICATE', 'b', 'eval'), row('clean', 'c', 'train')])
    script = Path(__file__).parents[1] / 'train-host/prepare_rank_data.py'
    subprocess.run([sys.executable, str(script), str(sessions), str(generated), str(tmp_path / 'out')], check=True)
    result = json.loads((tmp_path / 'out/preparation.json').read_text())
    assert result['excluded_generated_families'] == ['a']
    assert result['session_val'] == 1
    assert result['generated_rows'] == 2
