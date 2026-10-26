import json

import pytest

from bialy.student_host.specification import arguments, load


def test_both_students_use_identical_context_in_pilot_and_full_training():
    spec = load()
    for student in ('e4b', '12b'):
        for pilot in (False, True):
            args = arguments(spec, student, '/models', '/splits', '/run', 'cpu', '/mirror', pilot=pilot)
            assert args[args.index('--max-length')+1] == str(spec['context']['max_sequence_tokens'])
            assert '--checkpoint-mirror' in args
    assert spec['context']['thinking'] is False


def test_thinking_and_truncation_cannot_be_enabled(tmp_path):
    spec = load()
    spec['context']['thinking'] = True
    path = tmp_path / 'students.json'
    path.write_text(json.dumps(spec))
    with pytest.raises(ValueError, match='context'):
        load(path)
