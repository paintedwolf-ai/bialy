import json

import pytest

from bialy.student_host import __main__ as cli
from bialy.student_host.specification import arguments, load, students


def test_every_student_uses_identical_context_in_pilot_and_full_training():
    spec = load()
    for student in students(spec):
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


def test_students_are_whatever_the_specification_declares(tmp_path, capsys):
    spec = load()
    assert students(spec) == tuple(spec['models'])
    spec['models'] = {'solo': dict(spec['models']['e4b'])}
    path = tmp_path / 'students.json'
    path.write_text(json.dumps(spec))
    assert students(load(path)) == ('solo',)
    cli.main(['--specification', str(path), 'students'])
    assert capsys.readouterr().out.split() == ['solo']
    with pytest.raises(SystemExit):
        cli.main(['--specification', str(path), 'create', '--student', 'e4b', *[
            arg for name in ('root', 'checkout', 'models', 'splits', 'mirror') for arg in ('--' + name, str(tmp_path))]])


@pytest.mark.parametrize('models', [{}, {'Bad Name': None}, {'one': 'e4b', 'two': 'e4b'}])
def test_student_names_and_directories_must_be_usable_identities(tmp_path, models):
    spec = load()
    spec['models'] = {name: dict(spec['models'][source or 'e4b']) for name, source in models.items()}
    path = tmp_path / 'students.json'
    path.write_text(json.dumps(spec))
    with pytest.raises(ValueError, match='student'):
        load(path)
