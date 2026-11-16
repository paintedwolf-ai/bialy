import json

import pytest

from bialy.student_host import __main__ as cli
from bialy.student_host.specification import GIB, HEADROOM, REQUIREMENTS, arguments, load, students, training_config, training_memory


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


def test_one_recipe_reaches_the_trainer_whole_and_is_what_a_run_must_record():
    spec = load()
    for student in students(spec):
        args = arguments(spec, student, '/models', '/splits', '/run', 'cuda:0', '/mirror')
        passed = dict(zip(args[::2], args[1::2], strict=True))
        expected = training_config(spec, student, 'cuda:0')
        assert {'--' + name.replace('_', '-'): str(value) for name, value in expected.items()}.items() <= passed.items()
        assert set(expected) == {'mode', 'device', 'max_length', 'adapted_modules', *spec['recipe']} - {'version'}
        assert expected['adapted_modules'] == spec['models'][student]['adapted_modules']
    recipe = spec['recipe']
    assert recipe['accumulation'] > 1 and recipe['epochs'] > 1 and 0 < recipe['warmup_ratio'] < 0.2
    assert recipe['final_learning_rate_ratio'] < 1 and recipe['learning_rate'] >= 5e-5


def test_every_tier_adapts_only_language_model_attention_projections():
    import re
    pattern = re.compile(load()['recipe']['targets'][3:])
    assert pattern.fullmatch('model.language_model.layers.41.self_attn.q_proj')
    assert pattern.fullmatch('model.language_model.layers.0.self_attn.o_proj')
    for outside in ('model.vision_tower.encoder.layers.0.self_attn.q_proj', 'model.audio_tower.layers.0.self_attn.k_proj',
                    'model.language_model.layers.0.self_attn.q_proj.linear', 'model.language_model.layers.0.mlp.down_proj'):
        assert not pattern.fullmatch(outside)


@pytest.mark.parametrize('change', [
    {'recipe': {'learning_rate': 0}}, {'recipe': {'accumulation': 0}}, {'recipe': {'warmup_ratio': 1.0}},
    {'recipe': {'targets': 'q_proj'}}, {'recipe': {'rank': 16.0}}, {'recipe': {'momentum': 0.9}},
    {'rendering': {'template_sha256': 'unpinned'}}, {'rendering': {'template': '/etc/template.jinja'}},
])
def test_an_incomplete_recipe_or_rendering_profile_is_refused(tmp_path, change):
    spec = load()
    for section, values in change.items():
        spec[section].update(values)
    path = tmp_path / 'students.json'
    path.write_text(json.dumps(spec))
    with pytest.raises(ValueError, match='recipe|rendering'):
        load(path)


def test_each_tier_fits_the_gpu_memory_required_for_it_at_the_declared_sequence_length(tmp_path):
    spec = load()
    required = json.loads(REQUIREMENTS.read_text())
    estimates = {student: training_memory(spec, student) / GIB for student in students(spec)}
    assert 50 < estimates['e4b'] < 55 and 69 < estimates['12b'] < 73 and 79 < estimates['26b-a4b'] < 83
    for student, estimate in estimates.items():
        assert estimate <= HEADROOM * required['tiers'][student]['gpu_memory_gib']
    required['tiers']['26b-a4b']['gpu_memory_gib'] = 79
    path = tmp_path / 'requirements.json'
    path.write_text(json.dumps(required))
    with pytest.raises(ValueError, match='26b-a4b does not fit the GPU memory required'):
        load(requirements=path)
