"""Pinned students, one context policy, one rendering profile and one training recipe."""
import hashlib
import json
import re
from pathlib import Path

DEFAULT = Path(__file__).resolve().parents[3] / 'config/students.json'
REQUIREMENTS = DEFAULT.with_name('student-requirements.json')
GIB = 1 << 30
# Recipe values by the trainer flag that carries each, with the type the trainer records.
RECIPE = {'targets': str, 'rank': int, 'lora_alpha': int, 'lora_dropout': float, 'epochs': int, 'seed': int,
          'learning_rate': float, 'warmup_ratio': float, 'final_learning_rate_ratio': float, 'accumulation': int,
          'weight_decay': float, 'max_grad_norm': float}
ARCHITECTURE = ('parameters', 'layers', 'hidden_size', 'mlp_width', 'vocabulary')
# Longest completion the memory estimate allows for, in tokens.
COMPLETION_TOKENS = 2048
# Share of the required GPU memory the estimate may reach.
HEADROOM = 0.85


def positive_integers(value, names):
    return isinstance(value, dict) and set(value) == set(names) and all(type(value[name]) is int and value[name] > 0 for name in names)


def load(path=DEFAULT, requirements=REQUIREMENTS):
    spec = json.loads(Path(path).read_text())
    machines = json.loads(Path(requirements).read_text())['tiers']
    models = spec.get('models')
    # Names become stage names, provider ids, and directory components.
    if (spec.get('schema') != 'pw-students/2' or not isinstance(models, dict) or not models
            or any(not re.fullmatch(r'[a-z0-9][a-z0-9-]{0,31}', name) for name in models)):
        raise ValueError('student specification must name one or more students')
    if len({model.get('directory') for model in models.values()}) != len(models):
        raise ValueError('student model directories must be distinct')
    context = spec.get('context', {})
    if (type(context.get('max_sequence_tokens')) is not int or context['max_sequence_tokens'] < 4096
            or context.get('thinking') is not False or context.get('truncation') != 'forbidden'):
        raise ValueError('invalid student context policy')
    rendering = spec.get('rendering', {})
    if (set(rendering) != {'profile', 'template', 'template_sha256'} or not rendering['profile']
            or Path(rendering['template']).is_absolute() or not re.fullmatch(r'[0-9a-f]{64}', rendering['template_sha256'])):
        raise ValueError('invalid student rendering profile')
    recipe = spec.get('recipe', {})
    if (set(recipe) != {'version', *RECIPE} or type(recipe['version']) is not int
            or any(type(recipe[name]) is not kind for name, kind in RECIPE.items())
            or not recipe['targets'].startswith('re:') or min(recipe['rank'], recipe['lora_alpha'], recipe['epochs'], recipe['accumulation']) < 1
            or not 0 <= recipe['lora_dropout'] < 1 or recipe['learning_rate'] <= 0 or not 0 <= recipe['warmup_ratio'] < 1
            or not 0 <= recipe['final_learning_rate_ratio'] <= 1 or recipe['weight_decay'] < 0 or recipe['max_grad_norm'] <= 0):
        raise ValueError('invalid student training recipe')
    re.compile(recipe['targets'][3:])
    for name, model in models.items():
        if (not re.fullmatch(r'[0-9a-f]{40}', model['revision']) or not model['repository'].startswith('google/gemma-4-')
                or not re.fullmatch(r'[a-zA-Z0-9._-]+', model['directory'])
                or type(model.get('adapted_modules')) is not int or model['adapted_modules'] < 1
                or not positive_integers(model.get('architecture'), ARCHITECTURE)):
            raise ValueError('invalid student model identity')
        # A tier absent from the requirements is refused where a machine is chosen for it.
        if name in machines and training_memory(spec, name) > HEADROOM * machines[name]['gpu_memory_gib'] * GIB:
            raise ValueError('student %s does not fit the GPU memory required for it at the declared sequence length' % name)
    return spec


def training_memory(spec, student):
    """Estimated peak GPU bytes of one update on a row of the declared maximum length, in
    bfloat16 with gradient checkpointing and without a materialized attention matrix:
    the weights, one checkpointed hidden state per layer, the sliding-window mask and its
    additive form, the live activations of the layer being differentiated, and the
    logits, log-probabilities and their gradient at every completion position."""
    shape, tokens = spec['models'][student]['architecture'], spec['context']['max_sequence_tokens']
    weights = 2 * shape['parameters']
    checkpoints = 2 * shape['layers'] * tokens * shape['hidden_size']
    masks = 3 * tokens * tokens
    layer = 2 * tokens * (8 * shape['hidden_size'] + 3 * shape['mlp_width'])
    logits = 10 * COMPLETION_TOKENS * shape['vocabulary']
    return weights + checkpoints + masks + layer + logits


def students(spec=None):
    """Student names in their declared order."""
    return tuple((load() if spec is None else spec)['models'])


def identity(spec):
    return hashlib.sha256(json.dumps(spec, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def training_config(spec, student, device):
    """What a trainer run under this specification records as its configuration."""
    return {'mode': 'sft', 'device': device, 'max_length': spec['context']['max_sequence_tokens'],
            'adapted_modules': spec['models'][student]['adapted_modules'],
            **{name: spec['recipe'][name] for name in RECIPE}}


def arguments(spec, student, model_root, splits, out, device, mirror, *, checkpoint_every, max_updates=None, resume=False):
    """The trainer's arguments. `max_updates` bounds one invocation; a resumed one runs to the end."""
    model = spec['models'][student]
    args = ['--model', str(Path(model_root) / model['directory']),
            '--train', str(Path(splits) / 'train.jsonl'), '--validation', str(Path(splits) / 'val.jsonl'),
            '--holdout', str(Path(splits) / 'holdout.jsonl'), '--out', str(out),
            '--checkpoint-every', str(checkpoint_every), '--checkpoint-mirror', str(mirror)]
    for name, value in training_config(spec, student, device).items():
        args += ['--' + name.replace('_', '-'), str(value)]
    if max_updates and not resume:
        args += ['--max-updates', str(max_updates)]
    if resume:
        args += ['--resume']
    return args


def serving(spec=None):
    spec = load() if spec is None else spec
    return {name: {'provider': 'student-' + name, 'kind': 'openai-compatible',
                   'model': model['directory'], 'base_url': 'http://127.0.0.1:%d/v1' % (8080 + index),
                   'context_window': spec['context']['max_sequence_tokens']}
            for index, (name, model) in enumerate(spec['models'].items())}
