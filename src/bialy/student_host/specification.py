"""Pinned students and one context policy across training and serving."""
import hashlib
import json
import re
from pathlib import Path

DEFAULT = Path(__file__).resolve().parents[3] / 'config/students.json'


def load(path=DEFAULT):
    spec = json.loads(Path(path).read_text())
    models = spec.get('models')
    # Names become stage names, provider ids, and directory components.
    if (spec.get('schema') != 'pw-students/1' or not isinstance(models, dict) or not models
            or any(not re.fullmatch(r'[a-z0-9][a-z0-9-]{0,31}', name) for name in models)):
        raise ValueError('student specification must name one or more students')
    if len({model.get('directory') for model in models.values()}) != len(models):
        raise ValueError('student model directories must be distinct')
    context = spec.get('context', {})
    if (type(context.get('max_sequence_tokens')) is not int or context['max_sequence_tokens'] < 4096
            or context.get('thinking') is not False or context.get('truncation') != 'forbidden'):
        raise ValueError('invalid student context policy')
    for model in spec['models'].values():
        if (not re.fullmatch(r'[0-9a-f]{40}', model['revision']) or not model['repository'].startswith('google/gemma-4-')
                or not re.fullmatch(r'[a-zA-Z0-9._-]+', model['directory']) or not model['targets'].startswith('re:')):
            raise ValueError('invalid student model identity')
        re.compile(model['targets'][3:])
    return spec


def students(spec=None):
    """Student names in their declared order."""
    return tuple((load() if spec is None else spec)['models'])


def identity(spec):
    return hashlib.sha256(json.dumps(spec, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def arguments(spec, student, model_root, splits, out, device, mirror, *, pilot=False, resume=False, dry_run=False):
    model = spec['models'][student]
    args = ['--mode', 'sft', '--model', str(Path(model_root) / model['directory']),
            '--train', str(Path(splits) / 'train.jsonl'), '--validation', str(Path(splits) / 'val.jsonl'),
            '--holdout', str(Path(splits) / 'holdout.jsonl'), '--out', str(out), '--device', device,
            '--targets', model['targets'], '--max-length', str(spec['context']['max_sequence_tokens']),
            '--checkpoint-every', '1' if dry_run else '5' if pilot else '50', '--checkpoint-mirror', str(mirror)]
    if (pilot or dry_run) and not resume:
        args += ['--max-updates', '1' if dry_run else '10']
    if resume:
        args += ['--resume']
    return args


def serving(spec=None):
    spec = load() if spec is None else spec
    return {name: {'provider': 'student-' + name, 'kind': 'openai-compatible',
                   'model': model['directory'], 'base_url': 'http://127.0.0.1:%d/v1' % (8080 + index),
                   'context_window': spec['context']['max_sequence_tokens']}
            for index, (name, model) in enumerate(spec['models'].items())}
