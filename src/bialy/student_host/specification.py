"""Pinned students and one context policy across training and serving."""
import hashlib
import json
import re
from pathlib import Path

DEFAULT = Path(__file__).resolve().parents[3] / 'config/students.json'


def load(path=DEFAULT):
    spec = json.loads(Path(path).read_text())
    if spec.get('schema') != 'pw-students/1' or set(spec.get('models', {})) != {'e4b', '12b'}:
        raise ValueError('student specification must contain e4b and 12b')
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


def identity(spec):
    return hashlib.sha256(json.dumps(spec, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def arguments(spec, student, model_root, splits, out, device, mirror, *, pilot=False, resume=False):
    model = spec['models'][student]
    args = ['--mode', 'sft', '--model', str(Path(model_root) / model['directory']),
            '--train', str(Path(splits) / 'train.jsonl'), '--validation', str(Path(splits) / 'val.jsonl'),
            '--holdout', str(Path(splits) / 'holdout.jsonl'), '--out', str(out), '--device', device,
            '--targets', model['targets'], '--max-length', str(spec['context']['max_sequence_tokens']),
            '--checkpoint-every', '5' if pilot else '50', '--checkpoint-mirror', str(mirror)]
    if pilot and not resume:
        args += ['--max-updates', '10']
    if resume:
        args += ['--resume']
    return args
