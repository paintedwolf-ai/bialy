"""Unique training jobs with typed retries and content-verified completion."""
import fcntl
import json
import subprocess
import time
import uuid
from pathlib import Path

from ..provenance import file_sha256
from .resources import write
from .specification import arguments, identity


class JobError(RuntimeError):
    pass


def tree(path):
    path = Path(path)
    files = list(path.rglob('*'))
    if any(item.is_symlink() for item in files):
        raise JobError('artifact contains a symbolic link')
    result = {str(item.relative_to(path)): file_sha256(item) for item in files if item.is_file()}
    if not result:
        raise JobError('artifact is empty')
    return result


def create(root, spec, student, checkout, models, splits, mirror, device, pilot):
    checkout, models, splits, mirror = map(lambda path: Path(path).resolve(), (checkout, models, splits, mirror))
    revision = subprocess.run(['git', '-C', str(checkout), 'rev-parse', 'HEAD'], capture_output=True, text=True, check=True).stdout.strip()
    dirty = subprocess.run(['git', '-C', str(checkout), 'status', '--porcelain'], capture_output=True, text=True, check=True).stdout
    if dirty:
        raise JobError('trainer checkout must be clean')
    directory = Path(root).resolve() / (student + '-' + uuid.uuid4().hex)
    directory.mkdir(parents=True, exist_ok=False)
    plan = {'schema': 'pw-student-job/1', 'student': student, 'specification': spec, 'specification_sha256': identity(spec),
            'host_commit': revision, 'checkout': str(checkout), 'models': str(models), 'splits': str(splits),
            'dataset': {name: file_sha256(splits / name) for name in ('train.jsonl', 'val.jsonl', 'holdout.jsonl')},
            'model': tree(models / spec['models'][student]['directory']), 'device': device, 'pilot': pilot,
            'mirror': str(mirror / directory.name), 'out': str(directory / 'training')}
    write(directory / 'plan.json', plan)
    return directory


def validate_inputs(plan):
    if identity(plan['specification']) != plan['specification_sha256']:
        raise JobError('student specification changed')
    checkout = Path(plan['checkout'])
    revision = subprocess.run(['git', '-C', str(checkout), 'rev-parse', 'HEAD'], capture_output=True, text=True, check=True).stdout.strip()
    dirty = subprocess.run(['git', '-C', str(checkout), 'status', '--porcelain'], capture_output=True, text=True, check=True).stdout
    if dirty or revision != plan['host_commit']:
        raise JobError('trainer source changed')
    if any(file_sha256(Path(plan['splits']) / name) != value for name, value in plan['dataset'].items()):
        raise JobError('dataset changed')
    model_path = Path(plan['models']) / plan['specification']['models'][plan['student']]['directory']
    if tree(model_path) != plan['model']:
        raise JobError('student checkpoint changed')


def validate_result(plan):
    root = Path(plan['out'])
    terminal = json.loads((root / 'terminal.json').read_text())
    result = json.loads((root / 'result.json').read_text())
    inputs = json.loads((root / 'inputs.json').read_text())
    if (terminal.get('schema') != 'pw-training-terminal/1' or terminal.get('status') != 'complete'
            or terminal.get('result_sha256') != file_sha256(root / 'result.json')
            or terminal.get('inputs_sha256') != identity(inputs) or result.get('inputs_sha256') != identity(inputs)
            or result.get('resources', {}).get('status') != 'complete'
            or result.get('adapter') != tree(root / 'adapter')
            or inputs.get('model') != plan['model']
            or inputs.get('config', {}).get('max_length') != plan['specification']['context']['max_sequence_tokens']):
        raise JobError('training terminal receipt or artifact identity mismatch')
    for key, name in (('train_sha256', 'train.jsonl'), ('validation_sha256', 'val.jsonl'), ('holdout_sha256', 'holdout.jsonl')):
        if inputs.get(key) != plan['dataset'][name]:
            raise JobError('training result used a different dataset')
    return terminal


def train(directory, run=subprocess.run, sleep=time.sleep):
    directory = Path(directory)
    with (directory / '.execution.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise JobError('another controller owns this training job') from exc
        try:
            return train_owned(directory, run, sleep)
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def train_owned(directory, run, sleep):
    plan = json.loads((directory / 'plan.json').read_text())
    validate_inputs(plan)
    pipe = Path(plan['checkout']) / 'scripts/bialy/custom_model'
    state_path = directory / 'execution.json'
    state = json.loads(state_path.read_text()) if state_path.exists() else {'attempts': 0, 'status': 'pending'}
    terminal_path = Path(plan['out']) / 'terminal.json'
    if state['status'] == 'complete':
        return validate_result(plan)
    if state['status'] == 'running':
        if not terminal_path.exists():
            raise JobError('training execution outcome unknown; reconcile the owned process')
        terminal = json.loads(terminal_path.read_text())
        if terminal.get('status') == 'complete':
            result = validate_result(plan)
            write(state_path, dict(state, status='complete'))
            return result
        if terminal.get('status') == 'running':
            raise JobError('training process has not settled')
    while state['attempts'] < 3:
        resume = state['attempts'] > 0
        if resume:
            terminal = json.loads(terminal_path.read_text())
            if terminal.get('status') != 'paused' and not terminal.get('failure', {}).get('retryable', False):
                raise JobError('training recipe failed permanently')
            sleep(2 ** state['attempts'])
        args = arguments(plan['specification'], plan['student'], plan['models'], plan['splits'],
                         plan['out'], plan['device'], plan['mirror'], pilot=plan['pilot'], resume=resume)
        state.update(status='running', attempts=state['attempts'] + 1)
        write(state_path, state)
        previous_terminal = file_sha256(terminal_path) if terminal_path.exists() else None
        with (directory / ('attempt-%d.log' % state['attempts'])).open('x') as log:
            result = run(['uv', 'run', '--project', str(pipe), '--locked', 'python', str(pipe / 'train.py'), *args],
                         cwd=plan['checkout'], stdout=log, stderr=subprocess.STDOUT)
        if not terminal_path.exists():
            raise JobError('trainer did not publish a terminal receipt')
        if file_sha256(terminal_path) == previous_terminal:
            raise JobError('trainer did not publish a new terminal receipt')
        terminal = json.loads(terminal_path.read_text())
        if terminal.get('status') == 'complete' and result.returncode == 0:
            verified = validate_result(plan)
            write(state_path, dict(state, status='complete'))
            return verified
        if terminal.get('status') == 'running':
            raise JobError('trainer exit left an unsettled execution')
        retryable = terminal.get('status') == 'paused' or terminal.get('failure', {}).get('retryable', False)
        write(state_path, dict(state, status='retrying' if retryable and state['attempts'] < 3 else 'failed', terminal=terminal))
    raise JobError('typed transient retry allowance exhausted')
