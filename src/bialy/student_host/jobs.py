"""Unique training jobs with typed retries and content-verified completion."""
import fcntl
import json
import subprocess
import time
import uuid
from pathlib import Path

from .. import host_contract
from ..provenance import file_sha256
from .resources import write
from .specification import arguments, identity, training_config

# The host's student pipeline, relative to its checkout.
PIPE = 'scripts/bialy/custom_model'
# How a job's trainer invocations are bounded. A checkpoint is written every
# `checkpoint_every` updates; `updates` ends the first invocation, which pauses, and the
# job's next attempt resumes it to the end. `rehearsal` is the shortest schedule that
# still writes a checkpoint, pauses and resumes: with one update an epoch, three updates.
SCHEDULES = {'full': {'checkpoint_every': 50, 'updates': None},
             'pilot': {'checkpoint_every': 5, 'updates': 10},
             'rehearsal': {'checkpoint_every': 1, 'updates': 1}}
# What a trainer invocation reports about itself, kept for every attempt of a job.
REPORT = ('step', 'total_steps', 'elapsed_seconds', 'median_update_seconds', 'observed_update_range_seconds',
          'peak_cuda_bytes', 'peak_process_rss_bytes')


class JobError(RuntimeError):
    pass


class JobBusy(JobError):
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


def checkpoint(spec, student, root):
    model = spec['models'][student]
    receipt = json.loads((Path(root) / (model['directory'] + '.source.json')).read_text())
    files = tree(Path(root) / model['directory'])
    if (receipt.get('schema') != 'pw-student-source/1'
            or receipt.get('source') != {'repository': model['repository'], 'revision': model['revision']}
            or receipt.get('files') != files):
        raise JobError('student checkpoint source or content changed')
    return files


def rendering(spec, checkout):
    """The profile and template a job trains with; refused unless the checkout carries the pinned template."""
    declared = spec['rendering']
    if file_sha256(Path(checkout) / declared['template']) != declared['template_sha256']:
        raise JobError('trainer checkout does not carry the pinned chat template')
    return {'profile': declared['profile'], 'template_sha256': declared['template_sha256']}


def create(root, spec, student, checkout, models, splits, mirror, device, schedule, *, name=None):
    """A new job directory holding its plan. A named job is created once: asking again
    for the same plan returns it, so a caller that lost the answer can ask again."""
    if schedule not in SCHEDULES:
        raise JobError('unknown training schedule ' + str(schedule))
    checkout, models, splits, mirror = map(lambda path: Path(path).resolve(), (checkout, models, splits, mirror))
    revision = subprocess.run(['git', '-C', str(checkout), 'rev-parse', 'HEAD'], capture_output=True, text=True, check=True).stdout.strip()
    dirty = subprocess.run(['git', '-C', str(checkout), 'status', '--porcelain'], capture_output=True, text=True, check=True).stdout
    if dirty:
        raise JobError('trainer checkout must be clean')
    directory = Path(root).resolve() / (name or student + '-' + uuid.uuid4().hex)
    plan = {'schema': 'pw-student-job/1', 'student': student, 'specification': spec, 'specification_sha256': identity(spec),
            'host_commit': revision, 'checkout': str(checkout), 'models': str(models), 'splits': str(splits),
            'rendering': rendering(spec, checkout),
            'dataset': {name: file_sha256(splits / name) for name in ('train.jsonl', 'val.jsonl', 'holdout.jsonl', 'source.json')},
            'model': checkpoint(spec, student, models), 'device': device, 'schedule': schedule,
            'mirror': str(mirror / directory.name), 'out': str(directory / 'training')}
    if name and (directory / 'plan.json').exists():
        if json.loads((directory / 'plan.json').read_text()) != plan:
            raise JobError('a job of this name exists with another plan')
        return directory
    directory.mkdir(parents=True, exist_ok=bool(name))
    write(directory / 'plan.json', plan)
    return directory


def validate_inputs(plan):
    if identity(plan['specification']) != plan['specification_sha256']:
        raise JobError('student specification changed')
    checkout = Path(plan['checkout'])
    revision = subprocess.run(['git', '-C', str(checkout), 'rev-parse', 'HEAD'], capture_output=True, text=True, check=True).stdout.strip()
    dirty = subprocess.run(['git', '-C', str(checkout), 'status', '--porcelain'], capture_output=True, text=True, check=True).stdout
    if dirty or revision != plan['host_commit'] or rendering(plan['specification'], checkout) != plan['rendering']:
        raise JobError('trainer source changed')
    if any(file_sha256(Path(plan['splits']) / name) != value for name, value in plan['dataset'].items()):
        raise JobError('dataset changed')
    if checkpoint(plan['specification'], plan['student'], plan['models']) != plan['model']:
        raise JobError('student checkpoint changed')


def validate_result(plan):
    root = Path(plan['out'])
    terminal = json.loads((root / 'terminal.json').read_text())
    result = json.loads((root / 'result.json').read_text())
    inputs = json.loads((root / 'inputs.json').read_text())
    validate_training_inputs(plan, inputs)
    if (terminal.get('schema') != 'pw-training-terminal/1' or terminal.get('status') != 'complete'
            or terminal.get('result_sha256') != file_sha256(root / 'result.json')
            or terminal.get('inputs_sha256') != identity(inputs) or result.get('inputs_sha256') != identity(inputs)
            or result.get('resources', {}).get('status') != 'complete'
            or result.get('adapter') != tree(root / 'adapter')):
        raise JobError('training terminal receipt or artifact identity mismatch')
    return terminal


def validate_training_inputs(plan, inputs):
    expected = training_config(plan['specification'], plan['student'], plan['device'])
    if inputs.get('model') != plan['model'] or any(inputs.get('config', {}).get(key) != value for key, value in expected.items()):
        raise JobError('training input identity mismatch')
    if (inputs.get('rendering_profile') != plan['rendering']['profile']
            or inputs.get('template_sha256') != plan['rendering']['template_sha256']):
        raise JobError('training rendered its rows with another template or profile')
    for key, name in (('train_sha256', 'train.jsonl'), ('validation_sha256', 'val.jsonl'), ('holdout_sha256', 'holdout.jsonl')):
        if inputs.get(key) != plan['dataset'][name]:
            raise JobError('training result used a different dataset')


def train(directory, run=subprocess.run, sleep=time.sleep):
    directory = Path(directory)
    with (directory / '.execution.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise JobBusy('another controller owns this training job') from exc
        try:
            return train_owned(directory, run, sleep)
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def settled(state, plan, terminal, exit_status):
    """The execution state with one more attempt's own report, and the job's totals over
    every attempt. A paused attempt reports in its terminal receipt, a complete one in the
    result; the peak of an earlier attempt is not in a later one's report."""
    report = terminal.get('progress') or {}
    if terminal.get('status') == 'complete':
        report = json.loads((Path(plan['out']) / 'result.json').read_text()).get('resources', {})
    history = [*state.get('history', []), {'attempt': state['attempts'], 'status': terminal.get('status'), 'exit_status': exit_status,
                                          **{name: report.get(name) for name in REPORT}}]

    def greatest(name):
        return max((item[name] for item in history if item[name] is not None), default=None)
    resources = {'peak_cuda_bytes': greatest('peak_cuda_bytes'), 'peak_process_rss_bytes': greatest('peak_process_rss_bytes'),
                 'elapsed_seconds': sum(item['elapsed_seconds'] or 0 for item in history)}
    return dict(state, history=history, resources=resources)


def train_owned(directory, run, sleep):
    plan = json.loads((directory / 'plan.json').read_text())
    validate_inputs(plan)
    pipe = Path(plan['checkout']) / PIPE
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
            write(state_path, dict(settled(state, plan, terminal, None), status='complete'))
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
        schedule = SCHEDULES[plan['schedule']]
        args = arguments(plan['specification'], plan['student'], plan['models'], plan['splits'], plan['out'], plan['device'],
                         plan['mirror'], checkpoint_every=schedule['checkpoint_every'], max_updates=schedule['updates'], resume=resume)
        state.update(status='running', attempts=state['attempts'] + 1)
        write(state_path, state)
        previous_terminal = file_sha256(terminal_path) if terminal_path.exists() else None
        with (directory / ('attempt-%d.log' % state['attempts'])).open('x') as log:
            result = run(['uv', 'run', '--project', str(pipe), '--locked', 'python',
                          *host_contract.script(plan['checkout'], PIPE + '/train.py', args)],
                         cwd=plan['checkout'], stdout=log, stderr=subprocess.STDOUT)
        if not terminal_path.exists():
            raise JobError('trainer did not publish a terminal receipt')
        if file_sha256(terminal_path) == previous_terminal:
            raise JobError('trainer did not publish a new terminal receipt')
        terminal = json.loads(terminal_path.read_text())
        if terminal.get('status') == 'running':
            raise JobError('trainer exit left an unsettled execution')
        if terminal.get('status') == 'complete' and result.returncode == 0:
            verified = validate_result(plan)
            write(state_path, dict(settled(state, plan, terminal, 0), status='complete'))
            return verified
        state = settled(state, plan, {} if terminal.get('status') == 'complete' else terminal, result.returncode)
        retryable = terminal.get('status') == 'paused' or terminal.get('failure', {}).get('retryable', False)
        state.update(status='retrying' if retryable and state['attempts'] < 3 else 'failed', terminal=terminal)
        write(state_path, state)
    raise JobError('typed transient retry allowance exhausted')
