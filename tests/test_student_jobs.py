import hashlib
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from bialy.student_host import jobs
from bialy.student_host.resources import write
from bialy.student_host.specification import identity, load, training_config

TEMPLATE = 'training template\n'


def specification():
    """The declared students with the fixture checkout's template pinned."""
    spec = load()
    spec['rendering']['template_sha256'] = hashlib.sha256(TEMPLATE.encode()).hexdigest()
    return spec


def fixture(tmp_path):
    checkout = tmp_path / 'host'
    checkout.mkdir()
    template = checkout / load()['rendering']['template']
    template.parent.mkdir(parents=True)
    template.write_text(TEMPLATE)
    subprocess.run(['git', 'init', '-q', str(checkout)], check=True)
    subprocess.run(['git', '-C', str(checkout), 'add', '-A'], check=True)
    subprocess.run(['git', '-C', str(checkout), '-c', 'user.name=Fixture', '-c', 'user.email=fixture@example.invalid',
                    'commit', '-qm', 'fixture'], check=True)
    models, splits = tmp_path / 'models', tmp_path / 'splits'
    model = models / load()['models']['e4b']['directory']
    model.mkdir(parents=True)
    (model / 'config.json').write_text('{}')
    selected = load()['models']['e4b']
    write(models / (selected['directory'] + '.source.json'), {'schema': 'pw-student-source/1',
          'source': {'repository': selected['repository'], 'revision': selected['revision']}, 'files': jobs.tree(model)})
    splits.mkdir()
    for name in ('train.jsonl', 'val.jsonl', 'holdout.jsonl', 'source.json'):
        (splits / name).write_text(name)
    return checkout, models, splits


def test_new_jobs_preserve_existing_directories_and_freeze_input_bytes(tmp_path):
    checkout, models, splits = fixture(tmp_path)
    first = jobs.create(tmp_path / 'runs', specification(), 'e4b', checkout, models, splits, tmp_path / 'mirror', 'cpu', 'pilot')
    second = jobs.create(tmp_path / 'runs', specification(), 'e4b', checkout, models, splits, tmp_path / 'mirror', 'cpu', 'pilot')
    assert first != second and first.exists()
    plan = json.loads((first / 'plan.json').read_text())
    jobs.validate_inputs(plan)
    (splits / 'train.jsonl').write_text('changed')
    with pytest.raises(jobs.JobError, match='dataset changed'):
        jobs.validate_inputs(plan)


def test_oom_never_blindly_retries(tmp_path):
    checkout, models, splits = fixture(tmp_path)
    directory = jobs.create(tmp_path / 'runs', specification(), 'e4b', checkout, models, splits, tmp_path / 'mirror', 'cpu', 'full')
    plan = json.loads((directory / 'plan.json').read_text())
    calls = []
    def run(args, **kwargs):
        calls.append(args)
        out = Path(plan['out'])
        out.mkdir()
        write(out / 'terminal.json', {'status': 'failed', 'failure': {'code': 'out_of_memory', 'retryable': False}})
        return SimpleNamespace(returncode=1)
    with pytest.raises(jobs.JobError, match='permanently'):
        jobs.train(directory, run=run, sleep=lambda _: None)
    assert len(calls) == 1


def test_a_missing_result_does_not_trigger_another_launch(tmp_path):
    checkout, models, splits = fixture(tmp_path)
    directory = jobs.create(tmp_path / 'runs', specification(), 'e4b', checkout, models, splits, tmp_path / 'mirror', 'cpu', 'full')
    calls = []
    def run(args, **kwargs):
        calls.append(args)
        return SimpleNamespace(returncode=1)
    with pytest.raises(jobs.JobError, match='terminal receipt'):
        jobs.train(directory, run=run)
    with pytest.raises(jobs.JobError, match='outcome unknown'):
        jobs.train(directory, run=run)
    assert len(calls) == 1


def test_complete_receipt_must_bind_dataset_and_adapter(tmp_path):
    checkout, models, splits = fixture(tmp_path)
    directory = jobs.create(tmp_path / 'runs', specification(), 'e4b', checkout, models, splits, tmp_path / 'mirror', 'cpu', 'full')
    plan = json.loads((directory / 'plan.json').read_text())
    root = Path(plan['out'])
    (root / 'adapter').mkdir(parents=True)
    (root / 'adapter/weights').write_text('weights')
    inputs = {'model': plan['model'], 'config': {**training_config(specification(), 'e4b', 'cpu'), 'beta': 0.1, 'anchor': 0.0},
              'rendering_profile': 'gemma4-code-v1', 'template_sha256': plan['rendering']['template_sha256'],
              'train_sha256': plan['dataset']['train.jsonl'], 'validation_sha256': plan['dataset']['val.jsonl'],
              'holdout_sha256': plan['dataset']['holdout.jsonl']}
    write(root / 'inputs.json', inputs)
    write(root / 'result.json', {'inputs_sha256': identity(inputs), 'adapter': jobs.tree(root / 'adapter'),
                                 'resources': {'status': 'complete'}})
    write(root / 'terminal.json', {'schema': 'pw-training-terminal/1', 'status': 'complete',
                                   'result_sha256': jobs.file_sha256(root / 'result.json'), 'inputs_sha256': identity(inputs)})
    assert jobs.validate_result(plan)['status'] == 'complete'
    (root / 'adapter/weights').write_text('changed')
    with pytest.raises(jobs.JobError, match='identity mismatch'):
        jobs.validate_result(plan)
    for change, reason in (({'template_sha256': '0' * 64}, 'another template or profile'),
                           ({'rendering_profile': 'checkpoint'}, 'another template or profile'),
                           ({'config': {**inputs['config'], 'learning_rate': 2e-5}}, 'input identity mismatch'),
                           ({'config': {**inputs['config'], 'accumulation': 1}}, 'input identity mismatch'),
                           ({'config': {**inputs['config'], 'adapted_modules': 1}}, 'input identity mismatch')):
        with pytest.raises(jobs.JobError, match=reason):
            jobs.validate_training_inputs(plan, {**inputs, **change})


def test_a_job_binds_the_pinned_template_and_refuses_a_checkout_with_another(tmp_path):
    checkout, models, splits = fixture(tmp_path)
    directory = jobs.create(tmp_path / 'runs', specification(), 'e4b', checkout, models, splits, tmp_path / 'mirror', 'cpu', 'full')
    plan = json.loads((directory / 'plan.json').read_text())
    assert plan['rendering'] == {'profile': 'gemma4-code-v1', 'template_sha256': hashlib.sha256(TEMPLATE.encode()).hexdigest()}
    with pytest.raises(jobs.JobError, match='pinned chat template'):
        jobs.create(tmp_path / 'runs', load(), 'e4b', checkout, models, splits, tmp_path / 'mirror', 'cpu', 'full')
    plan['rendering']['template_sha256'] = '0' * 64
    with pytest.raises(jobs.JobError, match='trainer source changed'):
        jobs.validate_inputs(plan)


def test_retry_requires_a_new_terminal_receipt(tmp_path):
    checkout, models, splits = fixture(tmp_path)
    directory = jobs.create(tmp_path / 'runs', specification(), 'e4b', checkout, models, splits, tmp_path / 'mirror', 'cpu', 'full')
    out = Path(json.loads((directory / 'plan.json').read_text())['out'])
    calls = []
    def run(args, **kwargs):
        calls.append(args)
        if len(calls) == 1:
            out.mkdir()
            write(out / 'terminal.json', {'status': 'failed', 'failure': {'code': 'transport_unavailable', 'retryable': True}})
        return SimpleNamespace(returncode=1)
    with pytest.raises(jobs.JobError, match='new terminal receipt'):
        jobs.train(directory, run=run, sleep=lambda _: None)
    assert len(calls) == 2
    assert '--resume' in calls[1]


def complete(plan, resources):
    """What the trainer leaves when an invocation ends the run."""
    root = Path(plan['out'])
    (root / 'adapter').mkdir(parents=True, exist_ok=True)
    (root / 'adapter/weights').write_text('weights')
    inputs = {'model': plan['model'], 'config': training_config(plan['specification'], 'e4b', 'cpu'),
              'rendering_profile': 'gemma4-code-v1', 'template_sha256': plan['rendering']['template_sha256'],
              'train_sha256': plan['dataset']['train.jsonl'], 'validation_sha256': plan['dataset']['val.jsonl'],
              'holdout_sha256': plan['dataset']['holdout.jsonl']}
    write(root / 'inputs.json', inputs)
    write(root / 'result.json', {'inputs_sha256': identity(inputs), 'adapter': jobs.tree(root / 'adapter'),
                                 'resources': {'status': 'complete', **resources}})
    write(root / 'terminal.json', {'schema': 'pw-training-terminal/1', 'status': 'complete', 'attempt_id': 'second',
                                   'result_sha256': jobs.file_sha256(root / 'result.json'), 'inputs_sha256': identity(inputs)})


@pytest.mark.parametrize('schedule, first, resumed', [
    ('full', ['--checkpoint-every', '50'], None), ('pilot', ['--checkpoint-every', '5', '--max-updates', '10'], ['--checkpoint-every', '5']),
    ('rehearsal', ['--checkpoint-every', '1', '--max-updates', '1'], ['--checkpoint-every', '1'])])
def test_a_schedule_bounds_the_first_invocation_and_the_resumed_one_runs_to_the_end(tmp_path, schedule, first, resumed):
    checkout, models, splits = fixture(tmp_path)
    directory = jobs.create(tmp_path / 'runs', specification(), 'e4b', checkout, models, splits, tmp_path / 'mirror', 'cpu', schedule)
    plan = json.loads((directory / 'plan.json').read_text())
    calls = []

    def run(args, **kwargs):
        calls.append(args)
        if len(calls) == 1 and resumed:
            Path(plan['out']).mkdir()
            write(Path(plan['out']) / 'terminal.json', {'status': 'paused', 'progress': {'status': 'paused', 'step': 1}})
        else:
            complete(plan, {})
        return SimpleNamespace(returncode=0)
    jobs.train(directory, run=run, sleep=lambda _: None)

    def bounds(args):
        return [item for flag in ('--checkpoint-every', '--max-updates') if flag in args for item in (flag, args[args.index(flag) + 1])]
    assert bounds(calls[0]) == first and '--resume' not in calls[0]
    assert len(calls) == (2 if resumed else 1)
    if resumed:
        assert bounds(calls[1]) == resumed and '--resume' in calls[1]
    with pytest.raises(jobs.JobError, match='unknown training schedule'):
        jobs.create(tmp_path / 'runs', specification(), 'e4b', checkout, models, splits, tmp_path / 'mirror', 'cpu', 'two-updates')


def test_the_job_keeps_every_attempts_report_and_the_greatest_peak_among_them(tmp_path):
    checkout, models, splits = fixture(tmp_path)
    directory = jobs.create(tmp_path / 'runs', specification(), 'e4b', checkout, models, splits, tmp_path / 'mirror', 'cpu', 'rehearsal')
    plan = json.loads((directory / 'plan.json').read_text())
    calls = []

    def run(args, **kwargs):
        calls.append(args)
        if len(calls) == 1:
            # The attempt that trained the longest row holds the peak; the resumed one does not.
            Path(plan['out']).mkdir()
            write(Path(plan['out']) / 'terminal.json', {'status': 'paused', 'progress': {
                'status': 'paused', 'step': 1, 'total_steps': 3, 'elapsed_seconds': 400.0, 'peak_cuda_bytes': 119460112896,
                'peak_process_rss_bytes': 52757864448, 'observed_update_range_seconds': [0.3, 162.0]}})
        else:
            complete(plan, {'step': 3, 'total_steps': 3, 'elapsed_seconds': 211.0, 'peak_cuda_bytes': 80000000000,
                            'peak_process_rss_bytes': 52751720448})
        return SimpleNamespace(returncode=0)
    jobs.train(directory, run=run, sleep=lambda _: None)
    state = json.loads((directory / 'execution.json').read_text())
    assert state['status'] == 'complete' and state['attempts'] == 2
    assert [(item['attempt'], item['status'], item['step'], item['peak_cuda_bytes']) for item in state['history']] == [
        (1, 'paused', 1, 119460112896), (2, 'complete', 3, 80000000000)]
    assert state['history'][0]['observed_update_range_seconds'] == [0.3, 162.0]
    assert state['resources'] == {'peak_cuda_bytes': 119460112896, 'peak_process_rss_bytes': 52757864448, 'elapsed_seconds': 611.0}


def test_a_failed_attempt_keeps_the_reports_of_the_attempts_before_it(tmp_path):
    checkout, models, splits = fixture(tmp_path)
    directory = jobs.create(tmp_path / 'runs', specification(), 'e4b', checkout, models, splits, tmp_path / 'mirror', 'cpu', 'rehearsal')
    out = Path(json.loads((directory / 'plan.json').read_text())['out'])
    calls = []

    def run(args, **kwargs):
        calls.append(args)
        out.mkdir(exist_ok=True)
        write(out / 'terminal.json', {'status': 'paused', 'progress': {'peak_cuda_bytes': 7}} if len(calls) == 1 else
              {'status': 'failed', 'failure': {'code': 'out_of_memory', 'retryable': False}})
        return SimpleNamespace(returncode=len(calls) - 1)
    with pytest.raises(jobs.JobError, match='permanently'):
        jobs.train(directory, run=run, sleep=lambda _: None)
    state = json.loads((directory / 'execution.json').read_text())
    assert state['status'] == 'failed' and [item['status'] for item in state['history']] == ['paused', 'failed']
    assert state['resources']['peak_cuda_bytes'] == 7 and state['terminal']['failure']['code'] == 'out_of_memory'


def test_a_named_job_is_created_once_and_asked_for_again_is_the_same_job(tmp_path):
    checkout, models, splits = fixture(tmp_path)
    arguments = (tmp_path / 'runs', specification(), 'e4b', checkout, models, splits, tmp_path / 'mirror', 'cpu')
    first = jobs.create(*arguments, 'rehearsal', name='e4b-controller-run')
    assert first.name == 'e4b-controller-run' and jobs.create(*arguments, 'rehearsal', name='e4b-controller-run') == first
    assert len(list((tmp_path / 'runs').iterdir())) == 1
    with pytest.raises(jobs.JobError, match='another plan'):
        jobs.create(*arguments, 'pilot', name='e4b-controller-run')
