import json
import shlex
import time

import pytest
from rental_accounts import ACCOUNTS, Died, Machine, configuration

from bialy.student_host import machines, remote_training, status
from bialy.student_host.jobs import JobError, tree
from bialy.student_host.resources import Resources, write
from bialy.student_host.specification import identity


def job(tmp_path, provider='scaleway'):
    config = configuration(tmp_path, provider)
    if provider is None:
        config.pop('provisioning')
        config['machine'] = {'host': 'training.example.org', 'user': 'user', 'workspace': '/workspace/jobs', 'uv': '/home/user/.local/bin/uv'}
    else:
        # The larger tiers need more than the example card.
        config['provisioning']['specification']['gpu_memory_gib'] = 140
    config_path = tmp_path / 'configuration.json'
    write(config_path, config)
    splits = tmp_path / 'splits'
    splits.mkdir()
    (splits / 'train.jsonl').write_text('fixture')
    return config, config_path, splits


def train(config_path, student, splits, tmp_path, **options):
    return remote_training.train(config_path, student, splits, 'unused', tmp_path / 'instances', sleep=lambda _: None, **options)


def run_directory(tmp_path):
    return next((tmp_path / 'instances').iterdir())


def outcome_of(tmp_path):
    return json.loads((run_directory(tmp_path) / 'outcome.json').read_text())


class Worker(Machine):
    """A rented machine that runs the setup commands, then reports the given execution
    states in turn and ends its export on the given observations. `die` kills the
    controller at one command, before the machine runs it or just after."""
    states, exports, actions, die, started = [{'status': 'failed'}], [], [], None, False

    @classmethod
    def reset(cls):
        cls.states, cls.exports, cls.actions, cls.die, cls.started = [{'status': 'failed'}], [{'exit': 0, 'alive': False}], [], None, False

    def command(self, args, **kwargs):
        kind = type(self)
        if args[0] != machines.DEADMAN and args != ['true']:
            index = sum(action[0] == 'command' for action in kind.actions)
            if kind.die == (index, 'before'):
                raise Died()
            kind.actions.append(('command', args))
            result = self.answer(args)
            if kind.die == (index, 'after'):
                raise Died()
            return result
        return super().command(args, **kwargs)

    def answer(self, args):
        kind = type(self)
        if args[:2] == ['python3', '-c'] and 'export.exit' in args[2]:
            return json.dumps(kind.exports.pop(0) if len(kind.exports) > 1 else kind.exports[0])
        if args[:2] == ['python3', '-c']:
            state = kind.states.pop(0) if len(kind.states) > 1 else kind.states[0]
            if callable(state):
                state()
            if isinstance(state, Exception):
                raise state
            return json.dumps({'execution': state, 'heartbeat_age': 0})
        if 'create' in args:
            kind.actions.append(('create', args[args.index('--schedule') + 1]))
            return json.dumps({'directory': args[args.index('--root') + 1] + '/' + args[args.index('--name') + 1]})
        if args[:2] == ['bash', '-c'] and 'heartbeat.json' in args[2]:
            return 'started\n' if kind.started else 'absent\n'
        if args[:2] == ['bash', '-lc'] and 'export.exit' in args[2]:
            kind.actions.append(('export',))
            return '4242\n'
        if args[:2] == ['bash', '-lc']:
            kind.actions.append(('launch',))
            kind.started = True
        return ''

    def upload(self, *args):
        pass

    def receive(self, source, destination, exclude=(), timeout=600, limit=None):
        type(self).actions.append(('receive', source, tuple(exclude), limit))
        if source.endswith('/') and destination.name == 'job':
            destination.mkdir(parents=True, exist_ok=True)
            (destination / 'plan.json').write_text('{}')


def did(kind):
    return [action for action in Worker.actions if action[0] == kind]


@pytest.fixture
def rented(tmp_path, monkeypatch):
    """A provider account and worker behind `remote_training.train`, with a healthy reaper."""
    def account(provider):
        account = ACCOUNTS[provider]()
        monkeypatch.setattr(machines, 'provider', lambda provision: account.client())
        monkeypatch.setattr(status, 'provider', lambda provision: account.client())
        Resources(tmp_path / 'registry', account.client()).reap()
        return account
    Worker.reset()
    monkeypatch.setattr(remote_training, 'Remote', Worker)
    monkeypatch.setattr(remote_training, 'collect_checkpoint', lambda *args: [])
    return account


def killed(monkeypatch, config_path, splits, tmp_path, **options):
    """A controller that dies: no release and no cleanup of its own runs."""
    with monkeypatch.context() as dead, pytest.raises(Died):
        dead.setattr(machines, 'release', lambda *args: None)
        train(config_path, 'e4b', splits, tmp_path, **options)


def test_restart_observes_existing_worker_without_launching_or_renting_again(tmp_path, rented, monkeypatch):
    account = rented('vastai')
    config, config_path, splits = job(tmp_path, 'vastai')

    def dies():
        raise Died()
    Worker.states = [dies]
    killed(monkeypatch, config_path, splits, tmp_path)
    directory = run_directory(tmp_path)
    assert outcome_of(tmp_path)['status'] == 'running'
    assert len(account.owned()) == 1 and did('create') == [('create', 'full')] and len(did('launch')) == 1
    Worker.states = [{'status': 'failed', 'failure': {'code': 'out_of_memory'}}]
    with pytest.raises(JobError, match='did not complete'):
        train(config_path, 'e4b', splits, tmp_path, resume=directory)
    outcome = outcome_of(tmp_path)
    assert len(did('create')) == 1 and len(did('launch')) == 1
    assert [action[1] for action in did('receive')] == ['/workspace/training/%s/runs/%s/' % (directory.name, directory.name)]
    assert outcome['status'] == 'failed' and outcome['cleanup'] == 'confirmed'
    assert account.created == 1 and account.owned() == []


@pytest.mark.parametrize('moment', ['before', 'after'])
def test_a_controller_lost_at_any_command_of_setup_resumes_on_the_same_machine_and_launches_once(tmp_path, rented, monkeypatch, moment):
    account = rented('vastai')
    config, config_path, splits = job(tmp_path, 'vastai')
    with pytest.raises(JobError, match='did not complete'):
        train(config_path, 'e4b', splits, tmp_path)
    commands = [action[1] for action in did('command')]
    setup = next(index for index, args in enumerate(commands) if args[:2] == ['python3', '-c'])
    assert setup > 15 and account.created == 1
    for index in range(setup):
        root = tmp_path / str(index)
        root.mkdir()
        Worker.reset()
        Worker.die = (index, moment)
        killed(monkeypatch, config_path, splits, root)
        directory = run_directory(root)
        assert not (directory / 'remote.json').exists() or len(did('create')) == 1
        Worker.die = None
        with pytest.raises(JobError, match='did not complete'):
            train(config_path, 'e4b', splits, root, resume=directory)
        where = '%s command %d %r' % (moment, index, commands[index][:6])
        assert len(did('launch')) == 1, 'the trainer was launched %d times: %s' % (len(did('launch')), where)
        assert {action[1] for action in did('create')} == {'full'} and len(did('create')) <= 2, where
        assert json.loads((directory / 'setup.json').read_text())['done'] == [
            'preflight', 'sources', 'environments', 'llama', 'checkpoint', 'data', 'create', 'launch'], where
        outcome = json.loads((directory / 'outcome.json').read_text())
        assert outcome['status'] == 'failed' and outcome['cleanup'] == 'confirmed' and len(outcome['rental']) == 1, where
        assert account.created == 2 + index and account.peak == 1 and account.owned() == [], where


def test_a_run_lost_before_its_job_exists_moves_to_another_machine_and_starts_setup_over(tmp_path, rented, monkeypatch):
    account = rented('vastai')
    config, config_path, splits = job(tmp_path, 'vastai')
    Worker.die = (8, 'before')
    killed(monkeypatch, config_path, splits, tmp_path)
    directory = run_directory(tmp_path)
    done = json.loads((directory / 'setup.json').read_text())
    assert done['done'] and not (directory / 'remote.json').exists()
    late = time.time() + machines.LEASE_SECONDS + 1
    Resources(tmp_path / 'registry', account.client(), clock=lambda: late).reap()
    assert account.owned() == []
    Resources(tmp_path / 'registry', account.client()).reap()
    Worker.reset()
    with pytest.raises(JobError, match='did not complete'):
        train(config_path, 'e4b', splits, tmp_path, resume=directory)
    assert account.created == 2 and account.peak == 1 and len(did('launch')) == 1
    assert any('fetch' in action[1] for action in did('command'))
    assert [item['outcome'] for item in outcome_of(tmp_path)['rental']] == ['lost', 'used']


def test_a_run_whose_machine_is_gone_after_its_job_started_does_not_move(tmp_path, rented, monkeypatch):
    account = rented('vastai')
    config, config_path, splits = job(tmp_path, 'vastai')

    def dies():
        raise Died()
    Worker.states = [dies]
    killed(monkeypatch, config_path, splits, tmp_path)
    late = time.time() + machines.LEASE_SECONDS + 1
    Resources(tmp_path / 'registry', account.client(), clock=lambda: late).reap()
    Resources(tmp_path / 'registry', account.client()).reap()
    Worker.reset()
    with pytest.raises(JobError, match='did not complete'):
        train(config_path, 'e4b', splits, tmp_path, resume=run_directory(tmp_path))
    assert account.created == 1 and did('launch') == [] and outcome_of(tmp_path)['status'] == 'failed'


def test_a_failed_job_releases_its_machine_and_stays_terminal(tmp_path, rented):
    account = rented('scaleway')
    config, config_path, splits = job(tmp_path)
    with pytest.raises(JobError, match='did not complete'):
        train(config_path, 'e4b', splits, tmp_path)
    directory = run_directory(tmp_path)
    outcome = outcome_of(tmp_path)
    assert outcome['status'] == 'failed' and outcome['cleanup'] == 'confirmed'
    assert account.created == 1 and account.owned() == []
    assert did('create') == [('create', 'full')] and len(did('receive')) == 1
    with pytest.raises(JobError, match='terminal outcome'):
        train(config_path, 'e4b', splits, tmp_path, resume=directory)
    assert account.created == 1


@pytest.mark.parametrize('result', ['complete', 'failed'])
def test_resume_settles_a_release_the_provider_had_not_confirmed(tmp_path, rented, monkeypatch, result):
    account = rented('vastai')
    config, config_path, splits = job(tmp_path, 'vastai')
    directory = tmp_path / 'instance'
    directory.mkdir()
    ledger = Resources(tmp_path / 'registry', account.client())
    record = ledger.intent(config['provisioning']['specification'], 300, 3600)
    ledger.allocate(record['token'], 'startup')
    write(directory / 'resource.json', record)
    write(directory / 'binding.json', {'configuration': identity(config), 'student': 'e4b', 'splits': tree(splits), 'schedule': 'full'})
    write(directory / 'outcome.json', {'status': result, 'cleanup': 'unconfirmed'})
    monkeypatch.setattr(remote_training, 'validate_inputs', lambda plan: None)
    monkeypatch.setattr(remote_training, 'validate_result', lambda plan: None)
    write_plan = directory / 'job'
    write_plan.mkdir()
    write(write_plan / 'plan.json', {})
    account.fail(len(account.calls), 'outage')
    with pytest.raises(JobError, match='terminal outcome'):
        remote_training.train(config_path, 'e4b', splits, 'unused', tmp_path, resume=directory, sleep=lambda _: None)
    assert len(account.owned()) == 1 and json.loads((directory / 'outcome.json').read_text())['cleanup'] == 'unconfirmed'
    account.heal()
    if result == 'complete':
        settled = remote_training.train(config_path, 'e4b', splits, 'unused', tmp_path, resume=directory, sleep=lambda _: None)
        assert settled['status'] == 'complete' and settled['cleanup'] == 'confirmed'
    else:
        with pytest.raises(JobError, match='terminal outcome'):
            remote_training.train(config_path, 'e4b', splits, 'unused', tmp_path, resume=directory, sleep=lambda _: None)
        assert json.loads((directory / 'outcome.json').read_text())['cleanup'] == 'confirmed'
    assert account.owned() == [] and account.created == 1


def test_a_machine_the_provider_took_away_is_reported_as_lost_and_released(tmp_path, rented):
    account = rented('vastai')
    config, config_path, splits = job(tmp_path, 'vastai')
    Worker.states = [JobError('ssh')]
    answer = account.answer

    def exited(command):
        rows = answer(command)
        if command[2:4] == ['show', 'instances'] and did('launch'):
            for row in rows:
                row['actual_status'] = 'exited'
        return rows
    account.answer = exited
    with pytest.raises(JobError, match='did not complete'):
        train(config_path, 'e4b', splits, tmp_path)
    outcome = outcome_of(tmp_path)
    assert outcome['failure']['diagnostic'] == 'rented machine was lost: the provider reports it exited'
    assert outcome['cleanup'] == 'confirmed' and account.owned() == []


def test_a_specification_too_small_for_the_tier_rents_nothing(tmp_path, rented):
    account = rented('vastai')
    config, config_path, splits = job(tmp_path, 'vastai')
    config['provisioning']['specification']['disk_gib'] = 250
    write(config_path, config)
    with pytest.raises(JobError, match='cannot hold student 26b-a4b: disk_gib is 250'):
        train(config_path, '26b-a4b', splits, tmp_path)
    assert account.calls == [] and not (tmp_path / 'instances').exists()


@pytest.mark.parametrize('schedule', ['full', 'pilot', 'rehearsal'])
@pytest.mark.parametrize('provider', [None, 'scaleway'])
def test_the_job_on_the_machine_is_created_on_the_schedule_asked_for(tmp_path, rented, schedule, provider):
    if provider:
        rented(provider)
    config, config_path, splits = job(tmp_path, provider)
    with pytest.raises(JobError, match='did not complete'):
        train(config_path, '12b', splits, tmp_path, schedule=schedule)
    assert did('create') == [('create', schedule)]
    assert json.loads((run_directory(tmp_path) / 'binding.json').read_text())['schedule'] == schedule
    assert outcome_of(tmp_path)['cleanup'] == ('confirmed' if provider else 'not_applicable')
    with pytest.raises(JobError, match='unknown training schedule'):
        train(config_path, '12b', splits, tmp_path, schedule='two-updates')


def trained(monkeypatch):
    """A worker whose training completes after a paused attempt that held the peak."""
    monkeypatch.setattr(remote_training, 'validate_inputs', lambda plan: None)
    monkeypatch.setattr(remote_training, 'validate_result', lambda plan: None)
    execution = {'status': 'complete', 'attempts': 2,
                 'history': [{'attempt': 1, 'status': 'paused', 'peak_cuda_bytes': 119460112896},
                             {'attempt': 2, 'status': 'complete', 'peak_cuda_bytes': 80000000000}],
                 'resources': {'peak_cuda_bytes': 119460112896, 'peak_process_rss_bytes': 52757864448, 'elapsed_seconds': 611.0}}
    Worker.states = [{'status': 'running', 'attempts': 1}, execution]
    return execution


@pytest.mark.parametrize('exports, diagnostic', [([{'exit': 1, 'alive': False}], 'remote export failed with status 1'),
                                                 ([{'exit': None, 'alive': False}], 'remote export worker lost')])
def test_a_failed_export_brings_its_evidence_and_the_training_record_home_before_the_machine_is_released(
        tmp_path, rented, monkeypatch, exports, diagnostic):
    account = rented('vastai')
    config, config_path, splits = job(tmp_path, 'vastai')
    execution = trained(monkeypatch)
    Worker.exports = exports
    held = []
    receive = Worker.receive

    def receiving(self, source, destination, **kwargs):
        held.append(len(account.owned()))
        receive(self, source, destination, **kwargs)
    monkeypatch.setattr(Worker, 'receive', receiving)
    with pytest.raises(JobError, match='did not complete'):
        train(config_path, '26b-a4b', splits, tmp_path)
    outcome, directory = outcome_of(tmp_path), run_directory(tmp_path)
    remote = '/workspace/training/%s/runs/%s/' % (directory.name, directory.name)
    # The whole job directory comes home twice, around the export, and never with the bulk.
    assert did('receive') == [('receive', remote, remote_training.BULK, None),
                              ('receive', remote, remote_training.BULK, remote_training.EVIDENCE_FILE_LIMIT)]
    assert {'/merged/model/', '/gguf/*.gguf', '/mlx/*.safetensors'} <= set(remote_training.BULK)
    assert not any(pattern in remote_training.BULK for pattern in ('/lora_adapter/', '/training/adapter/', '*.log', '*.json'))
    assert held == [1, 1] and account.owned() == []
    assert outcome['status'] == 'failed' and outcome['evidence'] == 'returned' and outcome['cleanup'] == 'confirmed'
    assert outcome['failure'] == {'type': 'JobError', 'code': 'failed', 'diagnostic': diagnostic}
    assert outcome['resources'] == execution['resources'] and outcome['execution']['history'] == execution['history']


def test_evidence_that_cannot_be_fetched_does_not_keep_the_machine(tmp_path, rented, monkeypatch):
    import subprocess
    account = rented('vastai')
    config, config_path, splits = job(tmp_path, 'vastai')
    trained(monkeypatch)
    Worker.exports = [{'exit': 1, 'alive': False}]
    receive = Worker.receive

    def receiving(self, source, destination, limit=None, **kwargs):
        if limit:
            raise subprocess.CalledProcessError(23, 'rsync')
        receive(self, source, destination, **kwargs)
    monkeypatch.setattr(Worker, 'receive', receiving)
    with pytest.raises(JobError, match='did not complete'):
        train(config_path, '26b-a4b', splits, tmp_path)
    outcome = outcome_of(tmp_path)
    assert outcome['evidence'] == 'unavailable' and outcome['cleanup'] == 'confirmed' and account.owned() == []
    assert outcome['resources']['peak_cuda_bytes'] == 119460112896


def ticking(step=30.0):
    """A clock that moves `step` seconds each time it is read."""
    now = [1_800_000_000.0]

    def clock():
        now[0] += step
        return now[0]
    return clock


def test_a_complete_job_records_each_stages_time_and_what_its_machine_cost(tmp_path, rented, monkeypatch, capsys):
    account = rented('vastai')
    config, config_path, splits = job(tmp_path, 'vastai')
    trained(monkeypatch)
    monkeypatch.setattr(remote_training.exporting, 'publish', lambda *args, **kwargs: {'published': True})
    result = train(config_path, '26b-a4b', splits, tmp_path, clock=ticking())
    assert result['status'] == 'complete' and result['artifact'] == {'published': True} and result['cleanup'] == 'confirmed'
    assert list(result['stages']) == ['rent', 'boot', 'preflight', 'bootstrap', 'upload', 'train', 'retrieve', 'export', 'release']
    assert all(seconds > 0 for seconds in result['stages'].values())
    (machine,) = result['rental']
    assert machine['price_per_hour'] == 3.0 and machine['host'] == '70' and machine['outcome'] == 'used'
    assert machine['hours'] == round((machine['ended_at'] - machine['started_at']) / 3600, 4) > 0
    assert machine['cost'] == round(machine['hours'] * 3.0, 4)
    lines = [line.split() for line in capsys.readouterr().err.splitlines()]
    assert [line[2] for line in lines] == ['stage=' + name for name in (
        'rent', 'boot', 'preflight', 'bootstrap', 'upload', 'train', 'retrieve', 'export', 'retrieve', 'release')]
    assert {line[1] for line in lines} == {run_directory(tmp_path).name}
    assert account.owned() == []


def test_a_search_that_finds_no_offer_is_a_typed_outcome_with_nothing_left_in_the_ledger(tmp_path, rented):
    account = rented('vastai')
    account.offers = []
    config, config_path, splits = job(tmp_path, 'vastai')
    with pytest.raises(JobError, match='did not complete'):
        train(config_path, 'e4b', splits, tmp_path)
    outcome = outcome_of(tmp_path)
    assert outcome['failure']['code'] == 'no_offer' and outcome['failure']['type'] == 'NoOffer'
    assert outcome['cleanup'] == 'confirmed' and account.created == 0 and Worker.actions == []
    assert {item['outcome'] for item in outcome['rental']} == {'no_offer'} and sum(item['cost'] or 0 for item in outcome['rental']) == 0
    assert Resources(tmp_path / 'registry', account.client()).unsettled() == {'leases': [], 'instances': []}


@pytest.mark.parametrize('silent, code', [(1, None), (machines.RENT_ATTEMPTS, 'machine_unusable')])
def test_a_job_takes_the_next_offer_when_a_machine_is_unusable_and_records_each(tmp_path, rented, monkeypatch, silent, code):
    account = rented('vastai')
    config, config_path, splits = job(tmp_path, 'vastai')
    made = []

    class Sometimes(Worker):
        def __init__(self, address, settings):
            super().__init__(address, settings)
            made.append(self)
            self.silent = len(made) <= silent

        def command(self, args, **kwargs):
            if self.silent:
                raise machines.Refused('refused')
            return super().command(args, **kwargs)
    monkeypatch.setattr(remote_training, 'Remote', Sometimes)
    with pytest.raises(JobError, match='did not complete'):
        train(config_path, 'e4b', splits, tmp_path)
    outcome = outcome_of(tmp_path)
    assert [item['outcome'] for item in outcome['rental']] == ['unusable'] * silent + ['used'] * (code is None)
    assert all('refuses the registered SSH key' in item['reason'] for item in outcome['rental'][:silent])
    assert all(item['cleanup'] == 'confirmed' and item['cost'] is not None for item in outcome['rental'])
    assert outcome['failure'].get('code') == (code or 'failed') and len(did('launch')) == (1 if code is None else 0)
    assert account.peak == 1 and account.owned() == [] and outcome['cleanup'] == 'confirmed'


def test_a_job_does_not_start_over_an_unsettled_ledger_and_says_what_is_there(tmp_path, rented):
    account = rented('vastai')
    config, config_path, splits = job(tmp_path, 'vastai')
    account.instances[5] = {'id': 5, 'label': 'pw-training-' + 'f' * 32, 'listed': 99, 'is_bid': False, 'machine_id': 9, 'dph_total': 1.0}
    with pytest.raises(JobError, match='did not complete'):
        train(config_path, 'e4b', splits, tmp_path)
    outcome = outcome_of(tmp_path)
    assert outcome['failure']['code'] == 'ledger_unsettled' and '5 (running)' in outcome['failure']['diagnostic']
    assert account.created == 0 and outcome['rental'] == [] and outcome['cleanup'] == 'not_applicable'


def test_status_reports_a_running_job_from_its_records_and_one_listing(tmp_path, rented, monkeypatch):
    account = rented('vastai')
    config, config_path, splits = job(tmp_path, 'vastai')

    def dies():
        raise Died()
    Worker.states = [dies]
    killed(monkeypatch, config_path, splits, tmp_path, clock=ticking())
    directory = run_directory(tmp_path)
    (directory / 'checkpoints' / '00000001-step000000000001').mkdir(parents=True)
    calls = len(account.calls)
    clock = ticking()
    report = status.status(directory, clock=lambda: clock() + 3600)
    assert [command[2:4] for command in account.calls[calls:]] == [['show', 'instances']]
    assert (report['status'], report['stage'], report['student'], report['schedule']) == ('running', 'train', 'e4b', 'full')
    assert report['instance']['id'] == '9001' and report['instance']['state'] == 'running' and report['instance']['price_per_hour'] == 3.0
    assert report['checkpoints'] == ['00000001-step000000000001'] and report['cleanup'] is None
    assert report['rental'][0]['hours'] > 0.9 and report['cost'] == report['rental'][0]['cost'] > 2.7
    assert report['elapsed_seconds'] > 3600 and report['stage_seconds']['train'] > 3600
    assert [item['id'] for item in report['account']] == ['9001'] and report['account_empty'] is False
    assert [(item['state'], item['live']) for item in report['ledger']] == [('active', True)] and report['ledger_empty'] is False
    Worker.states = [{'status': 'failed'}]
    with pytest.raises(JobError, match='did not complete'):
        train(config_path, 'e4b', splits, tmp_path, resume=directory)
    report = status.status(directory)
    assert (report['status'], report['stage'], report['instance'], report['cleanup']) == ('failed', None, None, 'confirmed')
    assert report['account_empty'] is True and report['ledger_empty'] is True
    account.fail(len(account.calls), 'outage')
    assert status.status(directory)['account_empty'] is None


def test_status_of_a_job_on_an_existing_machine_asks_no_provider(tmp_path, rented):
    config, config_path, splits = job(tmp_path, None)
    with pytest.raises(JobError, match='did not complete'):
        train(config_path, 'e4b', splits, tmp_path)
    report = status.status(run_directory(tmp_path))
    assert report['status'] == 'failed' and report['cleanup'] == 'not_applicable' and 'account' not in report
    assert (run_directory(tmp_path) / 'configuration.json').stat().st_mode & 0o077 == 0


class ExportMachine:
    """A training machine whose launched export ends on the given observations."""
    host = 'fixture'

    def __init__(self, observations):
        self.observations = list(observations)
        self.commands, self.received = [], []

    def command(self, args, **kwargs):
        self.commands.append(args)
        if args[:2] == ['bash', '-lc']:
            return '4242\n'
        observation = self.observations.pop(0)
        if isinstance(observation, Exception):
            raise observation
        return json.dumps(observation)

    def receive(self, source, destination, exclude=(), timeout=600, limit=None):
        self.received.append((source, tuple(exclude)))


def export_fixture(tmp_path, monkeypatch):
    config = {'llama_commit': 'c' * 40, 'machine': {'uv': '/opt/uv/bin/uv'}}
    destination = tmp_path / 'instance'
    (destination / 'job').mkdir(parents=True)
    return config, destination


def test_export_launches_once_on_the_training_machine(tmp_path, monkeypatch):
    import subprocess
    config, destination = export_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(remote_training.exporting, 'publish', lambda *args, **kwargs: {'published': args})
    machine = ExportMachine([{'exit': None, 'alive': True}, subprocess.TimeoutExpired('probe', 30), {'exit': 0, 'alive': False}])
    artifact = remote_training.export_remotely(machine, config, '/w/job', '/w/job/runs/one', destination, lambda: None, lambda _: None)
    launch = next(args for args in machine.commands if args[:2] == ['bash', '-lc'])[2]
    outer = shlex.split(launch)
    assert outer[:3] == ['nohup', 'bash', '-c']
    inner = shlex.split(outer[3].split(';')[0])
    assert inner[-6:] == ['export', '/w/job/runs/one', '--llama', '/w/job/llama.cpp', '--revision', 'c' * 40]
    assert json.loads((destination / 'export.json').read_text()) == {'pid': 4242}
    assert ('/w/job/runs/one/gguf/', ('model-f16.gguf',)) in machine.received
    assert artifact == {'published': (destination / 'job', 'c' * 40)}
    assert ('/w/job/runs/one/export.log', ()) in machine.received
    resumed = ExportMachine([{'exit': 0, 'alive': False}])
    remote_training.export_remotely(resumed, config, '/w/job', '/w/job/runs/one', destination, lambda: None, lambda _: None)
    assert not any(args[:2] == ['bash', '-lc'] for args in resumed.commands)


@pytest.mark.parametrize('observations, message', [
    ([{'exit': 1, 'alive': False}], 'failed with status 1'),
    ([{'exit': None, 'alive': False}], 'worker lost'),
    ([JobError('ssh')] * 3, 'observation unavailable')])
def test_remote_export_failures_are_terminal(tmp_path, monkeypatch, observations, message):
    config, destination = export_fixture(tmp_path, monkeypatch)
    machine = ExportMachine(observations)
    with pytest.raises(JobError, match=message):
        remote_training.export_remotely(machine, config, '/w/job', '/w/job/runs/one', destination, lambda: None, lambda _: None)
    assert machine.received == []
