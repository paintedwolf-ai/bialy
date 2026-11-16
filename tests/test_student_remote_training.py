import json
import shlex

import pytest
from rental_accounts import ACCOUNTS, Died, Machine, configuration

from bialy.student_host import machines, remote_training, resources
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


class Worker(Machine):
    """A rented machine whose training worker reports the given execution states in turn."""
    states = [{'status': 'failed'}]
    actions = []

    def command(self, args, **kwargs):
        if args[:2] == ['python3', '-c']:
            state = type(self).states.pop(0) if len(type(self).states) > 1 else type(self).states[0]
            if callable(state):
                state()
            if isinstance(state, Exception):
                raise state
            return json.dumps({'execution': state, 'heartbeat_age': 0})
        if 'create' in args:
            type(self).actions.append(('create', '--pilot' in args))
            return json.dumps({'directory': '/workspace/runs/job'})
        return super().command(args, **kwargs)

    def upload(self, *args):
        pass

    def receive(self, source, destination, **kwargs):
        type(self).actions.append(('receive', source))


@pytest.fixture
def rented(tmp_path, monkeypatch):
    """A provider account and worker behind `remote_training.train`, with a healthy reaper."""
    def account(provider):
        account = ACCOUNTS[provider]()
        monkeypatch.setattr(machines, 'provider', lambda provision: account.client())
        Resources(tmp_path / 'registry', account.client()).reap()
        return account
    Worker.states, Worker.actions = [{'status': 'failed'}], []
    monkeypatch.setattr(remote_training, 'Remote', Worker)
    monkeypatch.setattr(remote_training, 'bootstrap', lambda *args: None)
    monkeypatch.setattr(remote_training, 'collect_checkpoint', lambda *args: [])
    return account


def test_restart_observes_existing_worker_without_launching_or_renting_again(tmp_path, rented, monkeypatch):
    account = rented('vastai')
    config, config_path, splits = job(tmp_path, 'vastai')
    ledger_write = resources.write

    def killed():
        # A killed controller neither reaches the provider nor writes its ledger again.
        account.dead = True
        monkeypatch.setattr(resources, 'write', lambda path, value: (_ for _ in ()).throw(Died()))
        raise Died()
    Worker.states = [killed]
    with pytest.raises(Died):
        remote_training.train(config_path, 'e4b', splits, 'unused', tmp_path / 'instances', sleep=lambda _: None)
    directory = next((tmp_path / 'instances').iterdir())
    assert json.loads((directory / 'outcome.json').read_text())['status'] == 'running'
    assert len(account.owned()) == 1 and Worker.actions == [('create', False)]
    account.heal()
    monkeypatch.setattr(resources, 'write', ledger_write)
    Worker.states = [{'status': 'failed', 'failure': {'code': 'out_of_memory'}}]
    with pytest.raises(JobError, match='did not complete'):
        remote_training.train(config_path, 'e4b', splits, 'unused', tmp_path / 'instances', resume=directory, sleep=lambda _: None)
    outcome = json.loads((directory / 'outcome.json').read_text())
    assert Worker.actions == [('create', False), ('receive', '/workspace/runs/job/')]
    assert outcome['status'] == 'failed' and outcome['cleanup'] == 'confirmed'
    assert account.created == 1 and account.owned() == []


def test_a_failed_job_releases_its_machine_and_stays_terminal(tmp_path, rented):
    account = rented('scaleway')
    config, config_path, splits = job(tmp_path)
    with pytest.raises(JobError, match='did not complete'):
        remote_training.train(config_path, 'e4b', splits, 'unused', tmp_path / 'instances', sleep=lambda _: None)
    directory = next((tmp_path / 'instances').iterdir())
    outcome = json.loads((directory / 'outcome.json').read_text())
    assert outcome['status'] == 'failed' and outcome['cleanup'] == 'confirmed'
    assert account.created == 1 and account.owned() == []
    assert Worker.actions == [('create', False), ('receive', '/workspace/runs/job/')]
    with pytest.raises(JobError, match='terminal outcome'):
        remote_training.train(config_path, 'e4b', splits, 'unused', tmp_path / 'instances', resume=directory, sleep=lambda _: None)
    assert account.created == 1


@pytest.mark.parametrize('status', ['complete', 'failed'])
def test_resume_settles_a_release_the_provider_had_not_confirmed(tmp_path, rented, monkeypatch, status):
    account = rented('vastai')
    config, config_path, splits = job(tmp_path, 'vastai')
    directory = tmp_path / 'instance'
    directory.mkdir()
    resources = Resources(tmp_path / 'registry', account.client())
    record = resources.intent(config['provisioning']['specification'], 300, 3600)
    resources.allocate(record['token'], 'startup')
    write(directory / 'resource.json', record)
    write(directory / 'binding.json', {'configuration': identity(config), 'student': 'e4b', 'splits': tree(splits), 'pilot': False})
    write(directory / 'outcome.json', {'status': status, 'cleanup': 'unconfirmed'})
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
    if status == 'complete':
        result = remote_training.train(config_path, 'e4b', splits, 'unused', tmp_path, resume=directory, sleep=lambda _: None)
        assert result['status'] == 'complete' and result['cleanup'] == 'confirmed'
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
        if command[2:4] == ['show', 'instances'] and any('--onstart-cmd' in call for call in account.calls) and len(Worker.actions) > 0:
            for row in rows:
                row['actual_status'] = 'exited'
        return rows
    account.answer = exited
    with pytest.raises(JobError, match='did not complete'):
        remote_training.train(config_path, 'e4b', splits, 'unused', tmp_path / 'instances', sleep=lambda _: None)
    outcome = json.loads((next((tmp_path / 'instances').iterdir()) / 'outcome.json').read_text())
    assert outcome['failure']['diagnostic'] == 'rented machine was lost: the provider reports it exited'
    assert outcome['cleanup'] == 'confirmed' and account.owned() == []


def test_a_specification_too_small_for_the_tier_rents_nothing(tmp_path, rented):
    account = rented('vastai')
    config, config_path, splits = job(tmp_path, 'vastai')
    config['provisioning']['specification']['disk_gib'] = 250
    write(config_path, config)
    with pytest.raises(JobError, match='cannot hold student 26b-a4b: disk_gib is 250'):
        remote_training.train(config_path, '26b-a4b', splits, 'unused', tmp_path / 'instances', sleep=lambda _: None)
    assert account.calls == [] and not (tmp_path / 'instances').exists()


@pytest.mark.parametrize('pilot', [False, True])
@pytest.mark.parametrize('provider', [None, 'scaleway'])
def test_remote_job_creation_preserves_pilot_mode(tmp_path, rented, pilot, provider):
    if provider:
        rented(provider)
    config, config_path, splits = job(tmp_path, provider)
    with pytest.raises(JobError, match='did not complete'):
        remote_training.train(config_path, '12b', splits, 'unused', tmp_path / 'instances', pilot=pilot, sleep=lambda _: None)
    assert Worker.actions[0] == ('create', pilot)
    directory = next((tmp_path / 'instances').iterdir())
    assert json.loads((directory / 'binding.json').read_text())['pilot'] is pilot
    assert json.loads((directory / 'outcome.json').read_text())['cleanup'] == ('confirmed' if provider else 'not_applicable')


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

    def receive(self, source, destination, exclude=(), timeout=600):
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
    resumed = ExportMachine([{'exit': 0, 'alive': False}])
    remote_training.export_remotely(resumed, config, '/w/job', '/w/job/runs/one', destination, lambda: None, lambda _: None)
    assert not any(args[:2] == ['bash', '-lc'] for args in resumed.commands)


@pytest.mark.parametrize('observations, message', [
    ([{'exit': 1, 'alive': False}], 'failed with status 1'),
    ([{'exit': None, 'alive': False}], 'worker lost'),
    ([JobError('ssh')] * 3, 'observation unavailable')])
def test_remote_export_failures_are_terminal_and_keep_the_log(tmp_path, monkeypatch, observations, message):
    config, destination = export_fixture(tmp_path, monkeypatch)
    machine = ExportMachine(observations)
    with pytest.raises(JobError, match=message):
        remote_training.export_remotely(machine, config, '/w/job', '/w/job/runs/one', destination, lambda: None, lambda _: None)
    assert (('/w/job/runs/one/export.log', ()) in machine.received) is (message == 'failed with status 1')
