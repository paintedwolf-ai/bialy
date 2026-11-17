import json
import subprocess
from types import SimpleNamespace

import pytest
from rental_accounts import Machine, VastAccount, configuration

from bialy.student_host import machines, reaper
from bialy.student_host.jobs import JobError
from bialy.student_host.remote_training import Remote, prepare, setup_steps
from bialy.student_host.resources import ResourceError
from bialy.student_host.specification import students

EXAMPLES = machines.ROOT / 'train-host/student'


def existing(tmp_path):
    return {'host_repository': 'https://example.invalid/host.git', 'host_commit': 'a' * 40,
            'bialy_repository': 'https://example.invalid/bialy.git', 'bialy_commit': 'b' * 40,
            'llama_repository': 'https://example.invalid/llama.cpp.git', 'llama_commit': 'c' * 40,
            'host_checkout': str(tmp_path / 'host'), 'models': str(tmp_path / 'models'),
            'machine': {'host': 'training.example.org', 'user': 'trainer', 'port': 2222,
                        'workspace': '/srv/training', 'uv': '/opt/uv/bin/uv', 'device': 'cuda:1'}}


def example(name):
    """An example file as an operator completes it: the blanks are deployment values."""
    value = json.loads((EXAMPLES / name).read_text())
    value.update(host_commit='a' * 40, bialy_commit='b' * 40, llama_commit='c' * 40)
    if 'provisioning' in value:
        value['provisioning'].update({key: 'filled' for key in ('project',) if key in value['provisioning']})
        value['provisioning']['specification'] = {key: field or 'filled' for key, field in value['provisioning']['specification'].items()}
    return value


@pytest.fixture(autouse=True)
def installed_cli(monkeypatch):
    """The pinned provider CLI, answered by a fake account instead of an installed one."""
    monkeypatch.setattr(machines, 'provider', lambda provision: VastAccount().client())


def bootstrap(remote, config, workspace, student, directory):
    """Every setup step up to the upload, as a job runs them on a machine it just reached."""
    job = directory / workspace.rsplit('/', 1)[1]
    job.mkdir(exist_ok=True)
    steps = [step for step in setup_steps(remote, config, workspace, student, directory / 'splits', directory / 'key', job, 'full')
             if step[0] in ('preflight', 'bootstrap')]
    prepare(steps, job, lambda: None, lambda name: None)
    return job


def test_every_declared_student_has_a_machine_requirement():
    for student in students():
        assert set(machines.requirement(student)) == set(machines.CAPACITY)
    with pytest.raises(JobError, match='no machine requirement'):
        machines.requirement('undeclared')


# A card that holds every tier; the example specifications declare one that holds E4B.
ROOMY = 140


# The tiers each shipped example is for. An existing machine declares nothing and is measured instead.
EXAMPLE_TIERS = {'machine.example.json': ('e4b', '12b', '26b-a4b'), 'provisioned-machine.example.json': ('e4b',),
            'vastai-machine.example.json': ('e4b',), 'vastai-large-machine.example.json': ('12b', '26b-a4b')}


def test_every_shipped_example_passes_the_offline_preflight_for_the_tiers_it_is_for():
    assert {path.name for path in EXAMPLES.glob('*machine.example.json')} == set(EXAMPLE_TIERS)
    assert {student for tiers in EXAMPLE_TIERS.values() for student in tiers} == set(students())
    for name, tiers in EXAMPLE_TIERS.items():
        for student in tiers:
            report = machines.preflight(example(name), student)
            assert report['requirement'] == machines.requirement(student)
            assert (report['declared'] is None) is (name == 'machine.example.json')


@pytest.mark.parametrize('name', ['provisioned-machine.example.json', 'vastai-machine.example.json'])
def test_an_80_gb_example_refuses_the_tiers_its_card_cannot_hold(name):
    for student in ('12b', '26b-a4b'):
        with pytest.raises(JobError, match='cannot hold student %s: gpu_memory_gib is 79' % student):
            machines.preflight(example(name), student)


@pytest.mark.parametrize('provider', ['scaleway', 'vastai'])
@pytest.mark.parametrize('field', machines.CAPACITY)
def test_a_specification_that_cannot_hold_the_tier_is_refused_before_renting(tmp_path, monkeypatch, provider, field):
    largest = max(students(), key=lambda student: machines.requirement(student)['disk_gib'])
    short = machines.requirement(largest)[field] - 1
    config = configuration(tmp_path, provider, **{'gpu_memory_gib': ROOMY, field: short})
    with monkeypatch.context() as untouched:
        untouched.setattr(machines, 'provider', lambda provision: pytest.fail('a provider was contacted'))
        with pytest.raises(JobError, match='%s is %d, the tier needs %d' % (field, short, short + 1)):
            machines.preflight(config, largest)
    machines.preflight(configuration(tmp_path, provider, gpu_memory_gib=ROOMY), largest)


@pytest.mark.parametrize('version', ['1.0.13', '', '2.0.0'])
def test_a_provider_cli_other_than_the_pinned_release_is_refused_before_renting(tmp_path, monkeypatch, version):
    account = VastAccount()
    account.version = version
    monkeypatch.setattr(machines, 'provider', lambda provision: account.client())
    with pytest.raises(JobError, match=r'this controller drives vastai==\d'):
        machines.preflight(configuration(tmp_path, 'vastai', gpu_memory_gib=ROOMY), 'e4b')
    assert account.calls == [['vastai', '--version']]


@pytest.mark.parametrize('change', [
    {'rental': 'interruptible'}, {'rental': None}, {'lifetime_hours': 0}, {'lifetime_hours': 1000}, {'lifetime_hours': '24'},
    {'start_minutes': 0}, {'start_minutes': 61}, {'start_minutes': '15'},
    {'registry': 'relative'}, {'unknown': 1}])
def test_a_rental_must_be_on_demand_and_bounded(tmp_path, change):
    config = configuration(tmp_path, 'vastai')
    config['provisioning'].update(change)
    with pytest.raises(JobError):
        machines.validate(config)
    for missing in ('rental', 'lifetime_hours', 'start_minutes'):
        config = configuration(tmp_path, 'scaleway')
        del config['provisioning'][missing]
        with pytest.raises(JobError):
            machines.validate(config)


@pytest.mark.parametrize('change', [
    {'bid_price': 1.1}, {'max_price': '3.5'}, {'max_price': 0}, {'disk_gib': '320'}, {'num_gpus': '1'},
    {'download_mbps': '500'}, {'download_mbps': 0},
    {'image': 'example/image'}, {'image': 'example/image:latest'}, {'image': ''}])
def test_a_vastai_specification_cannot_bid_or_leave_capacity_price_or_image_open(tmp_path, change):
    config = configuration(tmp_path, 'vastai', **change)
    with pytest.raises(JobError):
        machines.validate(config)
    machines.validate(configuration(tmp_path, 'vastai', image='example/image@sha256:' + 'a' * 64))


@pytest.mark.parametrize('machine', [{'user': 'trainer'}, {'host': 'training.example.org'}, {'port': 22}])
def test_a_rented_machine_is_reached_as_root_where_the_provider_says(tmp_path, machine):
    config = configuration(tmp_path, 'vastai')
    config['machine'].update(machine)
    with pytest.raises(JobError, match='root'):
        machines.validate(config)


@pytest.mark.parametrize('field,value', [('workspace', '/'), ('workspace', '/srv/../other'), ('user', '-root'), ('port', 0),
                                         ('host', '-option'), ('device', 'cuda:0; reboot')])
def test_invalid_machine_connection_is_rejected_before_execution(tmp_path, field, value):
    config = existing(tmp_path)
    config['machine'][field] = value
    with pytest.raises(JobError):
        machines.validate(config)


def test_export_tools_must_be_pinned_to_a_full_commit(tmp_path):
    config = existing(tmp_path)
    config['llama_commit'] = 'master'
    with pytest.raises(JobError, match='full commit'):
        machines.validate(config)


def test_existing_machine_never_enters_instance_provisioning_or_cleanup(tmp_path, monkeypatch):
    config = existing(tmp_path)
    machines.validate(config)
    monkeypatch.setattr(machines, 'Resources', lambda *args: pytest.fail('existing machine used provisioning'))
    connections = []

    def remote(host, settings):
        connections.append((host, settings))
        return Machine(host, settings)
    with machines.connect(config, tmp_path, remote, sleep=lambda _: None) as (connection, supervisor):
        assert connection.commands == [['true']] and supervisor.check() is None and supervisor.lost() is None
    assert connections == [('training.example.org', config['machine'])]
    assert not (tmp_path / 'resource.json').exists()
    assert json.loads((tmp_path / 'cleanup.json').read_text()) == {'status': 'not_applicable'}
    path = tmp_path / 'machine.json'
    path.write_text(json.dumps(config))
    assert reaper.install(path, run=lambda *args, **kwargs: pytest.fail('installed a reaper')) == {'status': 'not_required'}
    assert reaper.reap(path, once=True) is None


def test_ssh_transport_uses_the_configured_account_port_and_key(tmp_path):
    config = existing(tmp_path)
    key = tmp_path / 'key'
    key.write_text('fixture')
    config['machine']['identity_file'] = str(key)
    machines.validate(config)
    calls = []

    def run(args, **kwargs):
        calls.append((args, kwargs))
        return SimpleNamespace(returncode=0, stdout='ready')
    remote = Remote(config['machine']['host'], config['machine'], run=run)
    assert remote.command(['printf', '%s', 'value with spaces']) == 'ready'
    assert ['-p', '2222', '-i', str(key)] == calls[0][0][-6:-2]
    assert calls[0][0][-2:] == ['trainer@training.example.org', "printf %s 'value with spaces'"]
    script = tmp_path / 'script.sh'
    script.write_text('echo measured\n')
    assert remote.script(script, ['check', '/srv/training space']) == (0, 'ready')
    assert calls[1][0][-1] == "bash -s -- check '/srv/training space'" and calls[1][1]['input'] == 'echo measured\n'


class Measured:
    """A machine that answers the preflight script with the given report."""

    def __init__(self, report, status=0):
        self.report, self.status, self.scripts, self.commands = report, status, [], []

    def script(self, path, args, timeout=600):
        self.scripts.append((path, args))
        return self.status, 'apt noise\n' + json.dumps(self.report) + '\n'

    def command(self, args, **kwargs):
        self.commands.append(args)
        return ''


def test_a_machine_is_measured_against_the_tier_before_bootstrap_touches_it(tmp_path):
    needed = machines.requirement('26b-a4b')
    remote = Measured({'disk_gib': 250, 'memory_gib': 500, 'gpu_memory_gib': 79,
                       'problems': ['workspace has 250 GiB free; the tier needs %d' % needed['disk_gib']]}, status=1)
    with pytest.raises(JobError, match='workspace has 250 GiB free'):
        bootstrap(remote, configuration(tmp_path, 'vastai'), '/workspace/training/job', '26b-a4b', tmp_path)
    assert remote.commands == []
    path, args = remote.scripts[0]
    tmp_path = tmp_path / 'job'
    assert path == machines.PREFLIGHT_SCRIPT
    assert args == ['install', '/workspace/training', '/root/.local/bin/uv', 'cuda:0',
                    str(needed['disk_gib']), str(needed['memory_gib']), str(needed['gpu_memory_gib'])]
    assert json.loads((tmp_path / 'machine.json').read_text())['disk_gib'] == 250


def test_an_existing_machine_is_only_checked_and_a_silent_preflight_is_a_failure(tmp_path):
    remote = Measured({'disk_gib': 900, 'memory_gib': 500, 'gpu_memory_gib': 79, 'problems': []})
    assert machines.measure(remote, existing(tmp_path), 'e4b', tmp_path)['problems'] == []
    assert remote.scripts[0][1][:4] == ['check', '/srv/training', '/opt/uv/bin/uv', 'cuda:1']
    remote.script = lambda path, args, timeout: (255, '')
    with pytest.raises(JobError, match='did not report'):
        machines.measure(remote, existing(tmp_path), 'e4b', tmp_path)
    remote.script = lambda path, args, timeout: (1, json.dumps({'problems': []}))
    with pytest.raises(JobError, match='preflight failed'):
        machines.measure(remote, existing(tmp_path), 'e4b', tmp_path)


def test_bootstrap_scopes_sources_tools_and_models_to_the_job_workspace(tmp_path):
    config = existing(tmp_path)
    remote = Measured({'disk_gib': 900, 'memory_gib': 500, 'gpu_memory_gib': 79, 'problems': []})
    bootstrap(remote, config, '/srv/training/job', '26b-a4b', tmp_path)
    calls = remote.commands
    lock = ['flock', '/srv/training/job/.setup.lock']
    fetched = [args for args in calls if 'fetch' in args]
    assert [args[4] for args in fetched] == ['/srv/training/job/host', '/srv/training/job/bialy', '/srv/training/job/llama.cpp']
    assert all(args[:2] == lock for args in fetched)
    assert all('/root/' not in ' '.join(args) and '/scratch/' not in ' '.join(args) for args in calls)
    assert any('UV_TOOL_DIR=/srv/training/job/tools' in args for args in calls)
    models = next(args for args in calls if 'bialy.student_host' in args)
    assert models[-5:] == ['models', '--root', '/srv/training/job/models', '--student', '26b-a4b']
    llama = '/srv/training/job/llama.cpp'
    assert [*lock, 'git', '-C', llama, 'checkout', '-q', '--detach', 'c' * 40] in calls
    configure, build = [args for args in calls if 'cmake' in args]
    assert '-DGGML_CUDA=OFF' in configure
    assert build[2:5] == ['cmake', '--build', llama + '/build'] and {'llama-quantize', 'llama-server'} <= set(build)
    synced = [args[args.index('--project') + 1] for args in calls if 'sync' in args]
    assert synced == [prefix + suffix for prefix in ['/srv/training/job/'] for suffix in ('bialy', 'host/scripts/bialy/custom_model', 'host/scripts/generation')]


def test_setup_interrupted_at_any_command_resumes_without_repeating_a_finished_step(tmp_path):
    config = existing(tmp_path)
    report = {'disk_gib': 900, 'memory_gib': 500, 'gpu_memory_gib': 79, 'problems': []}
    clean = Measured(report)
    bootstrap(clean, config, '/srv/training/job', 'e4b', tmp_path)
    total = len(clean.commands)
    assert total > 10
    for index in range(total):
        root = tmp_path / str(index)
        root.mkdir()
        first = Measured(report)

        def dying(args, index=index, first=first, **kwargs):
            if len(first.commands) == index:
                raise JobError('connection lost')
            first.commands.append(args)
            return ''
        first.command = dying
        with pytest.raises(JobError, match='connection lost'):
            bootstrap(first, config, '/srv/training/job', 'e4b', root)
        done = json.loads((root / 'job' / 'setup.json').read_text())['done'] if (root / 'job' / 'setup.json').exists() else []
        second = Measured(report)
        job = bootstrap(second, config, '/srv/training/job', 'e4b', root)
        assert json.loads((job / 'setup.json').read_text())['done'] == ['preflight', 'sources', 'environments', 'llama', 'checkpoint']
        # The interrupted step runs whole again; the steps recorded before it do not.
        assert (len(second.scripts) == 0) is ('preflight' in done)
        assert clean.commands[-1] == second.commands[-1] and len(second.commands) <= total
        assert set(map(tuple, clean.commands)) == set(map(tuple, first.commands + second.commands))


def test_setup_starts_over_on_a_machine_that_replaced_the_one_it_began_on(tmp_path):
    from bialy.student_host.resources import write
    config = existing(tmp_path)
    remote = Measured({'disk_gib': 900, 'memory_gib': 500, 'gpu_memory_gib': 79, 'problems': []})
    job = bootstrap(remote, config, '/srv/training/job', 'e4b', tmp_path)
    again = Measured(remote.report)
    bootstrap(again, config, '/srv/training/job', 'e4b', tmp_path)
    assert again.commands == [] and again.scripts == []
    write(job / 'resource.json', {'token': 'another-lease'})
    bootstrap(again, config, '/srv/training/job', 'e4b', tmp_path)
    assert again.commands == remote.commands and len(again.scripts) == 1


def test_a_refused_key_is_told_apart_from_a_connection_not_yet_accepted(tmp_path):
    def remote(returncode, stderr):
        return Remote('192.0.2.1', existing(tmp_path)['machine'],
                      run=lambda args, **kwargs: SimpleNamespace(returncode=returncode, stdout='', stderr=stderr))
    with pytest.raises(machines.Refused):
        remote(255, 'root@192.0.2.1: Permission denied (publickey).').command(['true'])
    for returncode, stderr in ((255, 'ssh: connect to host 192.0.2.1 port 22: Connection refused'), (1, 'Permission denied')):
        with pytest.raises(JobError) as caught:
            remote(returncode, stderr).command(['true'])
        assert not isinstance(caught.value, machines.Refused)


def test_a_machine_that_refuses_the_key_is_given_up_on_long_before_one_still_starting(tmp_path):
    slept = []

    class Answering(Machine):
        error = machines.Refused('refused')

        def command(self, args, **kwargs):
            raise self.error
    tries = 15 * 60 // machines.POLL_SECONDS
    with pytest.raises(machines.Unreachable, match='refuses the registered SSH key'):
        machines.reachable(Answering('192.0.2.1', {}), machines.Unsupervised(), slept.append, tries)
    assert machines.REFUSED_SECONDS <= sum(slept) <= machines.REFUSED_SECONDS + machines.POLL_SECONDS
    slept.clear()
    Answering.error = JobError('connection refused')
    with pytest.raises(machines.Unreachable, match='did not become reachable'):
        machines.reachable(Answering('192.0.2.1', {}), machines.Unsupervised(), slept.append, tries)
    assert sum(slept) == 15 * 60


def test_receive_never_pulls_excluded_artifacts(tmp_path):
    calls = []
    remote = Remote('192.0.2.1', existing(tmp_path)['machine'], run=lambda args, **kwargs: calls.append((args, kwargs)))
    remote.receive('/w/job/', tmp_path / 'job', exclude=('/merged/', '/gguf/'), timeout=7200, limit='1g')
    args, kwargs = calls[0]
    assert args[args.index('/merged/') - 1] == '--exclude' and args[args.index('/gguf/') - 1] == '--exclude'
    assert '.pending-*' in args and '--max-size=1g' in args and kwargs['timeout'] == 7200


class Ledger:
    def __init__(self):
        self.renewed, self.fail, self.instance = [], None, {'state': 'running'}

    def renew(self, token, seconds):
        if self.fail:
            raise self.fail
        self.renewed.append((token, seconds))

    def observe(self, token):
        return self.instance


def test_heartbeats_extend_the_machine_deadline_and_a_silent_machine_ends_the_job(tmp_path):
    now = [0.0]
    supervisor = machines.Supervisor(Ledger(), 'token', clock=lambda: now[0])
    machine = Machine('192.0.2.1', {})
    supervisor.attach(machine, sleep=lambda _: None)
    assert machine.commands == [[machines.DEADMAN, 'extend', str(machines.HEARTBEAT_SECONDS)]]
    now[0] = machines.HEARTBEAT_SECONDS
    supervisor.check()
    machine.command = lambda args, **kwargs: (_ for _ in ()).throw(subprocess.TimeoutExpired(args, 30))
    with pytest.raises(subprocess.TimeoutExpired):
        supervisor.beat()
    now[0] += 1
    with pytest.raises(JobError, match='dead-man heartbeat'):
        supervisor.check()


def test_a_failed_lease_renewal_ends_the_job_and_a_lost_machine_is_named(tmp_path):
    ledger = Ledger()
    supervisor = machines.Supervisor(ledger, 'token')
    assert supervisor.lost() is None
    ledger.instance = {'state': 'exited'}
    assert supervisor.lost() == 'the provider reports it exited'
    ledger.instance = None
    assert supervisor.lost() == 'the provider no longer lists it'
    ledger.observe = lambda token: (_ for _ in ()).throw(ResourceError('offline'))
    assert supervisor.lost() is None
    supervisor.error = ResourceError('expired')
    with pytest.raises(JobError, match='renewal failed'):
        supervisor.check()


def test_the_supervising_thread_renews_the_lease_and_beats_until_stopped(monkeypatch):
    ledger, beats = Ledger(), []
    supervisor = machines.Supervisor(ledger, 'token')
    supervisor.beaten = 0.0
    supervisor.beat = lambda: beats.append(1)
    waits = iter([False, False, True])
    supervisor.stop = SimpleNamespace(wait=lambda seconds: next(waits), set=lambda: None)
    supervisor.loop()
    assert ledger.renewed == [('token', machines.LEASE_SECONDS)] * 2 and len(beats) == 2
    ledger.fail = ResourceError('taken')
    supervisor.stop = SimpleNamespace(wait=lambda seconds: False, set=lambda: None)
    supervisor.loop()
    assert isinstance(supervisor.error, ResourceError)
