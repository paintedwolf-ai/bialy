import json
import shlex

import pytest

from bialy.student_host import remote_training
from bialy.student_host.jobs import JobError, tree
from bialy.student_host.resources import write
from bialy.student_host.specification import identity


def test_restart_observes_existing_worker_without_launching_again(tmp_path, monkeypatch):
    config = {
              'host_repository': 'https://example.invalid/host.git', 'host_commit': 'a' * 40,
              'bialy_repository': 'https://example.invalid/bialy.git', 'bialy_commit': 'b' * 40,
            'llama_repository': 'https://example.invalid/llama.cpp.git', 'llama_commit': 'c' * 40,
              'host_checkout': str(tmp_path / 'host'), 'models': str(tmp_path / 'models'),
              'machine': {'user': 'user', 'workspace': '/workspace/jobs', 'uv': '/home/user/.local/bin/uv'},
              'provisioning': {'provider': 'scaleway', 'registry': str(tmp_path / 'registry'), 'profile': 'test',
                               'project': 'project', 'zone': 'zone',
                               'specification': {'type': 'gpu', 'image': 'image', 'root_volume': 'volume'}}}
    config_path = tmp_path / 'configuration.json'
    write(config_path, config)
    splits = tmp_path / 'splits'
    splits.mkdir()
    (splits / 'train.jsonl').write_text('fixture')
    directory = tmp_path / 'instance'
    directory.mkdir()
    write(directory / 'binding.json', {'configuration': identity(config), 'student': 'e4b', 'splits': tree(splits), 'pilot': False})
    write(directory / 'resource.json', {'token': 'owned'})
    write(directory / 'outcome.json', {'status': 'running', 'token': 'owned'})
    write(directory / 'remote.json', {'directory': '/scratch/runs/owned'})
    actions = []
    class Resources:
        def __init__(self, *args):
            pass
        def recover(self, token, seconds):
            actions.append(('recover', token))
            return {'server': {'public_ip': {'address': '192.0.2.1'}}}
        def release(self, token):
            actions.append(('release', token))
    class Remote:
        def __init__(self, address, machine):
            pass
        def command(self, args, **kwargs):
            if args == ['true']:
                return ''
            assert args[:2] == ['python3', '-c']
            return json.dumps({'execution': {'status': 'failed', 'failure': {'code': 'out_of_memory'}}, 'heartbeat_age': 0})
        def receive(self, source, destination, **kwargs):
            actions.append(('receive', source))
    monkeypatch.setattr(remote_training.machines, 'provider', lambda config: None)
    monkeypatch.setattr(remote_training.machines, 'Resources', Resources)
    monkeypatch.setattr(remote_training, 'Remote', Remote)
    monkeypatch.setattr(remote_training, 'collect_checkpoint', lambda *args: [])
    with pytest.raises(JobError, match='did not complete'):
        remote_training.train(config_path, 'e4b', splits, 'unused', tmp_path, resume=directory, sleep=lambda _: None)
    assert actions == [('recover', 'owned'), ('receive', '/scratch/runs/owned/'), ('release', 'owned')]
    outcome = json.loads((directory / 'outcome.json').read_text())
    assert outcome['status'] == 'failed' and outcome['cleanup'] == 'confirmed'


@pytest.mark.parametrize('pilot', [False, True])
@pytest.mark.parametrize('provision', [False, True])
def test_remote_job_creation_preserves_pilot_mode(tmp_path, monkeypatch, pilot, provision):
    config = {
              'host_repository': 'https://example.invalid/host.git', 'host_commit': 'a' * 40,
              'bialy_repository': 'https://example.invalid/bialy.git', 'bialy_commit': 'b' * 40,
            'llama_repository': 'https://example.invalid/llama.cpp.git', 'llama_commit': 'c' * 40,
              'host_checkout': str(tmp_path / 'host'), 'models': str(tmp_path / 'models'),
              'machine': {'user': 'user', 'workspace': '/workspace/jobs', 'uv': '/home/user/.local/bin/uv'},
              'provisioning': {'provider': 'scaleway', 'registry': str(tmp_path / 'registry'), 'profile': 'test',
                               'project': 'project', 'zone': 'zone',
                               'specification': {'type': 'gpu', 'image': 'image', 'root_volume': 'volume'}}}
    if not provision:
        config.pop('provisioning')
        config['machine']['host'] = 'training.example.org'
    config_path = tmp_path / 'configuration.json'
    write(config_path, config)
    splits = tmp_path / 'splits'
    splits.mkdir()
    (splits / 'train.jsonl').write_text('fixture')
    commands = []
    class Resources:
        def __init__(self, *args):
            assert provision
        def intent(self, *args):
            return {'token': 'instance'}
        def allocate(self, record):
            return {**record, 'server': {'public_ip': {'address': '192.0.2.1'}}}
        def release(self, token):
            assert token == 'instance'
    class Remote:
        host = 'fixture'
        def __init__(self, address, machine):
            pass
        def command(self, args, **kwargs):
            commands.append(args)
            if 'create' in args:
                return json.dumps({'directory': '/scratch/runs/job'})
            if args[:2] == ['python3', '-c']:
                return json.dumps({'execution': {'status': 'failed'}, 'heartbeat_age': 0})
            return ''
        def upload(self, *args):
            pass
        def receive(self, *args, **kwargs):
            pass
    monkeypatch.setattr(remote_training.machines, 'provider', lambda config: None)
    monkeypatch.setattr(remote_training.machines, 'Resources', Resources)
    monkeypatch.setattr(remote_training, 'Remote', Remote)
    monkeypatch.setattr(remote_training, 'bootstrap', lambda *args: None)
    monkeypatch.setattr(remote_training, 'collect_checkpoint', lambda *args: [])
    with pytest.raises(JobError, match='did not complete'):
        remote_training.train(config_path, '12b', splits, 'unused', tmp_path / 'instances', pilot=pilot, sleep=lambda _: None)
    create = next(command for command in commands if 'create' in command)
    assert ('--pilot' in create) is pilot
    binding = json.loads(next((tmp_path / 'instances').glob('*/binding.json')).read_text())
    assert binding['pilot'] is pilot


class ExportMachine:
    """A training machine whose export has already been launched or finishes on the given observations."""
    host = 'fixture'

    def __init__(self, observations, gguf_files=('model-Q4_K_M.gguf',)):
        self.observations, self.gguf_files = list(observations), gguf_files
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
        from bialy.provenance import file_sha256
        self.received.append((source, tuple(exclude)))
        destination.mkdir(parents=True, exist_ok=True)
        if source.endswith('merged/manifest.json'):
            write(destination / 'manifest.json', {'files': 'merged'})
        elif source.endswith('/gguf/'):
            for name in self.gguf_files:
                (destination / name).write_bytes(name.encode())
            job = destination.parent
            write(destination / 'manifest.json', {
                'schema': 'pw-custom-gguf/1', 'load_probe': 'passed', 'llama_cpp_revision': 'c' * 40,
                'merged_manifest_sha256': file_sha256(job / 'merged/manifest.json'),
                'unquantized': {'sha256': 'retained-on-machine'},
                'quantized': {'sha256': file_sha256(destination / 'model-Q4_K_M.gguf')}})


def export_fixture(tmp_path, monkeypatch):
    from bialy.student_host import exporting
    from bialy.student_host.specification import load
    config = {'llama_commit': 'c' * 40, 'machine': {'uv': '/opt/uv/bin/uv'}}
    destination = tmp_path / 'instance'
    write_plan = destination / 'job'
    write_plan.mkdir(parents=True)
    write(write_plan / 'plan.json', {'specification': load(), 'student': '26b-a4b', 'specification_sha256': 'spec',
                                     'host_commit': 'host'})
    monkeypatch.setattr(exporting, 'validate_inputs', lambda plan: None)
    monkeypatch.setattr(exporting, 'validate_result', lambda plan: None)
    return config, destination


def test_export_runs_on_the_training_machine_and_returns_only_the_served_gguf(tmp_path, monkeypatch):
    import subprocess
    config, destination = export_fixture(tmp_path, monkeypatch)
    machine = ExportMachine([{'exit': None, 'alive': True}, subprocess.TimeoutExpired('probe', 30), {'exit': 0, 'alive': False}])
    artifact = remote_training.export_remotely(machine, config, '/w/job', '/w/job/runs/one', destination, lambda: None, lambda _: None)
    launch = next(args for args in machine.commands if args[:2] == ['bash', '-lc'])[2]
    outer = shlex.split(launch)
    assert outer[:3] == ['nohup', 'bash', '-c']
    inner = shlex.split(outer[3].split(';')[0])
    assert inner[-6:] == ['export', '/w/job/runs/one', '--llama', '/w/job/llama.cpp', '--revision', 'c' * 40]
    assert json.loads((destination / 'export.json').read_text()) == {'pid': 4242}
    assert ('/w/job/runs/one/gguf/', ('model-f16.gguf',)) in machine.received
    assert artifact['unquantized_retained'] is False and set(artifact['files']) == {'model-Q4_K_M.gguf', 'manifest.json'}
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
