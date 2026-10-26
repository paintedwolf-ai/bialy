"""Lease-owned remote training with continuous checkpoint replication."""
import fcntl
import ipaddress
import json
import shlex
import subprocess
import threading
import time
import uuid
from pathlib import Path

from .jobs import JobError, tree, validate_inputs, validate_result, validate_training_inputs
from .reaper import provider
from .resources import Resources, write
from .specification import identity
from .transfer import checkpoints

SSH = ['ssh', '-o', 'BatchMode=yes', '-o', 'StrictHostKeyChecking=accept-new',
       '-o', 'ConnectTimeout=10', '-o', 'ServerAliveInterval=20', '-o', 'ServerAliveCountMax=3']


def validate_configuration(configuration):
    required = {'profile', 'project', 'zone', 'registry', 'host_repository', 'host_commit',
                'bialy_repository', 'bialy_commit', 'host_checkout', 'models', 'rental'}
    if set(configuration) != required:
        raise JobError('rental configuration fields do not match the deployment contract')
    for name in required - {'rental'}:
        if not isinstance(configuration[name], str) or not configuration[name]:
            raise JobError('rental configuration value is missing: ' + name)
    for name in ('host_commit', 'bialy_commit'):
        value = configuration[name]
        if len(value) != 40 or any(char not in '0123456789abcdef' for char in value):
            raise JobError('remote source must be pinned to a full commit')
    for name in ('registry', 'host_checkout', 'models'):
        if not Path(configuration[name]).is_absolute():
            raise JobError('deployment paths must be absolute')
    specification = configuration['rental']
    if (not isinstance(specification, dict) or set(specification) != {'type', 'image', 'root_volume'}
            or any(not isinstance(value, str) or not value for value in specification.values())):
        raise JobError('rental specification is incomplete')


class Remote:
    def __init__(self, address, run=subprocess.run):
        self.host = 'root@' + str(ipaddress.ip_address(address))
        self.run = run

    def command(self, args, timeout=600):
        result = self.run([*SSH, self.host, shlex.join(list(map(str, args)))], capture_output=True,
                          text=True, timeout=timeout)
        if result.returncode:
            raise JobError('remote operation failed with status %d' % result.returncode)
        return result.stdout

    def upload(self, source, destination):
        self.run(['rsync', '-a', '-e', shlex.join(SSH), str(source), self.host + ':' + destination], check=True, timeout=600)

    def receive(self, source, destination):
        Path(destination).mkdir(parents=True, exist_ok=True)
        self.run(['rsync', '-a', '--exclude', '.pending-*', '-e', shlex.join(SSH), self.host + ':' + source,
                  str(destination) + '/'], check=True, timeout=600)


def bootstrap(remote, configuration):
    for name in ('host_commit', 'bialy_commit'):
        value = configuration[name]
        if len(value) != 40 or any(char not in '0123456789abcdef' for char in value):
            raise JobError('remote source must be pinned to a full commit')
    remote.command(['bash', '-lc', 'command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh'])
    for name, repository, revision in (('host', configuration['host_repository'], configuration['host_commit']),
                                       ('bialy', configuration['bialy_repository'], configuration['bialy_commit'])):
        destination = '/scratch/' + name
        remote.command(['git', 'clone', '--no-checkout', repository, destination])
        remote.command(['git', '-C', destination, 'fetch', 'origin', revision])
        remote.command(['git', '-C', destination, 'checkout', '--detach', revision])
    remote.command(['/root/.local/bin/uv', 'sync', '--project', '/scratch/bialy', '--locked'])
    remote.command(['/root/.local/bin/uv', 'sync', '--project', '/scratch/host/scripts/bialy/custom_model', '--locked'])
    remote.command(['/root/.local/bin/uv', 'tool', 'install', 'huggingface_hub==2.2.0'])
    remote.command(['bash', '-lc', 'PATH=/root/.local/bin:$PATH uv run --project /scratch/bialy --locked '
                    'python -m bialy.student_host models --root /scratch/models'], timeout=3600)


class Renewal:
    def __init__(self, resources, token, seconds):
        self.resources, self.token, self.seconds = resources, token, seconds
        self.stop = threading.Event()
        self.error = None
        self.thread = threading.Thread(target=self.loop, daemon=True)

    def loop(self):
        while not self.stop.wait(20):
            try:
                self.resources.renew(self.token, self.seconds)
            except Exception as exc:
                self.error = exc
                return

    def check(self):
        if self.error:
            raise JobError('rental lease renewal failed') from self.error

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.stop.set()
        self.thread.join()


def collect_checkpoint(remote, remote_directory, destination):
    probe = ('import json,pathlib; p=pathlib.Path(' + repr(remote_directory) + '); '
             'i=p/"training/inputs.json"; print(json.dumps({"plan":json.loads((p/"plan.json").read_text()), '
             '"inputs":json.loads(i.read_text()) if i.exists() else None}))')
    snapshot = json.loads(remote.command(['python3', '-c', probe]))
    if snapshot['inputs'] is None:
        return []
    validate_training_inputs(snapshot['plan'], snapshot['inputs'])
    binding = Path(destination) / 'checkpoint-inputs.json'
    if binding.exists() and json.loads(binding.read_text()) != snapshot:
        raise JobError('remote checkpoint inputs changed')
    write(binding, snapshot)
    incoming = Path(destination) / 'incoming'
    remote.receive(remote_directory + '/training/checkpoints/', incoming)
    return checkpoints(incoming, Path(destination) / 'checkpoints', identity(snapshot['inputs']))


def train(configuration_path, student, splits, source_key, out, *, resume=None, sleep=time.sleep):
    destination = Path(resume) if resume else Path(out) / (student + "-" + uuid.uuid4().hex)
    if not resume:
        destination.mkdir(parents=True, exist_ok=False)
    with (destination / ".controller.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise JobError("another controller owns this rental") from exc
        try:
            return train_owned(configuration_path, student, splits, source_key, destination, resume, sleep)
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def train_owned(configuration_path, student, splits, source_key, destination, resume, sleep):
    configuration = json.loads(Path(configuration_path).read_text())
    validate_configuration(configuration)
    resources = Resources(configuration['registry'], provider(configuration))
    binding = {'configuration': identity(configuration), 'student': student, 'splits': tree(splits)}
    if resume:
        if json.loads((destination / 'binding.json').read_text()) != binding:
            raise JobError('rental restart inputs changed')
        record = json.loads((destination / 'resource.json').read_text())
        outcome = json.loads((destination / 'outcome.json').read_text())
        if outcome['status'] == 'complete' and outcome.get('cleanup') == 'confirmed':
            plan = json.loads((destination / 'job/plan.json').read_text())
            validate_inputs(plan)
            validate_result(plan)
            return {'directory': str(destination), **outcome}
        if outcome['status'] != 'running':
            raise JobError('rental has a terminal outcome')
    else:
        write(destination / 'binding.json', binding)
        record = resources.intent(configuration['rental'], 300)
        write(destination / 'resource.json', record)
        outcome = {'schema': 'pw-rented-training/1', 'student': student, 'token': record['token'], 'status': 'running'}
        write(destination / 'outcome.json', outcome)
    try:
        with Renewal(resources, record['token'], 300) as renewal:
            try:
                allocation = resources.recover(record['token'], 300) if resume else resources.allocate(record)
            except (ConnectionError, subprocess.TimeoutExpired):
                allocation = resources.allocate(record)
            write(destination / 'resource.json', allocation)
            server = allocation['server']
            address = (server.get('public_ip') or server['public_ips'][0])['address']
            remote = Remote(address)
            ready = False
            for _ in range(60):
                renewal.check()
                try:
                    remote.command(['true'], timeout=30)
                    ready = True
                    break
                except (JobError, subprocess.TimeoutExpired):
                    sleep(10)
            if not ready:
                raise JobError('rental did not become reachable')
            if resume:
                remote_directory = json.loads((destination / 'remote.json').read_text())['directory']
            else:
                bootstrap(remote, configuration)
                renewal.check()
                remote.command(['mkdir', '-p', '/scratch/data', '/scratch/private', '/scratch/runs', '/scratch/spool'])
                remote.upload(str(Path(splits)) + '/', '/scratch/data/')
                remote.upload(source_key, '/scratch/private/source.key')
                remote.command(['chmod', '600', '/scratch/private/source.key'])
                prefix = ['/root/.local/bin/uv', 'run', '--project', '/scratch/bialy', '--locked', 'python', '-m', 'bialy.student_host']
                created = json.loads(remote.command([*prefix, 'create', '--student', student, '--root', '/scratch/runs',
                                                     '--checkout', '/scratch/host', '--models', '/scratch/models',
                                                     '--splits', '/scratch/data', '--mirror', '/scratch/spool']))
                remote_directory = created['directory']
                write(destination / 'remote.json', {'host': remote.host, 'directory': remote_directory})
                command = shlex.join(['env', 'PATH=/root/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin',
                                     'PW_TRAINING_SOURCE_KEY=/scratch/private/source.key', *prefix, 'train', remote_directory])
                script = 'nohup ' + command + ' >' + shlex.quote(remote_directory + '/launcher.log') + ' 2>&1 </dev/null &'
                remote.command(['bash', '-lc', script])
            transport_failures = 0
            while True:
                renewal.check()
                try:
                    probe = ('import json,time,pathlib; p=pathlib.Path(' + repr(remote_directory) + '); '
                             's=json.loads((p/"execution.json").read_text()); '
                             'h=json.loads((p/"heartbeat.json").read_text()); '
                             'print(json.dumps({"execution":s,"heartbeat_age":time.time()-h["at"]}))')
                    observation = json.loads(remote.command(['python3', '-c', probe], timeout=30))
                    state = observation['execution']
                    if state['status'] not in {'complete', 'failed'} and observation['heartbeat_age'] > 90:
                        raise JobError('remote training worker lost')
                    try:
                        collect_checkpoint(remote, remote_directory, destination)
                    except subprocess.CalledProcessError:
                        if state['status'] == 'complete':
                            raise
                    transport_failures = 0
                except (JobError, subprocess.TimeoutExpired, subprocess.CalledProcessError):
                    transport_failures += 1
                    if transport_failures > 2:
                        raise JobError('remote training observation unavailable') from None
                    sleep(20)
                    continue
                if state['status'] in {'complete', 'failed'}:
                    remote.receive(remote_directory + '/', destination / 'job')
                    if state['status'] == 'complete':
                        plan_path = destination / 'job/plan.json'
                        plan = json.loads(plan_path.read_text())
                        write(destination / 'job/remote-plan.json', plan)
                        plan.update(checkout=configuration['host_checkout'], models=configuration['models'],
                                    splits=str(Path(splits).resolve()), out=str((destination / 'job/training').resolve()),
                                    mirror=str((destination / 'checkpoints').resolve()))
                        write(plan_path, plan)
                        validate_inputs(plan)
                        validate_result(plan)
                    outcome.update(status=state['status'], execution=state)
                    break
                sleep(20)
    except Exception as exc:
        outcome.update(status='failed', failure={'type': type(exc).__name__, 'diagnostic': str(exc)})
    finally:
        try:
            resources.release(record['token'])
            outcome['cleanup'] = 'confirmed'
        except Exception as exc:
            outcome['cleanup'] = 'unconfirmed'
            outcome['cleanup_failure'] = type(exc).__name__
        write(destination / 'outcome.json', outcome)
    if outcome['status'] != 'complete' or outcome['cleanup'] != 'confirmed':
        raise JobError('rented training did not complete; evidence: ' + str(destination))
    return {'directory': str(destination), **outcome}
