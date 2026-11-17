"""Training machines: what a tier needs from one, the checks made before one is rented or
used, and the supervised connection to an existing or a rented machine."""
import contextlib
import json
import re
import subprocess
import threading
import time
from pathlib import Path, PurePosixPath

from . import scaleway, vastai
from .jobs import JobError
from .reaper import provider
from .resources import ResourceError, Resources, write

ROOT = Path(__file__).resolve().parents[3]
REQUIREMENTS = ROOT / 'config/student-requirements.json'
DEADMAN_SCRIPT = ROOT / 'train-host/student/deadman.sh'
PREFLIGHT_SCRIPT = ROOT / 'train-host/student/machine-preflight.sh'
DEADMAN = '/usr/local/sbin/pw-deadman'
CAPACITY = ('gpu_memory_gib', 'memory_gib', 'disk_gib')

# The controller-side lease and the machine-side deadline are renewed every 20 seconds.
# The lease is short because the reaper acts on it; the deadline tolerates a longer
# silence because nothing on the machine can tell a slow controller from a dead one.
LEASE_SECONDS = 300
HEARTBEAT_SECONDS = 900
# The dead-man switch's first deadline, from boot until the controller's first heartbeat.
STARTUP_SECONDS = 1800


def requirement(student, path=REQUIREMENTS):
    """What one tier needs from the machine that trains and exports it."""
    document = json.loads(Path(path).read_text())
    tier = document.get('tiers', {}).get(student)
    if (document.get('schema') != 'pw-student-requirements/1' or not isinstance(tier, dict) or set(tier) != set(CAPACITY)
            or any(type(tier[name]) is not int or tier[name] <= 0 for name in CAPACITY)):
        raise JobError('no machine requirement is declared for student ' + student)
    return tier


def validate_connection(machine, provisioned):
    if (not isinstance(machine, dict) or not {'user', 'workspace', 'uv'} <= set(machine)
            or set(machine) - {'user', 'workspace', 'uv', 'host', 'port', 'identity_file', 'device'}):
        raise JobError('machine connection fields are invalid')
    if not re.fullmatch('[a-zA-Z_][a-zA-Z0-9_-]*', machine['user']):
        raise JobError('invalid SSH user')
    for name in ('workspace', 'uv'):
        path = PurePosixPath(machine[name])
        if not path.is_absolute() or '..' in path.parts or str(path) == '/':
            raise JobError('remote paths must be absolute and scoped')
    if type(machine.get('port', 22)) is not int or not 0 < machine.get('port', 22) < 65536:
        raise JobError('invalid SSH port')
    if machine.get('identity_file') and not Path(machine['identity_file']).is_file():
        raise JobError('SSH identity file is unavailable')
    if not re.fullmatch(r'cpu|cuda(:\d+)?', machine.get('device', 'cuda:0')):
        raise JobError('invalid training device')
    if provisioned:
        # Setup installs packages and the dead-man switch ends the rental; both need root.
        if 'host' in machine or 'port' in machine or machine['user'] != 'root':
            raise JobError('a rented machine is reached as root at the address its provider reports')
    elif not re.fullmatch('[a-zA-Z0-9][a-zA-Z0-9.:-]*', machine.get('host', '')):
        raise JobError('machine host is missing or invalid')


def validate_provisioning(provision):
    names = {'scaleway': {'profile', 'project', 'zone'}, 'vastai': set()}
    common = {'provider', 'registry', 'rental', 'lifetime_hours', 'specification'}
    if not isinstance(provision, dict) or provision.get('provider') not in names:
        raise JobError('unsupported instance provider')
    name = provision['provider']
    optional = {'api_key_file'} if name == 'vastai' else set()
    if not common | names[name] <= set(provision) or set(provision) - common - names[name] - optional:
        raise JobError('provisioning configuration is invalid')
    if not Path(str(provision['registry'])).is_absolute():
        raise JobError('resource registry must be an absolute path')
    # A provider may reclaim an interruptible machine mid-run, and a run does not move to another machine.
    if provision['rental'] != 'on-demand':
        raise JobError('machines are rented on demand; interruptible rentals are refused')
    if type(provision['lifetime_hours']) is not int or not 1 <= provision['lifetime_hours'] <= 336:
        raise JobError('lifetime_hours must bound the rental to between 1 and 336 hours')
    if any(not isinstance(provision[key], str) or not provision[key] for key in names[name]):
        raise JobError('instance provider configuration is incomplete')
    if provision.get('api_key_file') and not Path(provision['api_key_file']).is_file():
        raise JobError('Vast.ai API key file is unavailable')
    spec = provision['specification']
    required = vastai.REQUIRED_SPECIFICATION if name == 'vastai' else scaleway.SPECIFICATION
    allowed = vastai.ALLOWED_SPECIFICATION if name == 'vastai' else scaleway.SPECIFICATION
    if not isinstance(spec, dict) or not required <= set(spec) or set(spec) - allowed:
        raise JobError('instance specification fields are invalid')
    integers = {*CAPACITY, 'num_gpus', 'download_mbps'} & set(spec)
    strings = set(spec) - integers - {'max_price'}
    if (any(type(spec[key]) is not int or spec[key] <= 0 for key in integers)
            or any(not isinstance(spec[key], str) or not spec[key] for key in strings)
            or ('max_price' in spec and (type(spec['max_price']) not in (int, float) or not 0 < spec['max_price'] < 100))):
        raise JobError('instance specification is incomplete')
    # An image named without a tag or digest, or as `latest`, is a different image each time it is pulled.
    if name == 'vastai' and not re.fullmatch(r'[^\s:@]+(:(?!latest$)[\w.-]+|@sha256:[0-9a-f]{64})', spec['image']):
        raise JobError('instance image must be pinned to a tag or digest')


def validate(configuration):
    required = {'host_repository', 'host_commit', 'bialy_repository', 'bialy_commit', 'llama_repository', 'llama_commit',
                'host_checkout', 'models', 'machine'}
    if not required <= set(configuration) or set(configuration) - required - {'provisioning'}:
        raise JobError('remote training configuration fields are invalid')
    for name in required - {'machine'}:
        if not isinstance(configuration[name], str) or not configuration[name]:
            raise JobError('remote training configuration value is missing: ' + name)
    for name in ('host_commit', 'bialy_commit', 'llama_commit'):
        if not re.fullmatch('[0-9a-f]{40}', configuration[name]):
            raise JobError('remote source must be pinned to a full commit')
    if any(not Path(configuration[name]).is_absolute() for name in ('host_checkout', 'models')):
        raise JobError('local artifact paths must be absolute')
    validate_connection(configuration['machine'], 'provisioning' in configuration)
    if 'provisioning' in configuration:
        validate_provisioning(configuration['provisioning'])


def preflight(configuration, student):
    """Refuse, without contacting anything, a configuration whose machine cannot run the tier.
    An existing machine declares no capacity; it is measured when the job reaches it."""
    validate(configuration)
    needed = requirement(student)
    report = {'student': student, 'requirement': needed, 'declared': None}
    if 'provisioning' in configuration:
        spec = configuration['provisioning']['specification']
        report['declared'] = {name: spec[name] for name in CAPACITY}
        short = ['%s is %d, the tier needs %d' % (name, spec[name], needed[name]) for name in CAPACITY if spec[name] < needed[name]]
        if short:
            raise JobError('machine specification cannot hold student %s: %s' % (student, '; '.join(short)))
        startup(configuration['provisioning'])
    return report


def startup(provision):
    """The command a rented machine runs at every boot: install the dead-man switch and arm it."""
    text = ("#!/bin/bash\ncat >%s <<'DEADMAN'\n%sDEADMAN\nchmod 700 %s\n%s arm %s %d %d\n"
            % (DEADMAN, DEADMAN_SCRIPT.read_text(), DEADMAN, DEADMAN, provision['provider'], STARTUP_SECONDS,
               provision['lifetime_hours'] * 3600))
    if provision['provider'] == 'vastai' and len(text.encode()) > vastai.STARTUP_LIMIT:
        raise JobError('startup command exceeds the provider limit')
    return text


def measure(remote, configuration, student, directory):
    """Check the machine against the tier before any work starts on it; a rented one is completed first."""
    needed, machine = requirement(student), configuration['machine']
    mode = 'install' if 'provisioning' in configuration else 'check'
    arguments = [mode, machine['workspace'], machine['uv'], machine.get('device', 'cuda:0'),
                 *[str(needed[name]) for name in ('disk_gib', 'memory_gib', 'gpu_memory_gib')]]
    status, output = remote.script(PREFLIGHT_SCRIPT, arguments, timeout=1800)
    try:
        report = json.loads(output.strip().splitlines()[-1])
    except (ValueError, IndexError):
        raise JobError('training machine preflight did not report') from None
    write(Path(directory) / 'machine.json', {'requirement': needed, **report})
    if status or report['problems']:
        raise JobError('training machine cannot run student %s: %s' % (student, '; '.join(report['problems']) or 'preflight failed'))
    return report


class Unsupervised:
    """An existing machine: nothing is leased, so nothing is renewed or ended."""

    def check(self):
        pass

    def lost(self):
        return None


class Supervisor:
    """Keeps the lease and the machine's dead-man deadline alive from its own thread, so a
    long blocking step in the controller does not end the rental."""

    def __init__(self, resources, token, clock=time.monotonic):
        self.resources, self.token, self.clock = resources, token, clock
        self.stop, self.error, self.remote, self.beaten = threading.Event(), None, None, None
        self.thread = threading.Thread(target=self.loop, daemon=True)

    def beat(self):
        """Extend the machine's deadline. Only a switch that can still end the rental counts."""
        status = json.loads(self.remote.command([DEADMAN, 'extend', str(HEARTBEAT_SECONDS)], timeout=30))
        armed = (status['expired'] is False and status['can_end_rental'] is True and status['watcher_age'] <= 30
                 and status['now'] < status['deadline'] <= status['limit'])
        if armed:
            self.beaten = self.clock()
        return armed

    def attach(self, remote, sleep):
        """Start heartbeats once the switch armed at boot is confirmed able to end the rental."""
        self.remote = remote
        for _ in range(30):
            self.check()
            try:
                if self.beat():
                    return
            except (JobError, subprocess.TimeoutExpired, ValueError, KeyError):
                pass
            sleep(10)
        raise JobError('dead-man switch is not armed on the rented machine')

    def loop(self):
        while not self.stop.wait(20):
            try:
                self.resources.renew(self.token, LEASE_SECONDS)
            except Exception as exc:
                self.error = exc
                return
            if self.beaten is not None:
                try:
                    self.beat()
                except Exception:
                    # A missed heartbeat is judged by its age in check().
                    pass

    def check(self):
        if self.error:
            raise JobError('instance lease renewal failed') from self.error
        if self.beaten is not None and self.clock() - self.beaten > HEARTBEAT_SECONDS:
            raise JobError('rented machine stopped answering its dead-man heartbeat and has ended its rental')

    def lost(self):
        """Why the provider no longer runs the machine, or None while it does."""
        try:
            instance = self.resources.observe(self.token)
        except (ResourceError, ConnectionError, subprocess.TimeoutExpired):
            return None
        if instance is None:
            return 'the provider no longer lists it'
        return None if instance['state'] == 'running' else 'the provider reports it ' + instance['state']

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.stop.set()
        self.thread.join()


def endpoint(resources, token, supervisor, sleep):
    """Wait for the leased instance to report where it can be reached."""
    for _ in range(90):
        supervisor.check()
        try:
            instance = resources.observe(token)
        except (ConnectionError, subprocess.TimeoutExpired):
            instance = None
        if instance and instance['interruptible']:
            raise JobError('provider returned an interruptible instance')
        if instance and instance['endpoint']:
            return instance['endpoint']
        sleep(10)
    raise JobError('rented machine did not become ready')


def reachable(remote, supervisor, sleep):
    for _ in range(60):
        supervisor.check()
        try:
            remote.command(['true'], timeout=30)
            return
        except (JobError, subprocess.TimeoutExpired):
            sleep(10)
    raise JobError('training machine did not become reachable')


def release(configuration, directory, sleep=time.sleep):
    """Release the job's rented machine and record whether the provider confirmed it gone.
    An unconfirmed release stays with the reaper; calling this again settles the record."""
    directory, provision = Path(directory), configuration.get('provisioning')
    record = directory / 'resource.json'
    if provision is None or not record.exists():
        write(directory / 'cleanup.json', {'status': 'not_applicable'})
        return 'not_applicable'
    resources, token = Resources(provision['registry'], provider(provision)), json.loads(record.read_text())['token']
    status = 'unconfirmed'
    for attempt in range(6):
        try:
            resources.release(token)
            status = 'confirmed'
            break
        except Exception:
            if attempt < 5:
                sleep(20)
    write(directory / 'cleanup.json', {'status': status, 'token': token})
    return status


@contextlib.contextmanager
def connect(configuration, directory, resume, remote_type, sleep=time.sleep, supervisor_type=Supervisor):
    """A reachable machine and its supervisor. A rented machine is released on the way out,
    whatever happened; `cleanup.json` records whether the provider confirmed it."""
    provision, machine = configuration.get('provisioning'), configuration['machine']
    try:
        if provision is None:
            remote, supervisor = remote_type(machine['host'], machine), Unsupervised()
            reachable(remote, supervisor, sleep)
            yield remote, supervisor
            return
        resources = Resources(provision['registry'], provider(provision))
        path = directory / 'resource.json'
        if not resume:
            # The token is on disk here before the provider is asked for anything.
            write(path, resources.intent(provision['specification'], LEASE_SECONDS, provision['lifetime_hours'] * 3600))
        token = json.loads(path.read_text())['token']
        with supervisor_type(resources, token) as supervisor:
            def lease():
                return resources.recover(token, LEASE_SECONDS) if resume else resources.allocate(token, startup(provision))
            try:
                allocation = lease()
            except (ConnectionError, subprocess.TimeoutExpired):
                allocation = lease()
            write(path, allocation)
            address = endpoint(resources, token, supervisor, sleep)
            remote = remote_type(address['address'], {**machine, 'port': address['port']})
            reachable(remote, supervisor, sleep)
            supervisor.attach(remote, sleep)
            yield remote, supervisor
    finally:
        release(configuration, directory, sleep)
