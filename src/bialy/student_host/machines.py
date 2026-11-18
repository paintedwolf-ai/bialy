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
from .resources import NoOffer, ResourceError, Resources, write

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
POLL_SECONDS = 10
# How long a machine that answers SSH may go on refusing the registered key. The provider
# installs the key before the daemon starts, so a refusal does not heal the way a
# connection that is not yet accepted does; two minutes covers a daemon restarted at boot.
REFUSED_SECONDS = 120
# How long an existing machine may take to answer.
EXISTING_SECONDS = 600
# Machines one job may rent before any work has started on one. Each unusable machine is
# passed over afterwards, so a third failure in a row points at the image, the key or the
# account, which another machine does not fix.
RENT_ATTEMPTS = 3
# An offer search that finds nothing is repeated every five minutes for an hour: offers
# return within minutes when a rental ends, and an hour without one means the
# specification or its price cap no longer matches the market.
OFFER_WAIT_SECONDS = 300
OFFER_WAITS = 12


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
    common = {'provider', 'registry', 'rental', 'lifetime_hours', 'start_minutes', 'specification'}
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
    # From creation until the instance answers SSH: the image pull on this class of host.
    if type(provision['start_minutes']) is not int or not 1 <= provision['start_minutes'] <= 60:
        raise JobError('start_minutes must bound the wait for a new instance to between 1 and 60 minutes')
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
        try:
            report['cli'] = provider(configuration['provisioning']).check()
        except ResourceError as exc:
            raise JobError(str(exc)) from exc
    return report


def startup(provision):
    """The command a rented machine runs at every boot: install the dead-man switch and arm it."""
    text = ("#!/bin/bash\ncat >%s <<'DEADMAN'\n%sDEADMAN\nchmod 700 %s\n%s arm %s %d %d\n"
            % (DEADMAN, DEADMAN_SCRIPT.read_text(), DEADMAN, DEADMAN, provision['provider'], STARTUP_SECONDS,
               provision['lifetime_hours'] * 3600))
    if provision['provider'] == 'vastai' and len(text.encode()) > vastai.STARTUP_LIMIT:
        raise JobError('startup command exceeds the provider limit')
    return text


def inspect(remote, configuration, needed, directory):
    """Measure the machine against `needed` and record it; a rented one is completed first."""
    machine = configuration['machine']
    mode = 'install' if 'provisioning' in configuration else 'check'
    arguments = [mode, machine['workspace'], machine['uv'], machine.get('device', 'cuda:0'),
                 *[str(needed[name]) for name in ('disk_gib', 'memory_gib', 'gpu_memory_gib')]]
    status, output = remote.script(PREFLIGHT_SCRIPT, arguments, timeout=1800)
    try:
        report = json.loads(output.strip().splitlines()[-1])
    except (ValueError, IndexError):
        raise JobError('training machine preflight did not report') from None
    write(Path(directory) / 'machine.json', {'requirement': needed, **report})
    if status and not report['problems']:
        report['problems'] = ['preflight failed']
    return report


def measure(remote, configuration, student, directory):
    """Check the machine against the tier before any work starts on it."""
    report = inspect(remote, configuration, requirement(student), directory)
    if report['problems']:
        raise JobError('training machine cannot run student %s: %s' % (student, '; '.join(report['problems'])))
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
            sleep(POLL_SECONDS)
        raise Unarmed('dead-man switch is not armed on the rented machine')

    def quiet(self):
        """Stop the heartbeats and leave the lease renewed: the machine's own deadline runs out."""
        self.beaten = None

    def abandon(self):
        """Stop renewing the lease and keep the heartbeats: the reaper's deadline runs out."""
        self.stop.set()
        self.thread.join()

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
        if self.thread.is_alive():
            self.thread.join()


class Unusable(JobError):
    """A rented machine no work can start on. It is released and passed over."""


class Unreachable(Unusable):
    """The machine never reported an endpoint, never answered on it, or refused the registered key."""


class Unarmed(Unusable):
    """The machine's dead-man switch cannot end its rental."""


class Refused(JobError):
    """The machine's SSH daemon answered and refused the registered key."""


class Unsettled(JobError):
    """The ledger or the account holds rentals that are not accounted for."""


def endpoint(resources, token, supervisor, sleep, tries):
    """Wait for the leased instance to report where it can be reached. Returns the
    endpoint and how many of `tries` are left."""
    for used in range(tries):
        supervisor.check()
        try:
            instance = resources.observe(token)
        except (ConnectionError, subprocess.TimeoutExpired):
            instance = None
        if instance and instance['interruptible']:
            raise JobError('provider returned an interruptible instance')
        if instance and instance['endpoint']:
            return instance['endpoint'], tries - used
        sleep(POLL_SECONDS)
    raise Unreachable('rented machine did not start within its start_minutes')


def reachable(remote, supervisor, sleep, tries):
    """Wait for the machine to run a command. A connection not yet accepted is tried until
    `tries` run out; a refused key only for REFUSED_SECONDS."""
    refused = 0
    for _ in range(max(tries, 1)):
        supervisor.check()
        try:
            remote.command(['true'], timeout=30)
            return
        except Refused:
            refused += 1
            if refused * POLL_SECONDS > REFUSED_SECONDS:
                raise Unreachable('training machine refuses the registered SSH key') from None
        except (JobError, subprocess.TimeoutExpired):
            refused = 0
        sleep(POLL_SECONDS)
    raise Unreachable('training machine did not become reachable')


class Rental:
    """The job's own record of every machine it leased: `rental.json`, one entry an attempt."""

    def __init__(self, directory, clock=time.time):
        self.path, self.clock = Path(directory) / 'rental.json', clock

    def attempts(self):
        return json.loads(self.path.read_text())['attempts'] if self.path.exists() else []

    def open(self):
        """The attempt whose lease is still held, or None."""
        attempts = self.attempts()
        return attempts[-1] if attempts and attempts[-1]['outcome'] is None else None

    def count(self, *outcomes):
        return sum(attempt['outcome'] in outcomes for attempt in self.attempts())

    def begin(self, token):
        write(self.path, {'attempts': [*self.attempts(), {
            'token': token, 'started_at': self.clock(), 'ended_at': None, 'outcome': None, 'reason': None,
            'instance': None, 'host': None, 'price_per_hour': None, 'cleanup': None}]})

    def update(self, token, **facts):
        attempts = self.attempts()
        for attempt in attempts:
            if attempt['token'] == token:
                attempt.update(facts)
        write(self.path, {'attempts': attempts})

    def leased(self, lease):
        """Record the instance behind a lease and the hourly price it is charged at."""
        instance = next((item for item in lease['instances'] if item['id'] == lease['instance']), {})
        self.update(lease['token'], instance=lease['instance'], host=instance.get('host'),
                    price_per_hour=instance.get('price_per_hour') or (lease.get('offer') or {}).get('price_per_hour'))


def release(configuration, directory, sleep=time.sleep, clock=time.time):
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
    rental = Rental(directory, clock)
    attempt = next((item for item in rental.attempts() if item['token'] == token), None)
    if attempt:
        rental.update(token, cleanup=status, ended_at=attempt['ended_at'] or clock(), outcome=attempt['outcome'] or 'used')
    return status


def refuse_unsettled(resources):
    """Settle what the ledger can, then refuse to rent on top of anything left over."""
    try:
        found = resources.unsettled()
    except (ResourceError, ConnectionError, subprocess.TimeoutExpired) as exc:
        raise Unsettled('the rental ledger could not be checked against the provider') from exc
    if found['leases'] or found['instances']:
        raise Unsettled('the rental ledger is not settled; leases awaiting a confirmed release: %s; labelled instances no live '
                        'lease owns: %s' % (', '.join('%s (%s)' % (item['token'], item['state']) for item in found['leases']) or 'none',
                                            ', '.join('%s (%s)' % (item['id'], item['state']) for item in found['instances']) or 'none'))


def lease_machine(resources, provision, machine, token, held, supervisor, remote_type, sleep, stage):
    """The leased instance, reachable and with its dead-man switch armed."""
    def lease():
        if held and json.loads(resources.path(token).read_text())['state'] == 'active':
            return resources.recover(token, LEASE_SECONDS)
        return resources.allocate(token, startup(provision))
    try:
        allocation = lease()
    except (ConnectionError, subprocess.TimeoutExpired):
        allocation = lease()
    stage('boot')
    address, left = endpoint(resources, token, supervisor, sleep, provision['start_minutes'] * 60 // POLL_SECONDS)
    remote = remote_type(address['address'], {**machine, 'port': address['port']})
    reachable(remote, supervisor, sleep, left)
    supervisor.attach(remote, sleep)
    return allocation, remote


@contextlib.contextmanager
def connect(configuration, directory, remote_type, sleep=time.sleep, supervisor_type=Supervisor, *, replaceable=True,
            stage=lambda name: None, clock=time.time):
    """A reachable machine and its supervisor. A rented machine is released on the way out,
    whatever happened; `cleanup.json` records whether the provider confirmed it.

    A lease this job already holds is taken up again. A machine that proves unusable
    before any work starts is released, passed over, and replaced by the next offer, up
    to RENT_ATTEMPTS machines; the next is leased only once the provider confirmed the
    last one gone. A held machine that is gone is replaced only while `replaceable`."""
    provision, machine, directory = configuration.get('provisioning'), configuration['machine'], Path(directory)
    try:
        if provision is None:
            remote, supervisor = remote_type(machine['host'], machine), Unsupervised()
            reachable(remote, supervisor, sleep, EXISTING_SECONDS // POLL_SECONDS)
            yield remote, supervisor
            return
        resources, rental, path = Resources(provision['registry'], provider(provision)), Rental(directory, clock), directory / 'resource.json'
        while True:
            stage('rent')
            held = rental.open() is not None
            if not held:
                if not replaceable:
                    raise JobError('the machine this job ran on is gone, and a run does not move to another machine')
                refuse_unsettled(resources)
                # A confirmed release of the machine before this one says nothing about this one.
                (directory / 'cleanup.json').unlink(missing_ok=True)
                # The token is on disk here before the provider is asked for anything.
                write(path, resources.intent(provision['specification'], LEASE_SECONDS, provision['lifetime_hours'] * 3600))
                rental.begin(json.loads(path.read_text())['token'])
            token = json.loads(path.read_text())['token']
            with supervisor_type(resources, token) as supervisor:
                try:
                    allocation, remote = lease_machine(resources, provision, machine, token, held, supervisor, remote_type, sleep, stage)
                except NoOffer as exc:
                    failure, outcome = exc, 'no_offer'
                except Unusable as exc:
                    resources.avoid(token)
                    failure, outcome = exc, 'unusable'
                except ResourceError as exc:
                    # A held lease the reaper took, or whose instance the provider no longer lists.
                    if not held or not replaceable:
                        raise
                    failure, outcome = exc, 'lost'
                else:
                    write(path, allocation)
                    rental.leased(allocation)
                    yield remote, supervisor
                    return
                finally:
                    record = json.loads(resources.path(token).read_text())
                    if record.get('instance'):
                        rental.leased(record)
            rental.update(token, outcome=outcome, reason=str(failure))
            if release(configuration, directory, sleep, clock) != 'confirmed':
                raise JobError('a %s rental was not confirmed released, so no other machine is rented: %s' % (outcome, failure)) from failure
            if outcome == 'no_offer':
                if rental.count('no_offer') > OFFER_WAITS:
                    raise NoOffer('no matching offer in %d searches over %d minutes'
                                  % (OFFER_WAITS + 1, OFFER_WAITS * OFFER_WAIT_SECONDS // 60)) from failure
                sleep(OFFER_WAIT_SECONDS)
            elif rental.count('unusable', 'lost') >= RENT_ATTEMPTS:
                raise failure
    finally:
        release(configuration, directory, sleep, clock)
