"""Remote training on configured machines with continuous checkpoint replication.

The controller's job directory is the record a stopped controller resumes from:
`rental.json` (every machine leased), `setup.json` (the setup steps done on the current
machine), `remote.json` (the job on it), `stages.json` and `outcome.json`.
"""
import fcntl
import json
import os
import shlex
import subprocess
import time
import uuid
from pathlib import Path

from . import exporting, machines
from .jobs import SCHEDULES, JobError, tree, validate_inputs, validate_result, validate_training_inputs
from .progress import Stages, rental, seconds
from .resources import NoOffer, write
from .specification import identity
from .transfer import checkpoints

RELEASED = {'confirmed', 'not_applicable'}
SSH = ['ssh', '-o', 'BatchMode=yes', '-o', 'StrictHostKeyChecking=accept-new',
       '-o', 'ConnectTimeout=10', '-o', 'ServerAliveInterval=20', '-o', 'ServerAliveCountMax=3']
# What a job leaves on its machine that never comes home: the merged checkpoint, both
# GGUFs, the MLX weights, and the checkpoints replicated while training.
BULK = ('/merged/model/', '/gguf/*.gguf', '/mlx/*.safetensors', '/mlx/*.npz', '/training/checkpoints/')
# No single file of a failed job's evidence is larger; the LoRA adapter is tens of megabytes.
EVIDENCE_FILE_LIMIT = '1g'
FAILURES = ((NoOffer, 'no_offer'), (machines.Unsettled, 'ledger_unsettled'), (machines.Unusable, 'machine_unusable'))


class Remote:
    def __init__(self, address, machine, run=subprocess.run):
        self.host = machine['user'] + '@' + address
        self.run = run
        self.ssh = [*SSH, '-p', str(machine.get('port', 22))]
        if machine.get('identity_file'):
            self.ssh += ['-i', machine['identity_file']]

    def command(self, args, timeout=600):
        result = self.run([*self.ssh, self.host, shlex.join(list(map(str, args)))], capture_output=True,
                          text=True, timeout=timeout)
        # ssh exits 255 for its own failures; a refused key is one the daemon answered.
        if result.returncode == 255 and 'Permission denied' in (result.stderr or ''):
            raise machines.Refused('remote machine refused the registered SSH key')
        if result.returncode:
            raise JobError('remote operation failed with status %d' % result.returncode)
        return result.stdout

    def script(self, path, args, timeout=600):
        """Run a local script on the machine through the shell's standard input; nothing needs to be installed there."""
        result = self.run([*self.ssh, self.host, 'bash -s -- ' + shlex.join(list(map(str, args)))], input=Path(path).read_text(),
                          capture_output=True, text=True, timeout=timeout)
        return result.returncode, result.stdout

    def upload(self, source, destination):
        self.run(['rsync', '-a', '-e', shlex.join(self.ssh), str(source), self.host + ':' + shlex.quote(destination)], check=True, timeout=600)

    def receive(self, source, destination, exclude=(), timeout=600, limit=None):
        Path(destination).mkdir(parents=True, exist_ok=True)
        options = [argument for pattern in ('.pending-*', *exclude) for argument in ('--exclude', pattern)]
        if limit:
            options.append('--max-size=' + limit)
        self.run(['rsync', '-a', *options, '-e', shlex.join(self.ssh), self.host + ':' + shlex.quote(source),
                  str(destination) + '/'], check=True, timeout=timeout)


def prefix(configuration, workspace):
    uv = configuration['machine']['uv']
    return ['env', 'PATH=' + workspace + '/bin:' + str(Path(uv).parent) + ':/usr/local/bin:/usr/bin:/bin',
            'UV_TOOL_DIR=' + workspace + '/tools', 'UV_TOOL_BIN_DIR=' + workspace + '/bin', uv]


def module(configuration, workspace):
    return [*prefix(configuration, workspace), 'run', '--project', workspace + '/bialy', '--locked', 'python', '-m', 'bialy.student_host']


def setup_steps(remote, configuration, workspace, student, splits, source_key, destination, schedule):
    """The work between a reachable machine and a running trainer, as named steps. Each can
    be run again after an interruption at any point: it holds the machine's setup lock, so
    it waits out a run of itself that a lost controller left behind, and then finds
    that run's work done."""
    uv, locked = prefix(configuration, workspace), ['flock', workspace + '/.setup.lock']
    job = workspace + '/runs/' + destination.name

    def source(name, repository, commit):
        target = workspace + '/' + name
        remote.command(['git', 'init', '-q', target])
        remote.command([*locked, 'git', '-C', target, 'fetch', '-q', repository, commit], timeout=1800)
        remote.command([*locked, 'git', '-C', target, 'checkout', '-q', '--detach', commit])

    def preflight():
        machines.measure(remote, configuration, student, destination)
        remote.command(['mkdir', '-p', *[workspace + '/' + name for name in ('data', 'private', 'runs', 'spool', 'models')]])
        remote.command([*uv, '--version'])

    def sources():
        source('host', configuration['host_repository'], configuration['host_commit'])
        source('bialy', configuration['bialy_repository'], configuration['bialy_commit'])
        source('llama.cpp', configuration['llama_repository'], configuration['llama_commit'])

    def environments():
        remote.command([*locked, *uv, 'sync', '--project', workspace + '/bialy', '--locked'])
        for project in ('scripts/bialy/custom_model', 'scripts/generation'):
            remote.command([*locked, *uv, 'sync', '--project', workspace + '/host/' + project, '--locked'], timeout=1800)
        remote.command([*locked, *uv, 'tool', 'install', 'huggingface_hub==2.2.0'])

    def llama():
        directory = workspace + '/llama.cpp'
        remote.command([*locked, 'cmake', '-S', directory, '-B', directory + '/build', '-DGGML_CUDA=OFF', '-DLLAMA_CURL=OFF',
                        '-DCMAKE_BUILD_TYPE=Release'], timeout=1800)
        remote.command([*locked, 'cmake', '--build', directory + '/build', '--target', 'llama-quantize', 'llama-server',
                        '--parallel'], timeout=3600)

    def checkpoint():
        # A download that was cut off left a partial copy beside the one this run makes.
        remote.command([*locked, 'find', workspace + '/models', '-maxdepth', '1', '-name', '.download-*', '-exec', 'rm', '-rf', '{}', '+'])
        remote.command([*locked, *module(configuration, workspace), 'models', '--root', workspace + '/models', '--student', student],
                       timeout=3600)

    def data():
        remote.upload(str(Path(splits)) + '/', workspace + '/data/')
        remote.upload(source_key, workspace + '/private/source.key')
        remote.command(['chmod', '600', workspace + '/private/source.key'])

    def create():
        created = json.loads(remote.command([
            *module(configuration, workspace), 'create', '--student', student, '--root', workspace + '/runs', '--name', destination.name,
            '--checkout', workspace + '/host', '--models', workspace + '/models', '--splits', workspace + '/data',
            '--mirror', workspace + '/spool', '--device', configuration['machine'].get('device', 'cuda:0'), '--schedule', schedule]))
        if created['directory'] != job:
            raise JobError('the machine created another job than the one named')
        write(destination / 'remote.json', {'host': remote.host, 'directory': job})

    def launch():
        # A trainer that is running or has ended has written its heartbeat; a second
        # launch that races one is refused by the job's own execution lock.
        if remote.command(['bash', '-c', 'test -e ' + shlex.quote(job + '/heartbeat.json') + ' && echo started || echo absent']).strip() == 'started':
            return
        command = shlex.join(['env', 'PW_TRAINING_SOURCE_KEY=' + workspace + '/private/source.key',
                              *module(configuration, workspace), 'train', job])
        remote.command(['bash', '-lc', 'nohup ' + command + ' >>' + shlex.quote(job + '/launcher.log') + ' 2>&1 </dev/null &'])

    return (('preflight', 'preflight', preflight), ('bootstrap', 'sources', sources), ('bootstrap', 'environments', environments),
            ('bootstrap', 'llama', llama), ('bootstrap', 'checkpoint', checkpoint), ('upload', 'data', data),
            ('train', 'create', create), ('train', 'launch', launch))


def prepare(steps, destination, check_connection, stage):
    """Run the setup steps `setup.json` does not record as done on the machine now leased."""
    lease = destination / 'resource.json'
    machine = json.loads(lease.read_text())['token'] if lease.exists() else None
    record = destination / 'setup.json'
    recorded = json.loads(record.read_text()) if record.exists() else {'machine': machine, 'done': []}
    # Steps recorded for a machine since replaced were done on that machine.
    done = recorded['done'] if recorded['machine'] == machine else []
    for name, step, run in steps:
        if step in done:
            continue
        stage(name)
        check_connection()
        run()
        done = [*done, step]
        write(record, {'machine': machine, 'done': done})


def evidence(remote, remote_directory, destination):
    """Bring home what explains a failed job before its machine is released: the adapter,
    manifests, the behaviour reference and observations, and every log. Never the bulk."""
    try:
        remote.receive(remote_directory + '/', destination / 'job', exclude=BULK, timeout=1800, limit=EVIDENCE_FILE_LIMIT)
    except (JobError, subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError):
        return 'unavailable'
    return 'returned'


def export_remotely(remote, configuration, workspace, remote_directory, destination, check_connection, sleep, stage=lambda name: None):
    """Export on the training machine, which has the memory for a float32 merge; return the three outputs, not the merge."""
    stage('export')
    marker = destination / 'export.json'
    if not marker.exists():
        command = [*module(configuration, workspace), 'export', remote_directory, '--llama', workspace + '/llama.cpp',
                   '--revision', configuration['llama_commit']]
        status = shlex.quote(remote_directory + '/export.exit')
        inner = shlex.join(command) + '; echo "$?" >' + status + '.pending && mv ' + status + '.pending ' + status
        script = ('nohup bash -c ' + shlex.quote(inner) + ' >' + shlex.quote(remote_directory + '/export.log')
                  + ' 2>&1 </dev/null & echo $!')
        pid = int(remote.command(['bash', '-lc', script]).strip())
        write(marker, {'pid': pid})
    pid = json.loads(marker.read_text())['pid']
    probe = ('import json,os,pathlib; p=pathlib.Path(' + repr(remote_directory) + ')/"export.exit"\n'
             'try:\n os.kill(' + str(pid) + ',0); alive=True\n'
             'except OSError:\n alive=False\n'
             'print(json.dumps({"exit":int(p.read_text()) if p.exists() else None,"alive":alive}))')
    failures = 0
    while True:
        check_connection()
        try:
            observation = json.loads(remote.command(['python3', '-c', probe], timeout=30))
        except (JobError, subprocess.TimeoutExpired):
            failures += 1
            if failures > 2:
                raise JobError('remote export observation unavailable') from None
            sleep(30)
            continue
        failures = 0
        if observation['exit'] is not None:
            break
        if not observation['alive']:
            raise JobError('remote export worker lost')
        sleep(30)
    if observation['exit']:
        raise JobError('remote export failed with status %d' % observation['exit'])
    stage('retrieve')
    job = destination / 'job'
    remote.receive(remote_directory + '/merged/manifest.json', job / 'merged')
    remote.receive(remote_directory + '/merged/' + exporting.REFERENCE, job / 'merged')
    remote.receive(remote_directory + '/gguf/', job / 'gguf', exclude=(exporting.UNQUANTIZED,), timeout=7200)
    remote.receive(remote_directory + '/mlx/', job / 'mlx', timeout=7200)
    remote.receive(remote_directory + '/lora_adapter/', job / 'lora_adapter')
    remote.receive(remote_directory + '/export_receipt.json', job)
    remote.receive(remote_directory + '/export.log', job)
    return exporting.publish(job, configuration['llama_commit'], unquantized=False)


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


def observe(remote, remote_directory, destination, supervisor, sleep):
    """Watch the trainer to its end, replicating checkpoints; returns its last execution state."""
    probe = ('import json,time,pathlib; p=pathlib.Path(' + repr(remote_directory) + '); '
             's=json.loads((p/"execution.json").read_text()); '
             'h=json.loads((p/"heartbeat.json").read_text()); '
             'print(json.dumps({"execution":s,"heartbeat_age":time.time()-h["at"]}))')
    transport_failures = 0
    while True:
        supervisor.check()
        try:
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
                lost = supervisor.lost()
                raise JobError('rented machine was lost: ' + lost if lost else 'remote training observation unavailable') from None
            sleep(20)
            continue
        if state['status'] in {'complete', 'failed'}:
            return state
        sleep(20)


def train(configuration_path, student, splits, source_key, out, *, resume=None, schedule='full', sleep=time.sleep, clock=time.time):
    configuration = json.loads(Path(configuration_path).read_text())
    if schedule not in SCHEDULES:
        raise JobError('unknown training schedule ' + str(schedule))
    # Refused here, a machine that cannot run the tier leaves no job directory and rents nothing.
    machines.preflight(configuration, student)
    destination = Path(resume) if resume else Path(out) / (student + "-" + uuid.uuid4().hex)
    if not resume:
        destination.mkdir(parents=True, exist_ok=False)
    with (destination / ".controller.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise JobError("another controller holds the remote training lock") from exc
        try:
            return train_locked(configuration, student, splits, source_key, destination, resume, schedule, sleep, clock)
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def failure(exc):
    code = next((code for kind, code in FAILURES if isinstance(exc, kind)), 'failed')
    return {'type': type(exc).__name__, 'code': code, 'diagnostic': str(exc)}


def train_locked(configuration, student, splits, source_key, destination, resume, schedule, sleep, clock):
    binding = {'configuration': identity(configuration), 'student': student, 'splits': tree(splits), 'schedule': schedule}
    if resume:
        if json.loads((destination / 'binding.json').read_text()) != binding:
            raise JobError('remote training restart inputs changed')
        outcome = json.loads((destination / 'outcome.json').read_text())
        if outcome['status'] != 'running':
            # The work is settled; only a release the provider had not confirmed is still open.
            if outcome.get('cleanup') not in RELEASED:
                outcome['cleanup'] = machines.release(configuration, destination, sleep, clock)
                write(destination / 'outcome.json', outcome)
            if outcome['status'] != 'complete' or outcome['cleanup'] not in RELEASED:
                raise JobError('remote training has a terminal outcome; evidence: ' + str(destination))
            plan = json.loads((destination / 'job/plan.json').read_text())
            validate_inputs(plan)
            validate_result(plan)
            return {'directory': str(destination), **outcome}
    else:
        write(destination / 'binding.json', binding)
        # What `status` needs to ask the provider about this job; it names key files, not keys.
        descriptor = os.open(destination / 'configuration.json', os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, 'w') as stream:
            json.dump(configuration, stream)
        outcome = {'schema': 'pw-remote-training/1', 'student': student, 'schedule': schedule, 'status': 'running'}
        write(destination / 'outcome.json', outcome)
    stages = Stages(destination, clock)
    try:
        # Once the job exists on a machine, the run stays on that machine.
        with machines.connect(configuration, destination, Remote, sleep, replaceable=not (destination / 'remote.json').exists(),
                              stage=stages.enter, clock=clock) as (remote, supervisor):
            workspace = configuration['machine']['workspace'].rstrip('/') + '/' + destination.name
            try:
                prepare(setup_steps(remote, configuration, workspace, student, splits, source_key, destination, schedule),
                        destination, supervisor.check, stages.enter)
                remote_directory = json.loads((destination / 'remote.json').read_text())['directory']
                stages.enter('train')
                state = observe(remote, remote_directory, destination, supervisor, sleep)
                stages.enter('retrieve')
                remote.receive(remote_directory + '/', destination / 'job', exclude=BULK)
                # Recorded before the export, whose failure must not take the training record with it.
                outcome.update(execution=state, resources=state.get('resources'))
                write(destination / 'outcome.json', outcome)
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
                    try:
                        outcome['artifact'] = export_remotely(remote, configuration, workspace, remote_directory, destination,
                                                              supervisor.check, sleep, stages.enter)
                    except Exception:
                        stages.enter('retrieve')
                        outcome['evidence'] = evidence(remote, remote_directory, destination)
                        raise
                outcome['status'] = state['status']
            finally:
                stages.enter('release')
    except Exception as exc:
        outcome.update(status='failed', failure=failure(exc))
    finally:
        stages.close()
        cleanup = destination / 'cleanup.json'
        outcome['cleanup'] = json.loads(cleanup.read_text())['status'] if cleanup.exists() else 'unconfirmed'
        outcome.update(stages=seconds(stages.entries(), clock()), rental=rental(machines.Rental(destination).attempts(), clock()))
        write(destination / 'outcome.json', outcome)
    if outcome['status'] != 'complete' or outcome['cleanup'] not in RELEASED:
        raise JobError('remote training did not complete; evidence: ' + str(destination))
    return {'directory': str(destination), **outcome}
