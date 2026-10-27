"""Remote training on configured machines with continuous checkpoint replication."""
import fcntl
import json
import shlex
import subprocess
import threading
import time
import uuid
from pathlib import Path

from . import exporting, machines
from .jobs import JobError, tree, validate_inputs, validate_result, validate_training_inputs
from .resources import write
from .specification import identity
from .transfer import checkpoints

SSH = ['ssh', '-o', 'BatchMode=yes', '-o', 'StrictHostKeyChecking=accept-new',
       '-o', 'ConnectTimeout=10', '-o', 'ServerAliveInterval=20', '-o', 'ServerAliveCountMax=3']


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
        if result.returncode:
            raise JobError('remote operation failed with status %d' % result.returncode)
        return result.stdout

    def upload(self, source, destination):
        self.run(['rsync', '-a', '-e', shlex.join(self.ssh), str(source), self.host + ':' + shlex.quote(destination)], check=True, timeout=600)

    def receive(self, source, destination, exclude=(), timeout=600):
        Path(destination).mkdir(parents=True, exist_ok=True)
        excludes = [argument for pattern in ('.pending-*', *exclude) for argument in ('--exclude', pattern)]
        self.run(['rsync', '-a', *excludes, '-e', shlex.join(self.ssh), self.host + ':' + shlex.quote(source),
                  str(destination) + '/'], check=True, timeout=timeout)


def prefix(configuration, workspace):
    uv = configuration['machine']['uv']
    return ['env', 'PATH=' + workspace + '/bin:' + str(Path(uv).parent) + ':/usr/local/bin:/usr/bin:/bin',
            'UV_TOOL_DIR=' + workspace + '/tools', 'UV_TOOL_BIN_DIR=' + workspace + '/bin', uv]


def bootstrap(remote, configuration, workspace, student):
    remote.command(['mkdir', '-p', workspace])
    uv = prefix(configuration, workspace)
    remote.command([*uv, '--version'])
    for name in ('host', 'bialy'):
        destination = workspace + '/' + name
        remote.command(['git', 'clone', '--no-checkout', configuration[name + '_repository'], destination])
        remote.command(['git', '-C', destination, 'fetch', 'origin', configuration[name + '_commit']])
        remote.command(['git', '-C', destination, 'checkout', '--detach', configuration[name + '_commit']])
    remote.command([*uv, 'sync', '--project', workspace + '/bialy', '--locked'])
    remote.command([*uv, 'sync', '--project', workspace + '/host/scripts/bialy/custom_model', '--locked'])
    llama = workspace + '/llama.cpp'
    remote.command(['git', 'clone', '--no-checkout', configuration['llama_repository'], llama])
    remote.command(['git', '-C', llama, 'fetch', 'origin', configuration['llama_commit']])
    remote.command(['git', '-C', llama, 'checkout', '--detach', configuration['llama_commit']])
    remote.command(['cmake', '-S', llama, '-B', llama + '/build', '-DGGML_CUDA=OFF', '-DLLAMA_CURL=OFF',
                    '-DCMAKE_BUILD_TYPE=Release'], timeout=1800)
    remote.command(['cmake', '--build', llama + '/build', '--target', 'llama-quantize', 'llama-completion',
                    '--parallel'], timeout=3600)
    remote.command([*uv, 'tool', 'install', 'huggingface_hub==2.2.0'])
    remote.command([*uv, 'run', '--project', workspace + '/bialy', '--locked',
                    'python', '-m', 'bialy.student_host', 'models', '--root', workspace + '/models',
                    '--student', student], timeout=3600)


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
            raise JobError('instance lease renewal failed') from self.error

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.stop.set()
        self.thread.join()


def export_remotely(remote, configuration, workspace, remote_directory, destination, check_connection, sleep):
    """Export on the training machine, which has the memory for a float32 merge, and return only the served GGUF."""
    marker = destination / 'export.json'
    if not marker.exists():
        command = [*prefix(configuration, workspace), 'run', '--project', workspace + '/bialy', '--locked', 'python', '-m',
                   'bialy.student_host', 'export', remote_directory, '--llama', workspace + '/llama.cpp',
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
        remote.receive(remote_directory + '/export.log', destination / 'job')
        raise JobError('remote export failed with status %d' % observation['exit'])
    job = destination / 'job'
    remote.receive(remote_directory + '/merged/manifest.json', job / 'merged')
    remote.receive(remote_directory + '/gguf/', job / 'gguf', exclude=(exporting.UNQUANTIZED,), timeout=7200)
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


def train(configuration_path, student, splits, source_key, out, *, resume=None, pilot=False, sleep=time.sleep):
    destination = Path(resume) if resume else Path(out) / (student + "-" + uuid.uuid4().hex)
    if not resume:
        destination.mkdir(parents=True, exist_ok=False)
    with (destination / ".controller.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise JobError("another controller holds the remote training lock") from exc
        try:
            return train_locked(configuration_path, student, splits, source_key, destination, resume, pilot, sleep)
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def train_locked(configuration_path, student, splits, source_key, destination, resume, pilot, sleep):
    configuration = json.loads(Path(configuration_path).read_text())
    machines.validate(configuration)
    binding = {'configuration': identity(configuration), 'student': student, 'splits': tree(splits), 'pilot': pilot}
    if resume:
        if json.loads((destination / 'binding.json').read_text()) != binding:
            raise JobError('remote training restart inputs changed')
        outcome = json.loads((destination / 'outcome.json').read_text())
        if outcome['status'] == 'complete' and outcome.get('cleanup') in {'confirmed', 'not_applicable'}:
            plan = json.loads((destination / 'job/plan.json').read_text())
            validate_inputs(plan)
            validate_result(plan)
            return {'directory': str(destination), **outcome}
        if outcome['status'] != 'running':
            raise JobError('remote training has a terminal outcome')
    else:
        write(destination / 'binding.json', binding)
        outcome = {'schema': 'pw-remote-training/1', 'student': student, 'status': 'running'}
        write(destination / 'outcome.json', outcome)
    try:
        with machines.connect(configuration, destination, resume, Remote, Renewal) as (remote, check_connection):
            ready = False
            for _ in range(60):
                check_connection()
                try:
                    remote.command(['true'], timeout=30)
                    ready = True
                    break
                except (JobError, subprocess.TimeoutExpired):
                    sleep(10)
            if not ready:
                raise JobError('training machine did not become reachable')
            workspace = configuration['machine']['workspace'].rstrip('/') + '/' + destination.name
            if resume:
                remote_directory = json.loads((destination / 'remote.json').read_text())['directory']
            else:
                bootstrap(remote, configuration, workspace, student)
                check_connection()
                remote.command(['mkdir', '-p', *[workspace + '/' + name for name in ('data', 'private', 'runs', 'spool')]])
                remote.upload(str(Path(splits)) + '/', workspace + '/data/')
                remote.upload(source_key, workspace + '/private/source.key')
                remote.command(['chmod', '600', workspace + '/private/source.key'])
                command_prefix = [*prefix(configuration, workspace), 'run', '--project', workspace + '/bialy',
                                  '--locked', 'python', '-m', 'bialy.student_host']
                create = [*command_prefix, 'create', '--student', student, '--root', workspace + '/runs',
                          '--checkout', workspace + '/host', '--models', workspace + '/models',
                          '--splits', workspace + '/data', '--mirror', workspace + '/spool',
                          '--device', configuration['machine'].get('device', 'cuda:0')]
                if pilot:
                    create.append('--pilot')
                created = json.loads(remote.command(create))
                remote_directory = created['directory']
                write(destination / 'remote.json', {'host': remote.host, 'directory': remote_directory})
                command = shlex.join(['env', 'PW_TRAINING_SOURCE_KEY=' + workspace + '/private/source.key',
                                      *command_prefix, 'train', remote_directory])
                script = 'nohup ' + command + ' >' + shlex.quote(remote_directory + '/launcher.log') + ' 2>&1 </dev/null &'
                remote.command(['bash', '-lc', script])
            transport_failures = 0
            while True:
                check_connection()
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
                    # The merge and unquantized GGUF stay on the machine and are deleted with it.
                    remote.receive(remote_directory + '/', destination / 'job', exclude=('/merged/', '/gguf/'))
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
                        outcome['artifact'] = export_remotely(remote, configuration, workspace, remote_directory,
                                                              destination, check_connection, sleep)
                    outcome.update(status=state['status'], execution=state)
                    break
                sleep(20)
    except Exception as exc:
        outcome.update(status='failed', failure={'type': type(exc).__name__, 'diagnostic': str(exc)})
    finally:
        cleanup = destination / 'cleanup.json'
        outcome['cleanup'] = json.loads(cleanup.read_text())['status'] if cleanup.exists() else 'unconfirmed'
        write(destination / 'outcome.json', outcome)
    if outcome['status'] != 'complete' or outcome['cleanup'] not in {'confirmed', 'not_applicable'}:
        raise JobError('remote training did not complete; evidence: ' + str(destination))
    return {'directory': str(destination), **outcome}
