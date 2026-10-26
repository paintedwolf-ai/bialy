"""Run frozen task checks against a retained source snapshot in an isolated container."""
import os
import shutil
import subprocess
import uuid
from pathlib import Path, PurePosixPath

from ..provenance import file_sha256
from .containers import Container
from .execution import atomic_json
from .journal import CampaignFault, digest


def seed(task, bank, project):
    bank, project = Path(bank), Path(project)
    for name, source_name in task.get('fixture_files', {}).items():
        relative, source_relative = PurePosixPath(name), PurePosixPath(source_name)
        if any(path.is_absolute() or '..' in path.parts or not path.parts or path.parts[0] == '.git'
               for path in (relative, source_relative)):
            raise CampaignFault('fixture_path_outside_project')
        source = bank / source_relative
        if (not source.resolve().is_relative_to(bank.resolve()) or source.is_symlink()
                or file_sha256(source) != task['oracle_files'].get(source_name)):
            raise CampaignFault('fixture_source_changed')
        target = project / relative
        if target.exists():
            raise CampaignFault('fixture_destination_exists')
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('xb') as stream:
            stream.write(source.read_bytes())


def snapshot(project, destination, artifact_paths=(), run=subprocess.run):
    project, destination = Path(project), Path(destination)
    result = run(['git', '-C', str(project), 'ls-files', '--cached', '--others', '--exclude-standard', '-z'],
                 check=True, capture_output=True)
    names = {os.fsdecode(name) for name in result.stdout.split(b'\0') if name}
    names.update(artifact_paths)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.with_name('.' + destination.name + '-' + uuid.uuid4().hex)
    staging.mkdir()
    inventory = {}
    for name in sorted(names):
        relative = PurePosixPath(name)
        if relative.is_absolute() or '..' in relative.parts or not relative.parts or relative.parts[0] == '.git':
            raise CampaignFault('artifact_path_outside_project')
        source = project / relative
        if source.is_symlink() or any(parent.is_symlink() for parent in source.parents if parent != project and project in parent.parents):
            raise CampaignFault('artifact_symbolic_link')
        if not source.exists():
            continue
        if not source.is_file():
            raise CampaignFault('artifact_not_regular_file')
        target = staging / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        inventory[name] = {'sha256': file_sha256(target), 'executable': bool(target.stat().st_mode & 0o111)}
    if destination.exists():
        verify_snapshot(destination, inventory)
        shutil.rmtree(staging)
    else:
        os.rename(staging, destination)
    atomic_json(destination.parent / 'source-inventory.json', inventory)
    return inventory


def verify_snapshot(directory, inventory):
    paths = [path for path in Path(directory).rglob('*') if path.is_file() or path.is_symlink()]
    if any(path.is_symlink() for path in paths):
        raise CampaignFault('artifact_snapshot_changed')
    current = {str(path.relative_to(directory)): {'sha256': file_sha256(path), 'executable': bool(path.stat().st_mode & 0o111)} for path in paths}
    if current != inventory:
        raise CampaignFault('artifact_snapshot_changed')


def grade(task, source, inventory, oracle, evidence, image, destination, run=subprocess.run):
    source, oracle, evidence, destination = map(Path, (source, oracle, evidence, destination))
    verify_snapshot(source, inventory)
    if not image.startswith('sha256:'):
        raise CampaignFault('oracle_image_not_pinned')
    files = {str(path.relative_to(oracle)): file_sha256(path) for path in oracle.rglob('*') if path.is_file()}
    if files != task['oracle_files'] or any(path.is_symlink() for path in oracle.rglob('*')):
        raise CampaignFault('task_oracle_changed')
    identities = [check['id'] for check in task['checks']]
    if (not identities or len(identities) != len(set(identities)) or 'host-closeout' in identities
            or set(task['required_checks']) != {*identities, 'host-closeout'}):
        raise CampaignFault('task_checks_invalid')
    grader = digest({'image': image, 'oracle_files': files, 'checks': task['checks']})
    destination.mkdir(parents=True, exist_ok=True)
    results = []
    for index, check in enumerate(task['checks']):
        argv = check['argv']
        if not isinstance(argv, list) or not argv or any(not isinstance(value, str) or not value for value in argv):
            raise CampaignFault('task_check_argv_invalid')
        binding = {'task': digest(task), 'source': digest(inventory), 'grader': grader, 'check': check}
        name = 'bialy-oracle-' + digest({'directory': str(destination.resolve()), **binding})[:32]
        container = Container(destination / ('check-%d' % index), run=run)
        terminal = container.execute(name, [
            'docker', 'run', '--name', name, '--network', 'none', '--read-only', '--cap-drop', 'ALL',
            '--security-opt', 'no-new-privileges', '--tmpfs', '/work:rw,exec', '--tmpfs', '/tmp:rw,exec',
            '-e', 'PYTHONDONTWRITEBYTECODE=1', '-v', str(source.resolve()) + ':/source:ro',
            '-v', str(oracle.resolve()) + ':/oracle:ro', '-v', str(evidence.resolve()) + ':/evidence:ro',
            '--entrypoint', '/bin/bash', image, '-c', 'cp -a /source/. /work/ && cd /work && exec "$@"',
            'oracle', *argv], binding)
        log_path = destination / ('check-%d.log' % index)
        if not log_path.exists():
            with log_path.open('xb') as log:
                run(['docker', 'logs', name], stdout=log, stderr=subprocess.STDOUT, check=True)
        results.append({'id': check['id'], 'passed': terminal['exit_status'] == 0 and not terminal['oom_killed'],
                        'exit_status': terminal['exit_status'], 'source_sha256': digest(inventory),
                        'log_sha256': file_sha256(log_path)})
        container.release()
    verify_snapshot(source, inventory)
    receipt = {'source_sha256': digest(inventory), 'task_spec_sha256': digest(task), 'grader_identity': grader, 'checks': results}
    atomic_json(destination / 'oracle.json', receipt)
    return receipt
