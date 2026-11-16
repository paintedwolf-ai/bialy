"""Host evidence through the pinned image's training contract and episode bundles.

The host states what it exports (`pw-training-contract/1`) and exports each closed
episode as a bundle (`pw-training-episode-bundle/1`) whose files and blobs are bound
by SHA-256. Nothing here reads host storage or names a host tool.
"""
import hashlib
import json
import os
import subprocess
from pathlib import Path

CONTRACT = 'pw-training-contract/1'
BUNDLE = 'pw-training-episode-bundle/1'
ENTRYPOINT = '/opt/bialy/bin/lycaon-debug'
LINES = ('sessions', 'captures', 'operations', 'observations', 'effects', 'decisions', 'completions')


class HostEvidenceError(Exception):
    def __init__(self, code, details=None):
        super().__init__(code)
        self.code, self.details = code, details or {}


def _sha256(raw):
    return hashlib.sha256(raw).hexdigest()


class Docker:
    """The pinned runner image, run offline and read-only with the given mounts."""

    def __init__(self, image):
        self.image = image

    def command(self, mounts, arguments):
        command = ['docker', 'run', '--rm', '--network', 'none', '--read-only', '--cap-drop', 'ALL',
                   '--user', '%d:%d' % (os.getuid(), os.getgid())]
        for source, target, mode in mounts:
            command += ['-v', '%s:%s:%s' % (Path(source).resolve(), target, mode)]
        return [*command, '--entrypoint', ENTRYPOINT, self.image, 'eval', *arguments]


class Local:
    """The host's lycaon-debug run directly, each mount target read as its source path."""

    def __init__(self, binary):
        self.binary = binary

    def command(self, mounts, arguments):
        mapped = []
        for argument in arguments:
            for source, target, _ in mounts:
                if argument == target or argument.startswith(target + '/'):
                    argument = str(Path(source).resolve()) + argument[len(target):]
                    break
            mapped.append(argument)
        return [str(self.binary), 'eval', *mapped]


def _evaluate(image, mounts, arguments, process):
    runner = image if hasattr(image, 'command') else Docker(image)
    return process(runner.command(mounts, arguments), check=False, capture_output=True, text=True)


def _failed(destination, result, code):
    diagnostic = destination.with_name(destination.name + '-failure.json')
    diagnostic.write_text(json.dumps({'exit_status': result.returncode, 'stderr': result.stderr}))
    diagnostic.chmod(0o600)
    raise HostEvidenceError(code, {'exit_status': result.returncode, 'diagnostic': str(diagnostic)})


def contract(image, destination, commit, process=subprocess.run):
    """The image's training contract, exported once; it must be from commit and unmodified."""
    destination = Path(destination)
    path = destination / 'contract.json'
    if not path.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        result = _evaluate(image, [(destination.parent, '/out', 'rw')], ['training-contract', '--out', '/out/' + destination.name], process)
        if result.returncode:
            _failed(destination, result, 'host_contract_export_failed')
    raw = path.read_bytes()
    value = json.loads(raw)
    if value.get('schema') != CONTRACT or value.get('bundle') != BUNDLE:
        raise HostEvidenceError('host_contract_unsupported', {'schema': value.get('schema'), 'bundle': value.get('bundle')})
    if value.get('host_commit') != commit or value.get('host_modified') is not False:
        raise HostEvidenceError('host_contract_revision_mismatch', {'expected': commit, 'actual': value.get('host_commit')})
    return value, _sha256(raw)


def export(image, shard, session, destination, project=None, process=subprocess.run):
    """Export one closed episode once and return it."""
    destination = Path(destination)
    if not (destination / 'manifest.json').exists():
        # Docker would create a missing mount source as root.
        destination.parent.mkdir(parents=True, exist_ok=True)
        mounts = [(shard, '/capture', 'ro'), (destination.parent, '/out', 'rw')]
        arguments = ['training-bundle', '--capture', '/capture', '--captures', '/capture/training-captures',
                     '--session', session, '--out', '/out/' + destination.name]
        if project is not None:
            mounts.append((project, '/project', 'ro'))
            arguments += ['--project', '/project']
        result = _evaluate(image, mounts, arguments, process)
        if result.returncode:
            _failed(destination, result, 'host_evidence_export_failed')
    episode = Episode(destination)
    if episode.root_session != session:
        raise HostEvidenceError('episode_host_identity_mismatch')
    return episode


class Episode:
    """One verified bundle: its manifest binds every file and blob read here."""

    def __init__(self, directory):
        self.directory = Path(directory)
        try:
            raw = (self.directory / 'manifest.json').read_bytes()
            self.manifest = json.loads(raw)
        except (OSError, ValueError) as exc:
            raise HostEvidenceError('host_bundle_missing', {'directory': str(self.directory)}) from exc
        self.manifest_sha256 = _sha256(raw)
        if not isinstance(self.manifest, dict) or self.manifest.get('schema') != BUNDLE:
            raise HostEvidenceError('host_bundle_unsupported')
        for name, expected in self.manifest['files'].items():
            path = self.directory / name
            if not path.is_file() or _sha256(path.read_bytes()) != expected:
                raise HostEvidenceError('host_bundle_changed', {'file': name})
        self.contract_sha256 = self.manifest['files']['catalog.json']
        self.root_session = self.manifest['root_session_id']
        for name in LINES:
            text = (self.directory / (name + '.jsonl')).read_text()
            setattr(self, name, [json.loads(line) for line in text.splitlines() if line.strip()])

    def blob(self, name):
        if name not in self.manifest['blobs']:
            raise HostEvidenceError('host_bundle_blob_unlisted')
        raw = (self.directory / 'blobs' / name).read_bytes()
        if _sha256(raw) != name:
            raise HostEvidenceError('host_bundle_changed', {'blob': name})
        return json.loads(raw)

    def messages(self, session):
        return self.blob(session['messages_blob'])

    def capture_request(self, record):
        name = record.get('request_blob')
        return None if name is None else self.blob(name)

    @property
    def execution(self):
        return next((item['value'] for item in self.observations if item['kind'] == 'execution'), None)

    def calls(self):
        """Top-level operations, each with the children it ran."""
        children = {}
        for operation in self.operations:
            if operation.get('parent_id'):
                children.setdefault(operation['parent_id'], []).append(operation)
        return [(operation, children.get(operation['id'], [])) for operation in self.operations if not operation.get('parent_id')]
