"""Receive checkpoint generations without replacing an intact replica."""
import json
import os
import shutil
import uuid
from pathlib import Path

from ..provenance import file_sha256
from .jobs import JobError
from .resources import write


def checkpoints(incoming, durable, expected_identity=None):
    incoming, durable = Path(incoming), Path(durable)
    durable.mkdir(parents=True, exist_ok=True)
    accepted = []
    for source in sorted(incoming.glob('[0-9]*')):
        manifest_path = source / 'manifest.json'
        if source.is_symlink() or manifest_path.is_symlink():
            raise JobError('checkpoint transfer contains a symbolic link')
        if not manifest_path.is_file():
            continue
        manifest = json.loads(manifest_path.read_text())
        identity = manifest.get('identity')
        if expected_identity is not None and identity != expected_identity:
            raise JobError('foreign checkpoint in transfer')
        state = source / 'state.pt'
        if not state.is_file() or state.is_symlink() or source.is_symlink() or file_sha256(state) != manifest.get('state_sha256'):
            continue
        target = durable / source.name
        if target.exists():
            if json.loads((target / 'manifest.json').read_text()) != manifest or file_sha256(target / 'state.pt') != manifest['state_sha256']:
                raise JobError('checkpoint replica collision')
            accepted.append(source.name)
            continue
        staging = durable / ('.pending-' + uuid.uuid4().hex)
        staging.mkdir()
        shutil.copyfile(state, staging / 'state.pt')
        with (staging / 'state.pt').open('rb') as stream:
            os.fsync(stream.fileno())
        if file_sha256(staging / 'state.pt') != manifest['state_sha256']:
            raise JobError('checkpoint transfer changed during publication')
        write(staging / 'manifest.json', manifest)
        os.rename(staging, target)
        descriptor = os.open(durable, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        accepted.append(source.name)
    return accepted
