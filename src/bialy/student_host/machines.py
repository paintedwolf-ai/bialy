"""Connect to a configured machine, with optional instance provisioning."""
import contextlib
import json
import re
import subprocess
from pathlib import Path, PurePosixPath

from .jobs import JobError
from .reaper import provider
from .resources import Resources, write


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
    machine = configuration['machine']
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
    provision = configuration.get('provisioning')
    if provision is None:
        if not re.fullmatch('[a-zA-Z0-9][a-zA-Z0-9.:-]*', machine.get('host', '')):
            raise JobError('machine host is missing or invalid')
    else:
        if 'host' in machine or set(provision) != {'provider', 'profile', 'project', 'zone', 'registry', 'specification'}:
            raise JobError('provisioning configuration is invalid')
        if provision['provider'] != 'scaleway' or not Path(provision['registry']).is_absolute():
            raise JobError('unsupported instance provider or invalid registry')
        if any(not isinstance(provision[key], str) or not provision[key] for key in ('profile', 'project', 'zone')):
            raise JobError('instance provider configuration is incomplete')
        spec = provision['specification']
        if not isinstance(spec, dict) or set(spec) != {'type', 'image', 'root_volume'} or any(not isinstance(value, str) or not value for value in spec.values()):
            raise JobError('instance specification is incomplete')


@contextlib.contextmanager
def connect(configuration, directory, resume, remote_type, renewal_type):
    provision = configuration.get('provisioning')
    machine = configuration['machine']
    if provision is None:
        try:
            yield remote_type(machine['host'], machine), lambda: None
        finally:
            write(directory / 'cleanup.json', {'status': 'not_applicable'})
        return
    resources = Resources(provision['registry'], provider(provision))
    path = directory / 'resource.json'
    if resume:
        record = json.loads(path.read_text())
    else:
        record = resources.intent(provision['specification'], 300)
        write(path, record)
    try:
        with renewal_type(resources, record['token'], 300) as renewal:
            try:
                allocation = resources.recover(record['token'], 300) if resume else resources.allocate(record)
            except (ConnectionError, subprocess.TimeoutExpired):
                allocation = resources.recover(record['token'], 300) if resume else resources.allocate(record)
            write(path, allocation)
            server = allocation['server']
            address = (server.get('public_ip') or server['public_ips'][0])['address']
            yield remote_type(address, machine), renewal.check
    finally:
        try:
            resources.release(record['token'])
        except Exception:
            write(directory / 'cleanup.json', {'status': 'unconfirmed', 'token': record['token']})
            raise
        else:
            write(directory / 'cleanup.json', {'status': 'confirmed', 'token': record['token']})
