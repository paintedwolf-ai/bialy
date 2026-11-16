"""Scaleway CLI operations on the servers of one project that carry a lease token."""
import json
import subprocess

from .resources import ResourceError

SPECIFICATION = {'type', 'image', 'root_volume', 'gpu_memory_gib', 'memory_gib', 'disk_gib'}


def name(token):
    return 'pw-training-' + token


def normalize(server):
    """A server in any state, with the volumes and addresses that outlive it unless deleted."""
    addresses = {item['id']: item for item in [*(server.get('public_ips') or []), server.get('public_ip')] if item}
    address = next((item['address'] for item in addresses.values() if item.get('address')), None)
    children = [['volume', item['id']] for item in (server.get('volumes') or {}).values()] + [['ip', key] for key in addresses]
    return {'id': server['id'], 'state': str(server.get('state') or 'pending'), 'interruptible': False, 'children': sorted(children),
            'endpoint': {'address': address, 'port': 22} if server.get('state') == 'running' and address else None}


class Scaleway:
    def __init__(self, profile, project, zone, run=subprocess.run):
        self.profile, self.project, self.zone, self.run = profile, project, zone, run

    def call(self, *arguments, timeout=120):
        result = self.run(['scw', '-p', self.profile, '-o', 'json', *arguments, 'zone=' + self.zone],
                          capture_output=True, text=True, timeout=timeout)
        if result.returncode:
            raise ResourceError('Scaleway operation did not confirm success')
        try:
            return json.loads(result.stdout) if result.stdout.strip() else None
        except ValueError as exc:
            raise ResourceError('Scaleway returned an unreadable response') from exc

    def listing(self, service, kind, *filters):
        result = self.call(service, kind, 'list', 'project-id=' + self.project, *filters)
        if not isinstance(result, list) or any(not isinstance(item, dict) or not item.get('id') for item in result):
            raise ResourceError('Scaleway list returned an unexpected shape')
        return result

    def owned(self, token):
        return [normalize(server) for server in self.listing('instance', 'server', 'tags=' + token)
                if token in (server.get('tags') or []) and server.get('project') == self.project and server.get('name') == name(token)]

    def create(self, token, specification, startup):
        server = self.call('instance', 'server', 'create', 'project-id=' + self.project, 'name=' + name(token),
                           'tags.0=' + token, 'ip=new', 'type=' + specification['type'], 'image=' + specification['image'],
                           'root-volume=' + specification['root_volume'], 'cloud-init=' + startup)
        if not isinstance(server, dict) or not server.get('id') or server.get('project') != self.project:
            raise ResourceError('Scaleway creation returned an invalid identity')
        return normalize(server)

    def remaining(self, instance):
        children = {kind: {identifier for child, identifier in instance['children'] if child == kind} for kind in ('volume', 'ip')}
        resources = []
        for service, kind, expected in (('instance', 'server', {instance['id']}), ('instance', 'volume', children['volume']),
                                        ('block', 'volume', children['volume']), ('instance', 'ip', children['ip'])):
            if expected:
                resources.extend([service, kind, item['id']] for item in self.listing(service, kind) if item['id'] in expected)
        return resources

    def destroy(self, instance):
        """True when the provider acknowledged deleting the server. Volumes and addresses it
        left behind are deleted one by one."""
        acknowledged = False
        try:
            # The CLI stops the server before deleting it, which can take minutes.
            self.call('instance', 'server', 'delete', instance['id'], 'with-volumes=all', 'with-ip=true', 'force-shutdown=true',
                      timeout=900)
            acknowledged = True
        except (ResourceError, subprocess.TimeoutExpired):
            if ['instance', 'server', instance['id']] in self.remaining(instance):
                raise
        for service, kind, identifier in self.remaining(instance):
            if kind == 'server':
                raise ResourceError('Scaleway server deletion is not settled')
            self.call(service, kind, 'delete', identifier)
        return acknowledged
