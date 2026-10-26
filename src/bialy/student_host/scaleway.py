"""Scaleway CLI operations restricted to the recorded project and owned IDs."""
import json
import subprocess

from .resources import ResourceError


class Scaleway:
    def __init__(self, profile, project, zone, run=subprocess.run):
        self.profile, self.project, self.zone, self.run = profile, project, zone, run

    def call(self, *arguments):
        result = self.run(['scw', '-p', self.profile, '-o', 'json', *arguments, 'zone=' + self.zone],
                          capture_output=True, text=True, timeout=120)
        if result.returncode:
            raise ResourceError('Scaleway operation did not confirm success')
        return json.loads(result.stdout) if result.stdout.strip() else None

    def listing(self, service, kind, *filters):
        result = self.call(service, kind, 'list', 'project-id=' + self.project, *filters)
        if not isinstance(result, list):
            raise ResourceError('Scaleway list returned an unexpected shape')
        return result

    def find(self, token):
        servers = self.listing('instance', 'server', 'tags=' + token)
        if any(token not in server.get('tags', []) or server.get('project') != self.project for server in servers):
            raise ResourceError('Scaleway resource ownership mismatch')
        return servers

    def create(self, token, specification):
        allowed = {'type', 'image', 'root_volume'}
        if set(specification) != allowed:
            raise ResourceError('invalid rental specification')
        server = self.call('instance', 'server', 'create', 'project-id=' + self.project,
                           'name=pw-training-' + token, 'tags.0=' + token, 'ip=new',
                           'type=' + specification['type'], 'image=' + specification['image'],
                           'root-volume=' + specification['root_volume'])
        if not isinstance(server, dict) or not server.get('id') or server.get('project') != self.project:
            raise ResourceError('Scaleway creation returned an invalid identity')
        return server

    def remaining(self, server):
        volumes = {value['id'] for value in (server.get('volumes') or {}).values()}
        ips = {value['id'] for value in server.get('public_ips', [])}
        if server.get('public_ip'):
            ips.add(server['public_ip']['id'])
        resources = []
        for service, kind, expected in (('instance', 'server', {server['id']}), ('instance', 'volume', volumes),
                                         ('block', 'volume', volumes), ('instance', 'ip', ips)):
            if expected:
                resources.extend((service, kind, item['id']) for item in self.listing(service, kind) if item['id'] in expected)
        return resources

    def absent(self, server):
        return not self.remaining(server)

    def delete(self, server):
        resources = self.remaining(server)
        if ('instance', 'server', server['id']) in resources:
            self.call('instance', 'server', 'delete', server['id'], 'with-volumes=all', 'with-ip=true', 'force-shutdown=true')
        for service, kind, identifier in self.remaining(server):
            if kind == 'server':
                raise ResourceError('Scaleway server deletion is not settled')
            self.call(service, kind, 'delete', identifier)
