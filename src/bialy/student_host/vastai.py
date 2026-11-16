"""Vast.ai CLI operations on the on-demand instances labelled with a lease token."""
import json
import os
import subprocess
from pathlib import Path

from .resources import ResourceError

REQUIRED_SPECIFICATION = {'gpu_name', 'image', 'max_price', 'gpu_memory_gib', 'memory_gib', 'disk_gib'}
ALLOWED_SPECIFICATION = REQUIRED_SPECIFICATION | {'num_gpus'}
# The provider stores the startup command in a bounded field.
STARTUP_LIMIT = 4000


def label(token):
    return 'pw-training-' + token


def normalize(item):
    """An instance in any state. It has an endpoint only once it is running and reachable."""
    host, port = item.get('ssh_host') or item.get('public_ipaddr'), item.get('ssh_port')
    ready = item.get('actual_status') == 'running' and bool(host) and type(port) is int and 0 < port < 65536
    return {'id': str(item['id']), 'state': str(item.get('actual_status') or 'pending'),
            'interruptible': bool(item.get('is_bid')), 'children': [],
            'endpoint': {'address': host, 'port': port} if ready else None}


class VastAI:
    def __init__(self, api_key_file=None, run=subprocess.run):
        self.api_key_file, self.run = api_key_file, run

    def call(self, *arguments):
        environment = None
        if self.api_key_file:
            environment = dict(os.environ, VAST_API_KEY=Path(self.api_key_file).read_text().strip())
        result = self.run(['vastai', '--raw', *arguments], capture_output=True, text=True, timeout=120, env=environment)
        if result.returncode:
            raise ResourceError('Vast.ai operation did not confirm success')
        try:
            return json.loads(result.stdout) if result.stdout.strip() else None
        except ValueError as exc:
            raise ResourceError('Vast.ai returned an unreadable response') from exc

    def instances(self):
        result = self.call('show', 'instances')
        if not isinstance(result, list) or any(not isinstance(item, dict) or not item.get('id') for item in result):
            raise ResourceError('Vast.ai instance list returned an unexpected shape')
        return result

    def owned(self, token):
        return [normalize(item) for item in self.instances() if item.get('label') == label(token)]

    def create(self, token, specification, startup):
        if len(startup.encode()) > STARTUP_LIMIT:
            raise ResourceError('startup command exceeds the provider limit')
        query = ['gpu_name=' + specification['gpu_name'], 'num_gpus=%d' % specification.get('num_gpus', 1),
                 'gpu_ram>=%d' % specification['gpu_memory_gib'], 'cpu_ram>=%d' % specification['memory_gib'],
                 'disk_space>=%d' % specification['disk_gib'], 'dph_total<=%s' % specification['max_price'],
                 'rentable=true', 'verified=true']
        offers = self.call('search', 'offers', '--type', 'on-demand', '--order', 'dph_total', ' '.join(query))
        if not isinstance(offers, list) or not offers or not isinstance(offers[0], dict) or not offers[0].get('id'):
            raise ResourceError('Vast.ai found no matching on-demand offer')
        # No bid price is passed: a bid makes the instance interruptible.
        created = self.call('create', 'instance', str(offers[0]['id']), '--image', specification['image'],
                            '--disk', str(specification['disk_gib']), '--ssh', '--direct', '--label', label(token),
                            '--onstart-cmd', startup)
        if not isinstance(created, dict) or not created.get('new_contract'):
            raise ResourceError('Vast.ai creation returned an invalid contract identity')
        return {'id': str(created['new_contract']), 'state': 'pending', 'interruptible': False, 'children': [], 'endpoint': None}

    def remaining(self, instance):
        return [str(item['id']) for item in self.instances() if str(item['id']) == instance['id']]

    def destroy(self, instance):
        """True when the provider acknowledged the destruction. The request is made by identity
        whatever the listing says, because a new instance can be missing from it."""
        try:
            self.call('destroy', 'instance', instance['id'])
        except (ResourceError, subprocess.TimeoutExpired):
            if self.remaining(instance):
                raise
            return False
        return True
