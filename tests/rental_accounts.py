"""Fake provider accounts behind the real provider clients. Each account answers the CLI
calls the client makes and can fail any one of them, so a test can walk every failure point
of a rental instead of choosing a few."""
import json
import subprocess
from types import SimpleNamespace

from bialy.student_host import machines
from bialy.student_host.scaleway import Scaleway
from bialy.student_host.vastai import VastAI

# How one provider call can fail. `refused` has no effect on the account; `lost` and
# `timeout` take effect and then lose the answer; `outage` refuses this and every later
# call; `death` kills the controller at this call, so nothing it would do next happens.
FAULTS = ('refused', 'lost', 'timeout', 'outage', 'death')


class Died(BaseException):
    """The controller process is gone: no handler and no cleanup of its own runs."""


class Account:
    """State and fault injection shared by both providers."""

    def __init__(self, lag=0, boot=1):
        # A new instance is missing from the first `lag` listings and reports an endpoint after `boot` more.
        self.lag, self.boot = lag, boot
        self.calls, self.created, self.fault, self.dead = [], 0, None, False
        self.foreign()

    def fail(self, index, kind):
        self.fault = (index, kind)

    def heal(self):
        self.fault, self.dead = None, False

    def __call__(self, command, **kwargs):
        if self.dead:
            raise Died()
        index, kind = len(self.calls), None
        self.calls.append(command)
        if self.fault and (index == self.fault[0] or (self.fault[1] == 'outage' and index > self.fault[0])):
            kind = self.fault[1]
        if kind in ('refused', 'outage'):
            return SimpleNamespace(returncode=1, stdout='')
        answer = self.answer(command)
        if kind == 'death':
            self.dead = True
            raise Died()
        if kind == 'timeout':
            raise subprocess.TimeoutExpired(command, kwargs.get('timeout'))
        if kind == 'lost' or answer is None:
            return SimpleNamespace(returncode=1, stdout='')
        return SimpleNamespace(returncode=0, stdout=answer if isinstance(answer, str) else json.dumps(answer))

    def visible(self, items):
        shown = []
        for item in items:
            item['listed'] += 1
            if item['listed'] > self.lag:
                shown.append(item)
        return shown


class VastAccount(Account):
    name = 'vastai'

    def foreign(self):
        self.instances = {1: {'id': 1, 'label': 'someone-else', 'listed': 99, 'is_bid': False, 'machine_id': 1}}
        # Cheapest first, as the search is ordered.
        self.offers = [{'id': 777, 'machine_id': 70}, {'id': 778, 'machine_id': 71}]

    def client(self):
        return VastAI(run=self)

    def owned(self):
        return [item for item in self.instances.values() if item['label'] != 'someone-else']

    def answer(self, command):
        assert command[:2] == ['vastai', '--raw']
        action, arguments = command[2:4], command[4:]
        if action == ['show', 'instances']:
            rows = []
            for item in self.visible(self.instances.values()):
                ready = item['label'] != 'someone-else' and item['listed'] > self.lag + self.boot
                rows.append({'id': item['id'], 'label': item['label'], 'is_bid': item['is_bid'],
                             'actual_status': 'running' if ready else 'loading',
                             'machine_id': item['machine_id'], 'ssh_host': 'ssh.example.org', 'ssh_port': 41234,
                             'public_ipaddr': '203.0.113.7' if ready else None,
                             'ports': {'22/tcp': [{'HostIp': '0.0.0.0', 'HostPort': '41235'}]} if ready else {}})
            return rows
        if action == ['search', 'offers']:
            return [dict(offer) for offer in self.offers]
        if action == ['create', 'instance']:
            self.created += 1
            identifier = 9000 + self.created
            machine = next(offer['machine_id'] for offer in self.offers if str(offer['id']) == arguments[0])
            self.instances[identifier] = {'id': identifier, 'machine_id': machine, 'label': arguments[arguments.index('--label') + 1], 'listed': 0,
                                          'is_bid': '--bid_price' in arguments,
                                          'startup': arguments[arguments.index('--onstart-cmd') + 1]}
            return {'success': True, 'new_contract': identifier}
        if action == ['destroy', 'instance']:
            if '--yes' not in arguments:
                return 'Aborted.\n'
            return {'success': True} if self.instances.pop(int(arguments[0]), None) else None
        raise AssertionError('unexpected command: %r' % (command,))


class ScalewayAccount(Account):
    name = 'scaleway'

    def foreign(self):
        self.servers = {'foreign': {'id': 'foreign', 'name': 'other', 'tags': [], 'project': 'project', 'listed': 99,
                                    'volumes': {}, 'public_ips': []}}
        self.children = {('instance', 'volume'): set(), ('block', 'volume'): {'foreign-volume'}, ('instance', 'ip'): {'foreign-ip'}}
        # A server deleted while this is set leaves its volume and address behind.
        self.orphans = False

    def client(self):
        return Scaleway('profile', 'project', 'zone', run=self)

    def owned(self):
        left = [item for item in self.servers.values() if item['id'] != 'foreign']
        return left + sorted(self.children[('block', 'volume')] - {'foreign-volume'}) + sorted(self.children[('instance', 'ip')] - {'foreign-ip'})

    def answer(self, command):
        assert command[:5] == ['scw', '-p', 'profile', '-o', 'json'] and command[-1] == 'zone=zone'
        (service, kind, action), arguments = command[5:8], command[8:-1]
        if action == 'list':
            assert 'project-id=project' in arguments
            if kind != 'server':
                return [{'id': identifier} for identifier in sorted(self.children[(service, kind)])]
            tags = [value[5:] for value in arguments if value.startswith('tags=')]
            rows = []
            for item in self.visible([item for item in self.servers.values() if set(tags) <= set(item['tags'])]):
                ready = item['id'] != 'foreign' and item['listed'] > self.lag + self.boot
                rows.append({**{key: item[key] for key in ('id', 'name', 'tags', 'project', 'volumes', 'public_ips')},
                             'state': 'running' if ready else 'starting'})
            return rows
        if (kind, action) == ('server', 'create'):
            self.created += 1
            fields = dict(value.split('=', 1) for value in arguments)
            identifier = 'server-%d' % self.created
            self.servers[identifier] = {'id': identifier, 'name': fields['name'], 'tags': [fields['tags.0']], 'project': 'project',
                                        'listed': 0, 'startup': fields['cloud-init'],
                                        'volumes': {'0': {'id': identifier + '-volume'}},
                                        'public_ips': [{'id': identifier + '-ip', 'address': '192.0.2.7'}]}
            self.children[('block', 'volume')].add(identifier + '-volume')
            self.children[('instance', 'ip')].add(identifier + '-ip')
            return {**{key: self.servers[identifier][key] for key in ('id', 'name', 'tags', 'project', 'volumes', 'public_ips')},
                    'state': 'starting'}
        if action == 'delete' and kind == 'server':
            server = self.servers.pop(arguments[0], None)
            if server is None:
                return None
            if not self.orphans:
                self.children[('block', 'volume')].discard(server['id'] + '-volume')
                self.children[('instance', 'ip')].discard(server['id'] + '-ip')
            return {}
        if action == 'delete':
            if arguments[0] not in self.children[(service, kind)]:
                return None
            self.children[(service, kind)].discard(arguments[0])
            return {}
        raise AssertionError('unexpected command: %r' % (command,))


ACCOUNTS = {'vastai': VastAccount, 'scaleway': ScalewayAccount}


class Machine:
    """The rented machine as the controller reaches it: reachable, with a dead-man switch that answers heartbeats."""
    host = 'fixture'

    def __init__(self, address, settings, switch=None):
        self.address, self.settings, self.commands = address, settings, []
        self.switch = {'now': 100, 'deadline': 1000, 'limit': 5000, 'watcher_age': 1, 'expired': False, 'can_end_rental': True,
                       **(switch or {})}

    def command(self, args, **kwargs):
        self.commands.append(args)
        return json.dumps(self.switch) if args[0] == machines.DEADMAN else ''


def configuration(tmp_path, provider, **specification):
    """A machine configuration that rents from `provider` and can hold every tier."""
    capacity = {'gpu_memory_gib': 79, 'memory_gib': 128, 'disk_gib': 320}
    if provider == 'vastai':
        provision = {'provider': 'vastai', 'specification': {'gpu_name': 'H100_SXM', 'image': 'example/image:1', 'max_price': 3.5, 'download_mbps': 500,
                                                             **capacity, **specification}}
    else:
        provision = {'provider': 'scaleway', 'profile': 'profile', 'project': 'project', 'zone': 'zone',
                     'specification': {'type': 'H100-1-80G', 'image': 'image', 'root_volume': '1TB', **capacity, **specification}}
    provision.update(registry=str(tmp_path / 'registry'), rental='on-demand', lifetime_hours=24)
    return {'host_repository': 'https://example.invalid/host.git', 'host_commit': 'a' * 40,
            'bialy_repository': 'https://example.invalid/bialy.git', 'bialy_commit': 'b' * 40,
            'llama_repository': 'https://example.invalid/llama.cpp.git', 'llama_commit': 'c' * 40,
            'host_checkout': str(tmp_path / 'host'), 'models': str(tmp_path / 'models'), 'provisioning': provision,
            'machine': {'user': 'root', 'workspace': '/workspace/training', 'uv': '/root/.local/bin/uv'}}
