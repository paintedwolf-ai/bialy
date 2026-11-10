"""Vast.ai container CLI operations restricted to owned label-scoped instances."""
import json
import subprocess

from .resources import ResourceError

ALLOWED_SPECIFICATION = {'gpu_name', 'image', 'disk', 'num_gpus', 'bid_price'}
REQUIRED_SPECIFICATION = {'gpu_name', 'image', 'disk'}


class VastAI:
    def __init__(self, api_key_file=None, run=subprocess.run):
        self.api_key_file, self.run = api_key_file, run

    def call(self, *arguments):
        command = ['vastai', '--raw']
        if self.api_key_file:
            command.extend(['--api-key', self.api_key_file])
        command.extend(arguments)
        result = self.run(command, capture_output=True, text=True, timeout=120)
        if result.returncode:
            raise ResourceError('Vast.ai operation did not confirm success')
        return json.loads(result.stdout) if result.stdout.strip() else None

    def instances(self):
        result = self.call('show', 'instances')
        if not isinstance(result, list):
            raise ResourceError('Vast.ai instance list returned an unexpected shape')
        return result

    @staticmethod
    def _normalize(instance, token):
        label = instance.get('label')
        if label != 'pw-training-' + token:
            raise ResourceError('Vast.ai resource ownership mismatch')
        identifier = str(instance.get('id') or '')
        host = instance.get('ssh_host') or instance.get('public_ipaddr') or ''
        port = instance.get('ssh_port')
        if not identifier or not host or type(port) is not int or not 0 < port < 65536:
            raise ResourceError('Vast.ai instance is missing SSH connection details')
        return {
            'id': identifier,
            'label': label,
            'tags': [token],
            'public_ip': {'id': identifier, 'address': host},
            'ssh_port': port,
        }

    def find(self, token):
        label = 'pw-training-' + token
        matched = [item for item in self.instances() if item.get('label') == label or token in (item.get('tags') or [])]
        return [self._normalize(item, token) for item in matched]

    def create(self, token, specification):
        if not isinstance(specification, dict) or not REQUIRED_SPECIFICATION <= set(specification) or set(specification) - ALLOWED_SPECIFICATION:
            raise ResourceError('invalid rental specification')
        num_gpus = str(specification.get('num_gpus', '1'))
        disk = str(specification['disk'])
        query_parts = [
            'gpu_name=' + str(specification['gpu_name']),
            'num_gpus=' + num_gpus,
            'disk_space>=' + disk,
            'rentable=true',
            'verified=true',
        ]
        bid_price = specification.get('bid_price')
        if bid_price is not None:
            query_parts.append('dph_total<=' + str(bid_price))
        offers = self.call('search', 'offers', '--order', 'dph_total asc', ' '.join(query_parts))
        if not isinstance(offers, list) or not offers or not offers[0].get('id'):
            raise ResourceError('Vast.ai found no matching container offers')
        offer_id = str(offers[0]['id'])
        create_args = [
            'create', 'instance', offer_id,
            '--image', str(specification['image']),
            '--disk', disk,
            '--ssh', '--direct',
            '--label', 'pw-training-' + token,
        ]
        if bid_price is not None:
            create_args.extend(['--bid_price', str(bid_price)])
        created = self.call(*create_args)
        if not isinstance(created, dict) or not created.get('new_contract'):
            raise ResourceError('Vast.ai creation returned an invalid contract identity')
        contract_id = str(created['new_contract'])
        servers = [s for s in self.find(token) if s['id'] == contract_id]
        if len(servers) != 1:
            raise ResourceError('Vast.ai created instance could not be verified')
        return servers[0]

    def remaining(self, server):
        expected = str(server['id'])
        return [str(item.get('id')) for item in self.instances() if str(item.get('id')) == expected]

    def absent(self, server):
        return not self.remaining(server)

    def delete(self, server):
        identifier = str(server['id'])
        if self.remaining(server):
            self.call('destroy', 'instance', identifier)
        if self.remaining(server):
            raise ResourceError('Vast.ai instance deletion is not settled')
