"""Serve one pinned student artifact for a campaign evaluation arm."""
import contextlib
import json
import socket
import subprocess
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

from ..provenance import file_sha256
from .containers import Container
from .execution import atomic_json
from .journal import CampaignFault, digest


class Serving:
    def __init__(self, directory, configuration, run=subprocess.run, health=None, sleep=time.sleep, clock=time.monotonic):
        self.directory, self.configuration = Path(directory), configuration
        self.run, self.sleep, self.clock = run, sleep, clock
        self.health = health or self.read_health
        self.containers = Container(directory, run)

    @staticmethod
    def read_health(url):
        with urlopen(url + '/health', timeout=5) as response:
            return json.load(response).get('status') == 'ok'

    def plan(self, model, checksum, context, model_name):
        model = Path(model).resolve()
        if file_sha256(model) != checksum:
            raise CampaignFault('serving_artifact_changed')
        cfg = self.configuration
        if not cfg['image'].startswith('sha256:') or context['thinking'] is not False or context['truncation'] != 'forbidden':
            raise CampaignFault('serving_configuration_invalid')
        binding = {'model': str(model), 'model_sha256': checksum, 'context': context,
                   'model_name': model_name, 'configuration': cfg}
        path = self.directory / 'serving.json'
        if path.exists():
            plan = json.loads(path.read_text())
            if plan['binding'] != binding:
                raise CampaignFault('serving_inputs_changed')
            return plan
        with socket.socket() as sock:
            sock.bind((cfg['gateway'], 0))
            port = sock.getsockname()[1]
        name = 'bialy-student-' + digest({'directory': str(self.directory.resolve()), 'binding': binding})[:24]
        arguments = ['docker', 'create', '--name', name, '--label', 'bialy.serving=' + digest(binding),
                     '--gpus', cfg.get('gpus', 'device=0'), '--network', 'host',
                     '--cap-drop', 'ALL', '--security-opt', 'no-new-privileges', '--read-only',
                     '--tmpfs', '/tmp:rw',
                     '-v', str(model) + ':/models/model.gguf:ro', cfg['image'],
                     '--model', '/models/model.gguf', '--alias', model_name, '--host', cfg['gateway'], '--port', str(port),
                     '--ctx-size', str(context['max_sequence_tokens']), '--parallel', '1',
                     '--threads', '2', '--threads-batch', '2', '--n-gpu-layers', '999',
                     '--jinja', '--reasoning', 'off', '--no-context-shift', '--no-warmup']
        plan = {'binding': binding, 'name': name, 'arguments': arguments,
                'url': 'http://' + cfg['gateway'] + ':' + str(port), 'port': port}
        atomic_json(path, plan)
        return plan

    def checked_state(self, plan):
        state = self.containers.inspect(plan['name'])
        if state is not None and (state.get('Config', {}).get('Labels', {}).get('bialy.serving') != digest(plan['binding'])
                                  or state.get('Image') != self.configuration['image']):
            raise CampaignFault('serving_container_binding_mismatch')
        return state

    def start(self, plan):
        state = self.checked_state(plan)
        if state is None:
            try:
                self.run(plan['arguments'], check=True, capture_output=True, text=True)
            except (subprocess.SubprocessError, OSError):
                if self.checked_state(plan) is None:
                    raise CampaignFault('serving_container_creation_failed') from None
            state = self.checked_state(plan)
        if state is None:
            raise CampaignFault('serving_container_missing')
        if state['State']['Status'] == 'created':
            self.run(['docker', 'start', state['Id']], check=True, capture_output=True, text=True)
        deadline = self.clock() + self.configuration.get('startup_timeout_seconds', 300)
        while self.clock() < deadline:
            state = self.checked_state(plan)
            if state is None or state['State']['Status'] != 'running':
                raise CampaignFault('student_server_exited', details={'state': state['State'] if state else None})
            try:
                if self.health(plan['url']):
                    self.access(plan, True)
                    atomic_json(self.directory / 'ready.json', {'container_id': state['Id'], 'binding_sha256': digest(plan['binding'])})
                    return plan['url'] + '/v1'
            except (URLError, TimeoutError, OSError, ValueError):
                pass
            self.sleep(1)
        raise CampaignFault('student_server_startup_failed')

    def access(self, plan, enabled):
        if self.configuration['gateway'] == '127.0.0.1':
            return
        rule = ['INPUT', '-i', 'br-' + self.configuration['network'], '-d', self.configuration['gateway'],
                '-p', 'tcp', '--dport', str(plan['port']), '-m', 'comment', '--comment', plan['name'], '-j', 'ACCEPT']
        prefix = self.configuration.get('firewall_prefix', ['sudo', '-n']) + ['iptables']
        found = self.run([*prefix, '-C', *rule], capture_output=True, text=True)
        if found.returncode not in (0, 1):
            raise CampaignFault('serving_network_inspection_failed')
        if enabled and found.returncode == 1:
            self.run([*prefix, '-I', *rule], check=True, capture_output=True, text=True)
        elif not enabled and found.returncode == 0:
            self.run([*prefix, '-D', *rule], check=True, capture_output=True, text=True)

    def stop(self, plan):
        state = self.checked_state(plan)
        self.access(plan, False)
        if state is None:
            return
        identifier = state['Id']
        if state['State']['Status'] == 'running':
            self.run(['docker', 'stop', '--time', '30', identifier], check=True, capture_output=True, text=True)
        with (self.directory / ('server-' + identifier[:12] + '.log')).open('ab') as log:
            self.run(['docker', 'logs', identifier], check=True, stdout=log, stderr=subprocess.STDOUT)
        self.run(['docker', 'rm', identifier], check=True, capture_output=True, text=True)
        atomic_json(self.directory / 'stopped.json', {'container_id': identifier, 'binding_sha256': digest(plan['binding'])})

    @contextlib.contextmanager
    def session(self, model, checksum, context, model_name):
        plan = self.plan(model, checksum, context, model_name)
        try:
            yield self.start(plan)
        finally:
            self.stop(plan)
