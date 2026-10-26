"""Campaign Docker executions reconcile by immutable labels without replaying tasks."""
import json
import subprocess
from pathlib import Path

from .execution import atomic_json
from .journal import CampaignFault, digest


class Container:
    def __init__(self, directory, run=subprocess.run):
        self.directory, self.run = Path(directory), run

    def inspect(self, name):
        result = self.run(['docker', 'container', 'inspect', name], capture_output=True, text=True)
        if result.returncode:
            # A daemon outage must not be interpreted as a missing container.
            self.run(['docker', 'info', '--format', '{{.ID}}'], check=True, capture_output=True, text=True)
            listed = self.run(['docker', 'container', 'ls', '-a', '--format', '{{.Names}}'], check=True, capture_output=True, text=True)
            if name in listed.stdout.splitlines():
                raise CampaignFault('container_inspection_failed')
            return None
        records = json.loads(result.stdout)
        if len(records) != 1:
            raise CampaignFault('container_inspection_ambiguous')
        return records[0]

    def execute(self, name, arguments, inputs):
        self.directory.mkdir(parents=True, exist_ok=True)
        plan_path = self.directory / 'container.json'
        plan = {'name': name, 'arguments': arguments, 'inputs_sha256': digest(inputs)}
        if plan_path.exists():
            if json.loads(plan_path.read_text()) != plan:
                raise CampaignFault('container_plan_changed')
            return self.reconcile()
        atomic_json(plan_path, plan)
        command = list(arguments)
        if command[:2] != ['docker', 'run'] or command[command.index('--name') + 1] != name:
            raise CampaignFault('container_command_invalid')
        command[1] = 'create'
        if '--rm' in command:
            command.remove('--rm')
        command[2:2] = ['--label', 'bialy.campaign=' + plan['inputs_sha256']]
        try:
            self.run(command, check=True, capture_output=True, text=True)
        except (subprocess.SubprocessError, OSError):
            # Creation may have succeeded before the client lost its response.
            return self.observe(plan)
        return self.observe(plan)

    def reconcile(self):
        plan = json.loads((self.directory / 'container.json').read_text())
        terminal = self.directory / 'container-terminal.json'
        if terminal.exists():
            result = json.loads(terminal.read_text())
            if result['inputs_sha256'] != plan['inputs_sha256'] or result['name'] != plan['name']:
                raise CampaignFault('container_terminal_changed')
            return result
        return self.observe(plan)

    def observe(self, plan):
        state = self.inspect(plan['name'])
        if state is None:
            raise CampaignFault('container_outcome_unknown')
        if state.get('Config', {}).get('Labels', {}).get('bialy.campaign') != plan['inputs_sha256']:
            raise CampaignFault('container_binding_mismatch')
        status = state['State']['Status']
        if status == 'created':
            self.run(['docker', 'start', plan['name']], check=True, capture_output=True, text=True)
            status = 'running'
        if status in {'running', 'restarting'}:
            self.run(['docker', 'wait', plan['name']], check=True, capture_output=True, text=True)
            state = self.inspect(plan['name'])
            if state is None:
                raise CampaignFault('container_outcome_unknown')
        if state['State']['Status'] != 'exited':
            raise CampaignFault('container_unsettled')
        result = {'name': plan['name'], 'container_id': state['Id'], 'inputs_sha256': plan['inputs_sha256'],
                  'exit_status': state['State']['ExitCode'], 'oom_killed': state['State'].get('OOMKilled', False)}
        atomic_json(self.directory / 'container-terminal.json', result)
        return result

    def release(self):
        result = self.reconcile()
        state = self.inspect(result['name'])
        if state is not None:
            if (state['Id'] != result['container_id'] or state['State']['Status'] != 'exited'
                    or state.get('Config', {}).get('Labels', {}).get('bialy.campaign') != result['inputs_sha256']):
                raise CampaignFault('container_binding_mismatch')
            self.run(['docker', 'rm', result['name']], check=True, capture_output=True, text=True)
        atomic_json(self.directory / 'container-released.json', result)
