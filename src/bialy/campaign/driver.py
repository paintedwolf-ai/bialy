"""Drive an isolated Painted Wolf Code task and retain its graded episode."""
import json
import sqlite3
import subprocess
from pathlib import Path
from urllib.parse import quote

import yaml

from .. import fleet
from ..provenance import file_sha256
from . import evidence_reader, host_facts, oracle
from .containers import Container
from .episodes import Episodes
from .execution import atomic_json
from .journal import CampaignFault, digest
from .preflight import inspect


class EpisodeDriver:
    def __init__(self, journal, lineage, factory_run, reviewers, oracle_root, process=subprocess.run):
        self.journal, self.lineage, self.factory_run = journal, lineage, factory_run
        self.reviewers, self.oracle_root, self.process = reviewers, Path(oracle_root).resolve(), process

    def directory(self, attempt):
        return self.journal.directory / 'trials' / attempt

    def validate_task(self, task):
        registered = self.lineage.task(task['id'])
        if registered['spec_hash'] != digest(task):
            raise CampaignFault('trial_task_changed')
        workspace = self.factory_run.factory.workspace(task['workspace'])
        if (registered['source_class'] == 'synthetic' and workspace.kind != 'stack'
                or registered['source_class'] == 'permitted_repository' and not task.get('permission_sha256')
                or task['project_path'] != '/work/' + workspace.name):
            raise CampaignFault('trial_source_not_permitted')
        directory = (self.oracle_root / task['oracle_directory']).resolve()
        if not directory.is_relative_to(self.oracle_root):
            raise CampaignFault('trial_oracle_outside_bank')
        return workspace, directory

    def execute(self, name, inputs, attempt):
        custom = self.factory_run.setting('custom_model')
        inspect(self.factory_run.checkout, custom['host_commit'], tool_surface=custom['tool_surface'])
        directory = self.directory(attempt)
        directory.mkdir(parents=True, exist_ok=False)
        task = inputs['task']
        workspace, oracle_directory = self.validate_task(task)
        shard = directory / 'fleet' / 'shards' / attempt
        shard.mkdir(parents=True)
        custom = self.factory_run.setting('custom_model')
        selector = inputs['selector'].get('pilot')
        environment = fleet.pilot_env(Path(selector) if selector else None)
        environment.update(BIALY_CAMPAIGN_CAPTURE='1', PW_TRAINING_CAPTURE_DIR='/shard/training-captures',
                           PW_TRAINING_EPOCH=custom['epoch'],
                           PW_TRAINING_STUDENTS=json.dumps([{'provider': item['provider'], 'model': item['model']}
                                                           for item in custom['students'].values()]))
        providers, policy, env, chosen, wire = fleet.sidecar_files(self.factory_run.factory, environment, 0)
        actor = inputs['actor']
        if actor['kind'] == 'teacher':
            reference = {'provider_id': chosen[actor['id']], 'model': wire[actor['id']]}
        else:
            reference = {'provider_id': actor['provider'], 'model': actor['model']}
            if not any(item['id'] == actor['provider'] for item in yaml.safe_load(providers)['providers']):
                raise CampaignFault('student_serving_provider_missing')
        policy = yaml.safe_load(policy)
        policy['coordinator'] = policy['lite'] = reference
        policy['agent_pool'] = {'selection': 'first', 'models': [reference]}
        for filename, body in (('providers.local.yaml', providers), ('model-policy.yaml', yaml.safe_dump(policy)), ('sidecar.env', env)):
            (shard / filename).write_text(body)
        spec = {'workspace': workspace.name, 'kind': workspace.kind,
                'prompt_timeout': task.get('prompt_timeout') or self.factory_run.factory.fleet['prompt_timeout']}
        if workspace.kind == 'repository':
            spec['commit'] = workspace.commit
            if spec['commit'] != task['repository_commit']:
                raise CampaignFault('trial_repository_revision_changed')
        atomic_json(shard / 'workspace.json', spec)
        request = {key: task[key] for key in ('id', 'prompt', 'follow_ups') if key in task}
        request.update(provider_id=reference['provider_id'], model=reference['model'])
        request['meta'] = {'task_id': task['id'], 'prompt_group': self.lineage.task(task['id'])['root'], 'model': actor['id']}
        (shard / 'tasks.jsonl').write_text(json.dumps(request) + '\n')
        run_name = 'campaign-' + digest(str(self.journal.directory.resolve()))[:12]
        mounts = [(selector, fleet.PILOT_MOUNT)] if selector else []
        container_name, work, arguments = fleet.container_args(self.factory_run.factory, run_name, shard, mounts, inputs['host']['image'])
        oracle.seed(task, oracle_directory, work / 'work' / workspace.name)
        plan = {'inputs': inputs, 'inputs_sha256': digest(inputs), 'shard': str(shard), 'work': str(work),
                'name': container_name, 'arguments': arguments, 'request_files': {
                    path.name: file_sha256(path) for path in shard.iterdir() if path.is_file()}}
        atomic_json(directory / 'plan.json', plan)
        Container(directory, self.process).execute(container_name, arguments, inputs)
        return self.settle(inputs, attempt)

    def reconcile(self, name, inputs, attempt):
        directory = self.directory(attempt)
        if not (directory / 'plan.json').exists():
            raise CampaignFault('trial_start_outcome_unknown')
        plan = json.loads((directory / 'plan.json').read_text())
        if plan['inputs_sha256'] != digest(inputs):
            raise CampaignFault('trial_inputs_changed')
        result = directory / 'result.json'
        if result.exists():
            Container(directory, self.process).release()
            fleet.release_work(Path(plan['work']))
            return json.loads(result.read_text())
        Container(directory, self.process).reconcile()
        return self.settle(inputs, attempt)

    def settle(self, inputs, attempt):
        directory = self.directory(attempt)
        plan = json.loads((directory / 'plan.json').read_text())
        shard, work = Path(plan['shard']), Path(plan['work'])
        task, actor = inputs['task'], inputs['actor']
        workspace, oracle_directory = self.validate_task(task)
        for filename, checksum in plan['request_files'].items():
            if file_sha256(shard / filename) != checksum:
                raise CampaignFault('trial_request_changed')
        manifest = [json.loads(line) for line in (shard / 'manifest.jsonl').read_text().splitlines() if line.strip()]
        if len(manifest) != 1 or manifest[0]['task_id'] != task['id']:
            raise CampaignFault('trial_manifest_mismatch')
        root, store = manifest[0]['root_session'], shard / 'store.db'
        epoch = self.factory_run.setting('custom_model')['epoch']
        checksum = file_sha256(store)
        facts_path = directory / 'host-evidence.json'
        facts = evidence_reader.read(shard, work / 'work' / workspace.name, root, inputs['host']['image'], facts_path, self.process)
        sessions = host_facts.members(facts, root)
        self.lineage.bind_session(epoch, root, task['id'])
        for session in sessions:
            if session != root:
                child_id = 'worker-' + session
                child_spec = {'session_id': session, 'parent_session_id': sessions[session]['parent_id']}
                prior = self.journal.db.execute('SELECT spec_hash FROM tasks WHERE id=?', (child_id,)).fetchone()
                if prior is None:
                    self.lineage.descendant(child_id, task['id'], child_spec)
                elif prior[0] != digest(child_spec):
                    raise CampaignFault('worker_lineage_changed')
                self.lineage.bind_session(epoch, session, child_id)
        inventory_path = directory / 'source-inventory.json'
        inventory = json.loads(inventory_path.read_text()) if inventory_path.exists() else oracle.snapshot(
            work / 'work' / workspace.name, directory / 'source', task.get('artifact_paths', []), self.process)
        checks = oracle.grade(task, directory / 'source', inventory, oracle_directory, directory,
                              inputs['host']['image'], directory / 'grading', self.process)
        with sqlite3.connect('file:' + quote(str(store.resolve()), safe='/') + '?mode=ro', uri=True) as database:
            database.row_factory = sqlite3.Row
            surface = self.factory_run.setting('custom_model')['tool_surface']
            programs = host_facts.program_evidence(database, facts) if surface == 'command-v1' else {}
        receipt = host_facts.receipt(task, facts, checks, checksum, epoch, actor['family'], checks['grader_identity'], programs, surface)
        if file_sha256(store) != checksum:
            raise CampaignFault('trial_store_changed')
        reviews_path = directory / 'reviews.json'
        reviews = json.loads(reviews_path.read_text()) if reviews_path.exists() else self.reviewers.review(receipt, facts['sessions'])
        atomic_json(reviews_path, reviews)
        verdict = Episodes(self.journal, self.lineage).record(epoch, root, task['id'], list(sessions), task, receipt, reviews, checks['grader_identity'])
        controls = None
        if actor['kind'] == 'student':
            captures = [json.loads(path.read_text()) for path in (shard / 'training-captures').glob('*.json')]
            controls = host_facts.student_controls(captures, actor['provider'], actor['model'])
        terminal = Container(directory, self.process).reconcile()
        result = {'success': verdict['verdict'] == 'pass' and terminal['exit_status'] == 0,
                  'human_interventions': receipt['human_interventions'], 'unauthorized_effects': receipt['unauthorized_effects'],
                  'unsupported_completion_claims': int(any(review['dimensions']['truthful_closeout'] != 'pass' for review in reviews)),
                  'effective_thinking': controls, 'receipt': receipt, 'verdict': verdict, 'reviews': reviews,
                  'fleet': str(shard.parent.parent), 'shard': str(shard), 'host_evidence_sha256': file_sha256(facts_path),
                  'directory': str(directory), 'inputs_sha256': digest(inputs)}
        atomic_json(directory / 'result.json', result)
        Container(directory, self.process).release()
        fleet.release_work(work)
        return result

    def validate(self, name, inputs, output):
        directory = Path(output['directory'])
        if (output['inputs_sha256'] != digest(inputs) or json.loads((directory / 'result.json').read_text()) != output
                or file_sha256(Path(output['shard']) / 'store.db') != output['receipt']['store_sha256']
                or file_sha256(directory / 'host-evidence.json') != output['host_evidence_sha256']):
            raise CampaignFault('trial_evidence_changed')
