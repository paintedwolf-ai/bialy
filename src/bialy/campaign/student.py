"""Campaign stages for pinned training jobs and verified student exports."""
import json
import os
import subprocess
import sys
from pathlib import Path

from ..provenance import file_sha256
from ..student_host import exporting, jobs, remote_training
from ..student_host.specification import identity, load
from .execution import atomic_json
from .journal import CampaignFault, digest
from .sources import authority_key


class Training:
    def __init__(self, journal, configuration, student, pilot=False, run=subprocess.run, *, dry_run=False):
        self.journal, self.configuration = journal, configuration
        self.student, self.pilot, self.run, self.dry_run = student, pilot, run, dry_run
        self.specification = load(configuration['specification'])
        self.destination = configuration.get('pilot_training', configuration['full_training']) if pilot else configuration['full_training']
        if dry_run and self.destination != 'local':
            raise CampaignFault('dry_run_requires_local_training')
        if self.destination not in {'local', 'remote'}:
            raise CampaignFault('student_training_destination_invalid')

    def directory(self, attempt):
        return self.journal.directory / 'students' / self.student / ('pilot' if self.pilot else 'full') / attempt

    def dataset(self, inputs):
        data = inputs['dataset']['students'][self.student]
        directory = Path(data['directory'])
        names = ('train.jsonl', 'val.jsonl', 'holdout.jsonl', 'source.json')
        if set(data['files']) != set(names) or any(file_sha256(directory / name) != data['files'][name] for name in names):
            raise CampaignFault('student_dataset_changed')
        if data['specification_sha256'] != identity(self.specification):
            raise CampaignFault('student_dataset_context_changed')
        return directory

    def execute(self, inputs, attempt):
        directory = self.directory(attempt)
        directory.mkdir(parents=True, exist_ok=False)
        atomic_json(directory / 'binding.json', {'inputs_sha256': digest(inputs), 'configuration': self.configuration,
                                                'student': self.student, 'pilot': self.pilot, 'dry_run': self.dry_run})
        splits = self.dataset(inputs)
        if self.destination == 'remote':
            result = remote_training.train(self.configuration['remote'], self.student, splits,
                                  authority_key(self.journal.directory), directory / 'remote', pilot=self.pilot)
            job = Path(result['directory']) / 'job'
        else:
            job = jobs.create(directory / 'jobs', self.specification, self.student, self.configuration['checkout'],
                              self.configuration['models'], splits, self.configuration['mirror'],
                              self.configuration['device'], self.pilot, dry_run=self.dry_run)
            self.train_local(job)
        return self.result(job)

    def train_local(self, job):
        environment = dict(os.environ, PW_TRAINING_SOURCE_KEY=str(authority_key(self.journal.directory)))
        with (Path(job) / 'controller.log').open('a') as log:
            result = self.run([sys.executable, '-m', 'bialy.student_host', 'train', str(job)],
                              env=environment, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            raise CampaignFault('student_training_failed', details={'job': str(job), 'exit_status': result.returncode})

    def reconcile(self, inputs, attempt):
        directory = self.directory(attempt)
        expected = {'inputs_sha256': digest(inputs), 'configuration': self.configuration,
                    'student': self.student, 'pilot': self.pilot, 'dry_run': self.dry_run}
        if not (directory / 'binding.json').is_file() or json.loads((directory / 'binding.json').read_text()) != expected:
            raise CampaignFault('student_job_binding_unknown')
        self.dataset(inputs)
        if self.destination == 'remote':
            instances = list((directory / 'remote').glob('*'))
            if len(instances) != 1:
                raise CampaignFault('student_remote_job_identity_unknown')
            result = remote_training.train(self.configuration['remote'], self.student, self.dataset(inputs),
                                  authority_key(self.journal.directory), directory / 'remote', resume=instances[0], pilot=self.pilot)
            return self.result(Path(result['directory']) / 'job')
        candidates = list((directory / 'jobs').glob('*/plan.json'))
        if len(candidates) != 1:
            raise CampaignFault('student_job_identity_unknown')
        job = candidates[0].parent
        # The job controller reconciles terminal receipts before resuming work.
        self.train_local(job)
        return self.result(job)

    def result(self, job):
        plan = json.loads((Path(job) / 'plan.json').read_text())
        jobs.validate_inputs(plan)
        terminal = jobs.validate_result(plan)
        return {'job': str(job), 'plan_sha256': digest(plan), 'terminal_sha256': digest(terminal),
                'student': self.student, 'pilot': self.pilot, 'dry_run': self.dry_run}

    def validate(self, inputs, output):
        splits = self.dataset(inputs)
        plan = json.loads((Path(output['job']) / 'plan.json').read_text())
        if (plan['student'] != self.student or plan['pilot'] is not self.pilot or plan.get('dry_run', False) is not self.dry_run
                or plan['host_commit'] != self.configuration['host_commit']
                or plan['specification_sha256'] != identity(self.specification)
                or plan['dataset'] != {name: file_sha256(splits / name) for name in plan['dataset']}
                or self.result(output['job']) != output):
            raise CampaignFault('student_job_result_mismatch')


class Export:
    def __init__(self, configuration, student):
        self.configuration, self.student = configuration, student

    def execute(self, inputs, attempt):
        job = inputs['training']['job']
        if exporting.returned(job):
            return exporting.publish(job, self.configuration['llama_revision'], unquantized=False)
        return exporting.export(job, self.configuration['llama'], self.configuration['llama_revision'])

    def reconcile(self, inputs, attempt):
        return self.execute(inputs, attempt)

    def validate(self, inputs, output):
        job = Path(inputs['training']['job'])
        plan = json.loads((job / 'plan.json').read_text())
        jobs.validate_inputs(plan)
        jobs.validate_result(plan)
        if (digest(plan) != inputs['training']['plan_sha256'] or output['student'] != self.student
                or output['host_commit'] != self.configuration['host_commit']
                or output['specification_sha256'] != plan['specification_sha256']
                or output['context'] != plan['specification']['context']
                or output['files'] != jobs.tree(Path(output['directory']))
                or json.loads((job / 'artifact.json').read_text()) != output):
            raise CampaignFault('student_export_changed')
