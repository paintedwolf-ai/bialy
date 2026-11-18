"""Campaign stages for pinned training jobs and verified student exports."""
import json
import os
import subprocess
import sys
from pathlib import Path

from ..provenance import file_sha256
from ..student_host import exporting, jobs, remote_training
from ..student_host.jobs import JobError
from ..student_host.specification import identity, load
from .execution import atomic_json
from .journal import CampaignFault, digest
from .sources import authority_key


def training_fault(job, exit_status):
    """The job's typed failure. Running out of memory is terminal for the recipe and never retried."""
    plan = Path(job) / 'plan.json'
    terminal = Path(json.loads(plan.read_text())['out']) / 'terminal.json' if plan.exists() else None
    failure = json.loads(terminal.read_text()).get('failure', {}) if terminal and terminal.exists() else {}
    if failure.get('code') == 'out_of_memory':
        return CampaignFault('student_out_of_memory', details={'job': str(job)})
    return CampaignFault('student_training_failed', details={'job': str(job), 'exit_status': exit_status})


# Why a remote job found no machine to start on. The job has already waited and tried other
# machines within its own bounds, so the stage is not retried.
UNAVAILABLE = {'no_offer', 'machine_unusable', 'ledger_unsettled'}


def remote_fault(runs):
    """The typed failure of the one remote job under `runs`, from the outcome it recorded."""
    outcomes = [json.loads(path.read_text()) for path in Path(runs).glob('*/outcome.json')]
    outcome = outcomes[0] if len(outcomes) == 1 else {}
    failure, execution = outcome.get('failure') or {}, outcome.get('execution') or {}
    details = {'run': str(runs), 'diagnostic': failure.get('diagnostic'), 'cleanup': outcome.get('cleanup')}
    if failure.get('code') in UNAVAILABLE:
        return CampaignFault('training_machine_unavailable', details={**details, 'reason': failure['code']})
    if ((execution.get('terminal') or {}).get('failure') or {}).get('code') == 'out_of_memory':
        return CampaignFault('student_out_of_memory', details=details)
    return CampaignFault('student_training_failed', details=details)


class Training:
    """One student's training on a schedule of `student_host.jobs.SCHEDULES`."""

    def __init__(self, journal, configuration, student, schedule='full', run=subprocess.run):
        self.journal, self.configuration = journal, configuration
        self.student, self.schedule, self.run = student, schedule, run
        self.specification = load(configuration['specification'])
        self.destination = configuration['full_training'] if schedule == 'full' else configuration.get('pilot_training', configuration['full_training'])
        if self.destination not in {'local', 'remote'} or schedule not in jobs.SCHEDULES:
            raise CampaignFault('student_training_destination_invalid')

    def directory(self, attempt):
        return self.journal.directory / 'students' / self.student / self.schedule / attempt

    def remote(self, splits, runs, resume=None):
        try:
            result = remote_training.train(self.configuration['remote'], self.student, splits, authority_key(self.journal.directory),
                                           runs, resume=resume, schedule=self.schedule)
        except JobError as exc:
            raise remote_fault(runs) from exc
        return Path(result['directory']) / 'job'

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
                                                'student': self.student, 'schedule': self.schedule})
        splits = self.dataset(inputs)
        if self.destination == 'remote':
            job = self.remote(splits, directory / 'remote')
        else:
            job = jobs.create(directory / 'jobs', self.specification, self.student, self.configuration['checkout'],
                              self.configuration['models'], splits, self.configuration['mirror'],
                              self.configuration['device'], self.schedule)
            self.train_local(job)
        return self.result(job)

    def train_local(self, job):
        environment = dict(os.environ, PW_TRAINING_SOURCE_KEY=str(authority_key(self.journal.directory)))
        with (Path(job) / 'controller.log').open('a') as log:
            result = self.run([sys.executable, '-m', 'bialy.student_host', 'train', str(job)],
                              env=environment, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            raise training_fault(job, result.returncode)

    def reconcile(self, inputs, attempt):
        directory = self.directory(attempt)
        expected = {'inputs_sha256': digest(inputs), 'configuration': self.configuration,
                    'student': self.student, 'schedule': self.schedule}
        if not (directory / 'binding.json').is_file() or json.loads((directory / 'binding.json').read_text()) != expected:
            raise CampaignFault('student_job_binding_unknown')
        self.dataset(inputs)
        if self.destination == 'remote':
            instances = list((directory / 'remote').glob('*'))
            if len(instances) != 1:
                raise CampaignFault('student_remote_job_identity_unknown')
            return self.result(self.remote(self.dataset(inputs), directory / 'remote', resume=instances[0]))
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
                'student': self.student, 'schedule': self.schedule}

    def validate(self, inputs, output):
        splits = self.dataset(inputs)
        plan = json.loads((Path(output['job']) / 'plan.json').read_text())
        if (plan['student'] != self.student or plan['schedule'] != self.schedule
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
