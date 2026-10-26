"""Measure teacher yield and collect fresh episodes from the frozen task bank."""
import json
import time
from collections import Counter
from pathlib import Path

from ..provenance import file_sha256
from .execution import Executor
from .grading import teacher_routes
from .journal import CampaignFault, digest
from .lineage import FAMILIES


class TeacherBaseline:
    def __init__(self, journal, lineage, driver, teachers, sleep=time.sleep, minimum_tasks=4, repetitions=2):
        self.journal, self.lineage, self.teachers = journal, lineage, teachers
        self.executor = Executor(journal, driver, sleep)
        if type(minimum_tasks) is not int or minimum_tasks < 1 or type(repetitions) is not int or repetitions < 1:
            raise CampaignFault('teacher_baseline_sampling_invalid')
        self.minimum_tasks, self.repetitions = minimum_tasks, repetitions

    def execute(self, inputs, attempt):
        tasks = [json.loads(row[0]) for row in self.journal.db.execute(
            "SELECT spec FROM tasks WHERE partition='development' AND parent IS NULL ORDER BY id")]
        if any(sum(task['family'] == family for task in tasks) < self.minimum_tasks for family in FAMILIES):
            raise CampaignFault('teacher_development_bank_incomplete')
        measurements, evidence = [], []
        for teacher in self.teachers:
            counts, successes = Counter(), Counter()
            for task in tasks:
                for repetition in range(self.repetitions):
                    trial = {**inputs, 'task': task, 'repetition': repetition, 'actor': teacher,
                             'selector': inputs['host']['conservative_reference']}
                    name = 'teacher_trial_' + digest({'teacher': teacher['id'], 'task': task['id'], 'repetition': repetition})
                    result = self.executor.run(name, trial)
                    counts[task['family']] += 1
                    successes[task['family']] += result['status'] == 'passed' and result['output']['success'] is True
                    evidence.append({'stage': name, 'sha256': digest(result)})
            for family in FAMILIES:
                measurements.append({'teacher_id': teacher['id'], 'family': family, 'attempts': counts[family],
                                     'qualified': successes[family],
                                     'capture_qualified': inputs['host']['teacher_capture_qualified'].get(teacher['id']) is True})
        return {**teacher_routes(measurements), 'measurements': measurements, 'evidence': evidence,
                'teachers': self.teachers, 'lineage_sha256': self.lineage.freeze()}

    def reconcile(self, inputs, attempt):
        return self.execute(inputs, attempt)

    def validate(self, inputs, output):
        routes = teacher_routes(output['measurements'])
        if (output['families'] != routes['families'] or output['lineage_sha256'] != self.lineage.freeze()
                or any(digest(self.journal.stage(item['stage'])) != item['sha256'] for item in output['evidence'])):
            raise CampaignFault('teacher_baseline_changed')


class Collection:
    def __init__(self, journal, lineage, driver, export, assemble, pilot, sleep=time.sleep, minimum=None):
        self.journal, self.lineage = journal, lineage
        self.executor = Executor(journal, driver, sleep)
        self.export, self.assemble, self.pilot = export, assemble, pilot
        self.minimum = minimum

    def execute(self, inputs, attempt):
        tasks = [json.loads(row[0]) for row in self.journal.db.execute(
            "SELECT spec FROM tasks WHERE partition IN ('train','validation','system_test') AND parent IS NULL ORDER BY id")]
        teachers = {teacher['id']: teacher for teacher in inputs['teachers']['teachers']}
        manifests = {'e4b': [], '12b': []}
        counts, retained = Counter(), Counter()
        minimum = self.minimum or (
            {'train': 32, 'validation': 16, 'system_test': 24} if self.pilot
            else inputs['plan']['collection_episodes'])
        if (set(minimum) != {'train', 'validation', 'system_test'}
                or any(type(count) is not int or count < len(FAMILIES) for count in minimum.values())):
            raise CampaignFault('collection_episode_requirements_invalid')
        families = {partition: set() for partition in minimum}
        evidence = []
        for task in tasks:
            partition = task['partition']
            if counts[partition] >= minimum[partition] and families[partition] == set(FAMILIES):
                continue
            teacher = teachers[inputs['teachers']['families'][task['family']]['teacher_id']]
            for repetition in range(3):
                trial = {**inputs, 'task': task, 'repetition': repetition, 'actor': teacher}
                # Pilot and full collection share completed attempts from the same task bank.
                name = 'collection_trial_' + digest({'task': task['id'], 'teacher': teacher['id'], 'repetition': repetition,
                                                    'selector': inputs['selector']})
                trial = {key: trial[key] for key in ('host', 'selector', 'task', 'repetition', 'actor')}
                result = self.executor.run(name, trial)
                evidence.append({'stage': name, 'sha256': digest(result)})
                if result['status'] != 'passed' or not result['output']['success']:
                    continue
                exported = self.export(trial, result['output'])
                if not all(exported[student]['rows'] > 0 for student in manifests):
                    continue
                for student in manifests:
                    manifests[student].append(exported[student]['manifest'])
                    retained[student] += exported[student]['rows']
                counts[partition] += 1
                families[partition].add(task['family'])
                break
        if any(counts[key] < minimum[key] or families[key] != set(FAMILIES) for key in minimum):
            raise CampaignFault('insufficient_qualified_data', details={'episodes': dict(counts), 'required': minimum})
        result = self.assemble(manifests, self.journal.directory / 'datasets' / attempt, minimum)
        return {**result, 'episode_counts': dict(counts), 'retained_rows': dict(retained), 'evidence': evidence}

    def reconcile(self, inputs, attempt):
        return self.execute(inputs, attempt)

    def validate(self, inputs, output):
        if output['lineage_sha256'] != self.lineage.freeze():
            raise CampaignFault('collection_lineage_changed')
        for student in ('e4b', '12b'):
            dataset = output['students'][student]
            if any(file_sha256(Path(dataset['directory']) / name) != checksum for name, checksum in dataset['files'].items()):
                raise CampaignFault('collection_dataset_changed')
        for item in output['evidence']:
            if digest(self.journal.stage(item['stage'])) != item['sha256']:
                raise CampaignFault('collection_episode_changed')
