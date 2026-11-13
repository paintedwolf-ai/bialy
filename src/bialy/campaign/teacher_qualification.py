"""Per-focus teacher routes, measured on frozen development tasks before any student result.

Every configured generator attempts each development task of each focus. An episode the
driver could not finish is an infrastructure fault, never a semantic failure: it stops the
stage typed instead of spending further paid episodes against a broken environment. A focus's route order ranks the teachers with a verified success by the lower
bound of their success rate; a focus with none fails the campaign as `teachers_unqualified`.
The policy is written once, as `teacher-policy.json`, and never after a student result.
"""
import json
import time
from collections import Counter

from .execution import Executor, atomic_json
from .grading import wilson
from .journal import CampaignFault, digest

SCHEMA = 'pw-teacher-policy/1'
STUDENT_RESULTS = "SELECT 1 FROM stages WHERE status='passed' AND (name LIKE 'evaluate_%' OR name LIKE 'train_%' OR name LIKE 'pilot_%')"


def policy(measurements, focuses):
    keys = [(row['teacher_id'], row['focus']) for row in measurements]
    if len(keys) != len(set(keys)):
        raise CampaignFault('duplicate_teacher_measurement')
    for row in measurements:
        if (row['focus'] not in focuses or any(type(row[key]) is not int for key in ('qualified', 'attempts'))
                or type(row['capture_qualified']) is not bool):
            raise CampaignFault('invalid_teacher_measurement')
        wilson(row['qualified'], row['attempts'])
    routes, unqualified = {}, []
    for focus in focuses:
        verified = [row for row in measurements if row['focus'] == focus and row['qualified'] > 0 and row['capture_qualified']]
        if not verified:
            unqualified.append(focus)
            continue
        ordered = sorted(verified, key=lambda row: (-wilson(row['qualified'], row['attempts'])[0], row['teacher_id']))
        routes[focus] = [{'teacher_id': row['teacher_id'], 'floor': wilson(row['qualified'], row['attempts'])[0],
                          'qualified': row['qualified'], 'attempts': row['attempts']} for row in ordered]
    if unqualified:
        raise CampaignFault('teachers_unqualified', details={'focuses': unqualified})
    return {'schema': SCHEMA, 'measurements_sha256': digest(measurements), 'focuses': routes}


class TeacherQualification:
    def __init__(self, journal, lineage, driver, teachers, sleep=time.sleep, minimum_tasks=4, repetitions=2):
        self.journal, self.lineage, self.teachers = journal, lineage, teachers
        self.executor = Executor(journal, driver, sleep)
        if type(minimum_tasks) is not int or minimum_tasks < 1 or type(repetitions) is not int or repetitions < 1:
            raise CampaignFault('teacher_qualification_sampling_invalid')
        self.minimum_tasks, self.repetitions = minimum_tasks, repetitions
        self.path = journal.directory / 'teacher-policy.json'

    def execute(self, inputs, attempt):
        if not self.path.exists() and self.journal.db.execute(STUDENT_RESULTS).fetchone():
            raise CampaignFault('teacher_policy_after_student_results')
        tasks = [json.loads(row[0]) for row in self.journal.db.execute(
            "SELECT spec FROM tasks WHERE partition='development' AND parent IS NULL ORDER BY id")]
        if any(sum(task['focus'] == focus for task in tasks) < self.minimum_tasks for focus in self.lineage.focuses):
            raise CampaignFault('teacher_development_bank_incomplete')
        measurements, evidence = [], []
        for teacher in self.teachers:
            attempts, successes = Counter(), Counter()
            for task in tasks:
                for repetition in range(self.repetitions):
                    trial = {**inputs, 'task': task, 'repetition': repetition, 'actor': teacher,
                             'selector': inputs['host']['conservative_reference']}
                    name = 'teacher_trial_' + digest({'teacher': teacher['id'], 'task': task['id'], 'repetition': repetition})
                    result = self.executor.run(name, trial)
                    evidence.append({'stage': name, 'sha256': digest(result)})
                    if result['status'] != 'passed':
                        raise CampaignFault('teacher_episode_incomplete', details={'stage': name, 'fault': result['fault']})
                    attempts[task['focus']] += 1
                    successes[task['focus']] += result['output']['success'] is True
            for focus in self.lineage.focuses:
                measurements.append({'teacher_id': teacher['id'], 'focus': focus, 'attempts': attempts[focus],
                                     'qualified': successes[focus],
                                     'capture_qualified': inputs['host']['teacher_capture_qualified'].get(teacher['id']) is True})
        result = policy(measurements, self.lineage.focuses)
        document = {**result, 'lineage_sha256': self.lineage.freeze()}
        if self.path.exists():
            if json.loads(self.path.read_text()) != document:
                raise CampaignFault('teacher_policy_changed')
        else:
            atomic_json(self.path, document)
        return {**document, 'measurements': measurements, 'evidence': evidence, 'teachers': self.teachers}

    def reconcile(self, inputs, attempt):
        return self.execute(inputs, attempt)

    def validate(self, inputs, output):
        recomputed = policy(output['measurements'], self.lineage.focuses)
        if (output['focuses'] != recomputed['focuses'] or output['lineage_sha256'] != self.lineage.freeze()
                or json.loads(self.path.read_text()) != {key: output[key] for key in ('schema', 'measurements_sha256', 'focuses', 'lineage_sha256')}
                or any(digest(self.journal.stage(item['stage'])) != item['sha256'] for item in output['evidence'])):
            raise CampaignFault('teacher_policy_changed')
