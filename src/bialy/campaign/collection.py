"""Measure teacher yield and collect fresh episodes from the frozen task bank."""
import json
import time
from collections import Counter

from ..provenance import file_sha256
from .execution import Executor
from .grading import reviewed
from .journal import CampaignFault, digest
from .selector import stratum


class Collection:
    """Drive each task until one attempt is admitted, recording every episode for assembly.

    With human review, episodes only the reviewers disagree on are exported too, so a
    person's pass can admit them, and every exported episode is offered to the queue.
    """
    def __init__(self, journal, lineage, driver, export, students, minimum, groups_per_focus, sleep=time.sleep, review=None):
        self.journal, self.lineage, self.students = journal, lineage, tuple(students)
        self.executor = Executor(journal, driver, sleep)
        self.export, self.minimum, self.groups_per_focus, self.review = export, minimum, groups_per_focus, review

    def execute(self, inputs, attempt):
        tasks = [(json.loads(row[0]), row[1]) for row in self.journal.db.execute(
            "SELECT spec, partition FROM tasks WHERE partition IN ('train','validation','system_test') AND parent IS NULL ORDER BY id")]
        teachers = {teacher['id']: teacher for teacher in inputs['teachers']['teachers']}
        minimum = self.minimum
        if (set(minimum) != {'train', 'validation', 'system_test'}
                or any(type(count) is not int or count < 1 for count in minimum.values())):
            raise CampaignFault('collection_episode_requirements_invalid')
        counts, covered = Counter(), {partition: {focus: set() for focus in self.lineage.focuses} for partition in minimum}
        episodes, evidence = [], []
        for task, partition in tasks:
            if counts[partition] >= minimum[partition] and all(
                    len(covered[partition][focus]) >= self.groups_per_focus for focus in self.lineage.focuses):
                continue
            # The policy's first route for the focus collects it.
            teacher = teachers[inputs['teachers']['focuses'][task['focus']][0]['teacher_id']]
            for repetition in range(3):
                drawn = stratum(inputs['selector'], task['id'], repetition)
                selector = inputs['host']['conservative_reference'] if drawn == 'planned_fallback' else inputs['selector']
                trial = {'host': inputs['host'], 'selector': selector, 'task': task,
                         'repetition': repetition, 'actor': teacher}
                # Pilot and full collection share completed attempts from the same task bank.
                name = 'collection_trial_' + digest({'task': task['id'], 'teacher': teacher['id'], 'repetition': repetition,
                                                    'selector': selector})
                result = self.executor.run(name, trial)
                evidence.append({'stage': name, 'sha256': digest(result)})
                if result['status'] != 'passed':
                    raise CampaignFault('collection_episode_incomplete', details={'stage': name, 'fault': result['fault']})
                episode = {'stage': name, 'task_id': task['id'], 'partition': partition, 'focus': task['focus'],
                           'repetition': repetition, 'stratum': drawn, 'episode_sha256': result['output']['verdict']['episode_sha256'],
                           'admitted': False, 'exports': None}
                episodes.append(episode)
                reviewable = self.review is not None and reviewed(result['output']) == 'split'
                if not result['output']['success'] and not reviewable:
                    continue
                exported = self.export(trial, result['output'])
                episode['exports'] = {student: {'manifest': exported[student]['manifest'],
                                                'manifest_sha256': file_sha256(exported[student]['manifest']),
                                                'rows': exported[student]['rows']} for student in self.students}
                if not all(exported[student]['rows'] > 0 for student in self.students):
                    continue
                if self.review is not None:
                    self.review.offer_episode(episode, result['output'], task, partition)
                if not result['output']['success']:
                    continue
                episode['admitted'] = True
                counts[partition] += 1
                covered[partition][task['focus']].add(task.get('group', task['id']))
                break
        return {'episodes': episodes, 'minimum': minimum, 'evidence': evidence, 'lineage_sha256': self.lineage.freeze()}

    def reconcile(self, inputs, attempt):
        return self.execute(inputs, attempt)

    def validate(self, inputs, output):
        if output['lineage_sha256'] != self.lineage.freeze():
            raise CampaignFault('collection_lineage_changed')
        for item in output['evidence']:
            if digest(self.journal.stage(item['stage'])) != item['sha256']:
                raise CampaignFault('collection_episode_changed')
        for episode in output['episodes']:
            for exported in (episode['exports'] or {}).values():
                if file_sha256(exported['manifest']) != exported['manifest_sha256']:
                    raise CampaignFault('collection_export_changed')
