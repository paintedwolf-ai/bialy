"""Execute every registered holdout attempt and retain failed attempts."""
import json
import time

from . import evaluation_policy
from .execution import Executor, atomic_json
from .grading import behavioral_result, effective_attempts, reviewed
from .journal import CampaignFault, digest
from .selector import plan_record


class EvaluationPlan:
    """The locked tasks and attempts each tier is evaluated on, with the behavioral policy
    frozen beside them before any student result exists."""

    def __init__(self, journal, lineage, policy=None, collection_episodes=None, judged=True, tasks=None):
        self.journal, self.lineage = journal, lineage
        self.policy = policy or evaluation_policy.load()
        self.collection_episodes, self.judged, self.tasks = collection_episodes, judged, tasks

    def build(self, inputs):
        policy = self.policy
        tasks = [dict(row) for row in self.journal.db.execute(
            "SELECT id,focus,spec_hash,partition FROM tasks WHERE partition IN (?,?) AND parent IS NULL ORDER BY partition,id",
            (policy['decides'], policy['diagnostic']))]
        deciding = [task for task in tasks if task['partition'] == policy['decides']]
        if self.tasks is not None:
            tasks = deciding[:self.tasks]
        elif any(sum(task['focus'] == focus for task in deciding) < 3 for focus in self.lineage.focuses):
            raise CampaignFault('insufficient_independent_evaluation_tasks')
        if self.judged and self.tasks is not None:
            raise CampaignFault('evaluation_plan_invalid')
        collection = self.collection_episodes or {row[0]: row[1] for row in self.journal.db.execute(
            "SELECT partition,COUNT(*) FROM tasks WHERE parent IS NULL AND partition!='development' GROUP BY partition")}
        if (set(collection) != {'train', 'validation', 'system_test'}
                or any(type(count) is not int or count < 1 for count in collection.values())):
            raise CampaignFault('collection_episode_requirements_invalid')
        design = policy['design']
        return {'collection_episodes': collection, 'tasks': tasks, 'focuses': list(self.lineage.focuses),
                'attempts_per_task': design['attempts_per_task'], 'bootstrap_samples': design['bootstrap_samples'],
                'bootstrap_seed': design['bootstrap_seed'], 'lineage_sha256': self.lineage.freeze(),
                'measurements_sha256': digest(inputs['measurements']), 'selector': plan_record(inputs['selector']),
                'policy_sha256': digest(policy), 'judged': self.judged,
                'planned_attempts_per_arm': len(tasks) * design['attempts_per_task']}

    def execute(self, inputs, attempt):
        result = self.build(inputs)
        evaluation_policy.freeze(self.journal.directory, self.policy)
        atomic_json(self.journal.directory / 'evaluation-plan.json', result)
        return result

    def reconcile(self, inputs, attempt):
        path = self.journal.directory / 'evaluation-plan.json'
        if not path.exists():
            return self.execute(inputs, attempt)
        return json.loads(path.read_text())

    def validate(self, inputs, output):
        if self.build(inputs) != output:
            raise CampaignFault('evaluation_plan_changed')
        evaluation_policy.frozen(self.journal.directory, output)


class Evaluation:
    def __init__(self, journal, lineage, driver, student, arm, actor, sleep=time.sleep):
        self.journal, self.lineage, self.student, self.arm = journal, lineage, student, arm
        self.executor = Executor(journal, driver, sleep)
        self.actor = actor

    def execute(self, inputs, attempt):
        plan = inputs['plan']
        if plan['lineage_sha256'] != self.lineage.freeze():
            raise CampaignFault('evaluation_lineage_changed')
        evaluation_policy.frozen(self.journal.directory, plan)
        attempts = []
        for task in plan['tasks']:
            registered = self.lineage.admit(task['id'], task['partition'], task['spec_hash'])
            for repetition in range(plan['attempts_per_task']):
                trial = {**inputs, 'task': json.loads(registered['spec']), 'repetition': repetition,
                         'arm': self.arm, 'student': self.student, 'actor': self.actor}
                name = 'trial_' + digest({'student': self.student, 'arm': self.arm, 'task': task['id'], 'repetition': repetition})
                result = self.executor.run(name, trial)
                record = {'task_id': task['id'], 'attempt': repetition, 'arm': self.arm, 'student': self.student,
                          'plan_sha256': digest(plan), 'trial': name, 'stage_status': result['status']}
                if result['status'] == 'passed':
                    evidence = result['output']
                    record.update({key: evidence[key] for key in ('success', 'human_interventions', 'unauthorized_effects',
                                                                  'unsupported_completion_claims', 'effective_thinking')})
                    record.update(evidence_sha256=digest(evidence), episode_sha256=evidence['verdict']['episode_sha256'],
                                  review=reviewed(evidence))
                else:
                    record.update(success=False, human_interventions=None, unauthorized_effects=None,
                                  unsupported_completion_claims=None, effective_thinking=None, fault=result['fault'],
                                  episode_sha256=None, review='fail')
                attempts.append(record)
        return {'attempts': attempts, 'plan_sha256': digest(plan)}

    def reconcile(self, inputs, attempt):
        return self.execute(inputs, attempt)

    def validate(self, inputs, output):
        plan = inputs['plan']
        expected = {(task['id'], repetition) for task in plan['tasks'] for repetition in range(plan['attempts_per_task'])}
        records = output['attempts']
        if (output['plan_sha256'] != digest(plan) or len(records) != len(expected)
                or {(row['task_id'], row['attempt']) for row in records} != expected):
            raise CampaignFault('evaluation_attempts_changed')
        for row in records:
            result = self.journal.stage(row['trial'])
            if (row['student'] != self.student or row['arm'] != self.arm or row['plan_sha256'] != digest(plan)
                    or result is None or result['status'] != row['stage_status']):
                raise CampaignFault('evaluation_trial_changed')
            if result['status'] == 'passed':
                keys = ('success', 'human_interventions', 'unauthorized_effects', 'unsupported_completion_claims', 'effective_thinking')
                if (digest(result['output']) != row['evidence_sha256'] or any(row[key] != result['output'][key] for key in keys)
                        or row['episode_sha256'] != result['output']['verdict']['episode_sha256']
                        or row['review'] != reviewed(result['output'])):
                    raise CampaignFault('evaluation_evidence_changed')
            elif row.get('fault') != result['fault'] or row['success'] is not False:
                raise CampaignFault('evaluation_failure_changed')


class Comparison:
    """Grade base against trained under the frozen policy, applying any human evaluation decisions."""
    def __init__(self, journal):
        self.journal = journal

    def execute(self, inputs, attempt):
        policy = evaluation_policy.frozen(self.journal.directory, inputs['plan'])
        decisions = inputs['review']['decisions'] if inputs['review'] else {}
        attempts = effective_attempts(inputs['base']['attempts'] + inputs['trained']['attempts'], decisions)
        return behavioral_result(inputs['plan'], policy, attempts, inputs['student'])

    def reconcile(self, inputs, attempt):
        return self.execute(inputs, attempt)

    def validate(self, inputs, output):
        if self.execute(inputs, None) != output:
            raise CampaignFault('evaluation_comparison_changed')
