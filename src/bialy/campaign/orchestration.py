"""One campaign graph for every profile: the selector segment, the host adoption boundary,
and the student segment. A profile sets which segments run and their workload; no
profile has stages of its own."""
import json

from .execution import Executor, atomic_json, paused, resuming
from .journal import CampaignFault, StageWaiting, digest

BOUNDARY = 'awaiting_host_adoption'


def tier(**stages):
    """One student's outcome: its artifact, execution and behavioral statuses kept apart, so one
    tier's failure never hides another's usable artifact."""
    return {'artifact_status': 'absent', 'artifact': None, 'execution_status': 'not_executed',
            'behavioral_status': 'not_measured', **stages}


def data_status(result):
    if result['status'] == 'passed':
        return result['output'].get('data_status', 'data_ready') if isinstance(result['output'], dict) else 'data_ready'
    fault = result.get('fault') or {}
    return 'data_incomplete' if fault.get('code') == 'data_incomplete' else 'data_unavailable'


class Campaign:
    """With human review, collection and evaluation each pause at a review gate."""

    def __init__(self, journal, backend, sleep, profile, students, review=False, resume=None):
        self.journal, self.profile, self.students = journal, profile, tuple(students)
        self.review, self.resume = review, resume
        self.executor = Executor(journal, backend, sleep)

    def run(self):
        with self.journal.controller_lock():
            resuming(self.journal)
            try:
                return self.execute_stages()
            except StageWaiting as waiting:
                record = paused(self.journal, waiting, self.resume)
                status = BOUNDARY if record['reason'] == BOUNDARY else 'waiting'
                return {'schema': 'pw-training-campaign-result/2', 'status': status, 'waiting': record}

    def stage(self, name, inputs):
        result = self.executor.run(name, inputs)
        atomic_json(self.journal.directory / 'status.json', self.journal.snapshot())
        if result['status'] == 'waiting':
            raise StageWaiting(result['output']['reason'], {**result['output'], 'stage': name})
        return result

    def gate(self, name, inputs):
        """The review stage's result; without human review, a passed result that carries no decisions."""
        if not self.review:
            return {'status': 'passed', 'output': None}
        return self.stage(name, inputs)

    def qualified_host(self, name, inputs):
        host = self.stage(name, inputs)
        if host['status'] != 'passed':
            return None
        reference = host['output'].get('conservative_reference')
        if not reference or reference.get('qualified') is not True:
            return None
        return host['output']

    def selection(self, common):
        """The selector the student segment collects under, and how it was chosen."""
        mode = self.profile['selector']
        reference = common['host']['conservative_reference']
        if mode == 'engine_disabled':
            return reference, 'engine_disabled'
        if mode == 'supplied_or_engine_disabled':
            supplied = self.stage('selector_supplied', common)
            if supplied['status'] == 'passed' and supplied['output'].get('qualified') is True:
                return supplied['output']['configuration'], 'supplied'
            return reference, 'engine_disabled'
        raise CampaignFault('campaign_selector_mode_unknown', details={'mode': mode})

    def selector_segment(self, common):
        """Head rows, P0, and the adopted host; None when P0 did not qualify."""
        rows = self.stage('head_rows', common)
        if rows['status'] != 'passed':
            return None
        trained = self.stage('selector_training', {**common, 'rows': rows['output']})
        if trained['status'] != 'passed':
            return None
        qualification = self.stage('selector_qualification', {**common, 'selector': trained['output']})
        if qualification['status'] != 'passed' or qualification['output'].get('qualified') is not True:
            return None
        adoption = self.stage('host_adoption', {'policy': trained['output']['policy']})
        if adoption['status'] != 'passed':
            return None
        return qualification['output']['configuration'], adoption['output']

    def execute_stages(self):
        host = self.qualified_host('host_qualification', {})
        if host is None:
            return self.finish('dependency_unqualified')
        common = {'host': host}
        reviewers = self.stage('reviewer_qualification', common)
        teachers = self.stage('teacher_qualification', common)
        if reviewers['status'] != 'passed' or teachers['status'] != 'passed':
            return self.finish('teachers_or_reviewers_unqualified')
        if 'selector' in self.profile['segments']:
            segment = self.selector_segment(common)
            if segment is None:
                return self.finish('selector_unqualified', 'unqualified')
            selected, adoption = segment
            host = self.qualified_host('student_host_qualification', {'adoption': adoption})
            if host is None:
                return self.finish('dependency_unqualified', 'p0')
            common, selector_status = {'host': host}, 'p0'
        else:
            selected, selector_status = self.selection(common)
        return self.student_segment(common, teachers['output'], selected, selector_status)

    def student_segment(self, common, teachers, selected, selector_status):
        collection = self.stage('collection', {**common, 'teachers': teachers, 'selector': selected})
        if collection['status'] != 'passed':
            return self.finish('insufficient_qualified_data', selector_status, data='data_unavailable')
        review = self.gate('review_collection', {'collection': collection['output']})
        if review['status'] != 'passed':
            return self.finish('insufficient_qualified_data', selector_status, data='data_unavailable')
        data = self.stage('dataset', {**common, 'selector': selected, 'collection': collection['output'], 'review': review['output']})
        if data['status'] != 'passed':
            return self.finish('insufficient_qualified_data', selector_status, data=data_status(data))
        plan = self.stage('evaluation_plan', {**common, 'selector': selected, 'measurements': collection['output']})
        if plan['status'] != 'passed':
            return self.finish('evaluation_plan_unavailable', selector_status, data=data_status(data))
        frozen = {**common, 'plan': plan['output'], 'selector': selected}
        outcomes, evaluated = {}, {}
        for student in self.students:
            outcome = outcomes[student] = tier()
            baseline = self.stage('baseline_' + student, {**frozen, 'student': student})
            base = baseline if baseline['status'] != 'passed' else self.stage(
                'evaluate_base_' + student, {**frozen, 'student': student, 'artifact': baseline['output']})
            outcome['base'] = base
            trained = self.stage('train_' + student, {**frozen, 'student': student, 'dataset': data['output']})
            outcome['training'] = trained
            if trained['status'] != 'passed':
                outcome['execution_status'] = 'training_incomplete'
                continue
            exported = self.stage('export_' + student, {**frozen, 'student': student, 'training': trained['output']})
            outcome['export'] = exported
            if exported['status'] != 'passed':
                outcome['execution_status'] = 'export_incomplete'
                continue
            outcome.update(artifact_status='exported', artifact=exported['output'])
            evaluation = self.stage('evaluate_trained_' + student, {**frozen, 'student': student, 'artifact': exported['output']})
            outcome['evaluation'] = evaluation
            if base['status'] != 'passed' or evaluation['status'] != 'passed':
                outcome['execution_status'] = 'evaluation_incomplete'
                continue
            if not self.profile['judged']:
                outcome['execution_status'] = 'complete'
                continue
            evaluated[student] = {'base': base['output'], 'trained': evaluation['output'], 'artifact': exported['output']}
        if evaluated:
            self.compare(frozen, evaluated, outcomes)
        cleanup = self.stage('cleanup', {'students': sorted(self.students)})
        if not self.profile['judged']:
            complete = all(outcome['execution_status'] == 'complete' for outcome in outcomes.values())
            status = 'mechanical_pipeline_complete' if complete else 'execution_incomplete'
        else:
            status = 'qualified' if all(outcomes[student]['behavioral_status'] == 'pass' for student in self.students) else 'no_go'
        return self.finish(status, selector_status, outcomes, data_status(data), cleanup)

    def compare(self, frozen, evaluated, outcomes):
        review = self.gate('review_evaluation', {'evaluations': {student: {'base': item['base'], 'trained': item['trained']}
                                                                  for student, item in evaluated.items()}})
        for student, item in evaluated.items():
            outcome = outcomes[student]
            if review['status'] != 'passed':
                outcome.update(execution_status='evaluation_incomplete', review=review)
                continue
            grade = self.stage('compare_' + student, {**frozen, 'student': student, 'base': item['base'],
                                                      'trained': item['trained'], 'artifact': item['artifact'],
                                                      'review': review['output']})
            outcome['grade'] = grade
            if grade['status'] != 'passed':
                outcome['execution_status'] = 'evaluation_incomplete'
                continue
            outcome.update(execution_status='complete', behavioral_status=grade['output']['status'])

    def finish(self, status, selector=None, students=None, data='not_reached', cleanup=None):
        snapshot = self.journal.snapshot()
        tiers = students or {name: tier() for name in self.students}
        complete = all(item['execution_status'] == 'complete' for item in tiers.values())
        result = {'schema': 'pw-training-campaign-result/2', 'status': status, 'profile': self.profile['name'],
                  'dry_run': self.profile['name'] == 'dry-run', 'selector': selector,
                  'fallback': selector == 'engine_disabled', 'data_status': data,
                  'execution_status': 'complete' if complete else 'incomplete', 'students': tiers,
                  'promotion_qualified': self.profile['name'] == 'full' and status == 'qualified',
                  'cleanup': cleanup, 'journal_sha256': digest(snapshot), 'journal': snapshot}
        path = self.journal.directory / 'result.json'
        if path.exists():
            if json.loads(path.read_text()) != result:
                raise CampaignFault('campaign_terminal_result_changed')
        else:
            atomic_json(path, result)
        return result
