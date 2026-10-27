"""Campaign dependencies keep selector qualification separate from student learning."""
import json

from .execution import Executor, atomic_json, paused, resuming
from .journal import CampaignFault, StageWaiting, digest


class Campaign:
    """The full campaign. With human review, collection and evaluation each pause at a review gate."""
    def __init__(self, journal, backend, sleep, students, review=False, resume=None):
        self.journal, self.students, self.review, self.resume = journal, tuple(students), review, resume
        self.executor = Executor(journal, backend, sleep)

    def run(self):
        with self.journal.controller_lock():
            resuming(self.journal)
            try:
                return self.execute_stages()
            except StageWaiting as waiting:
                return {'schema': 'pw-training-campaign-result/1', 'status': 'waiting',
                        'waiting': paused(self.journal, waiting, self.resume)}

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

    def dataset(self, collection_name, inputs):
        collection = self.stage(collection_name, inputs)
        if collection['status'] != 'passed':
            return collection, collection
        suffix = collection_name.removeprefix('collection_')
        review = self.gate('review_collection_' + suffix, {'collection': collection['output']})
        if review['status'] != 'passed':
            return collection, review
        return collection, self.stage('dataset_' + suffix, {**inputs, 'collection': collection['output'],
                                                            'review': review['output']})

    def execute_stages(self):
        host = self.stage('host_qualification', {})
        if host['status'] != 'passed':
            return self.finish('dependency_unqualified')
        reference = host['output'].get('conservative_reference')
        if not reference or reference.get('qualified') is not True:
            return self.finish('conservative_reference_unqualified')
        common = {'host': host['output']}
        teachers = self.stage('teacher_baseline', common)
        selector = self.stage('selector_training', common)
        selected = reference
        selector_status = 'conservative'
        if selector['status'] == 'passed':
            qualification = self.stage('selector_qualification', {'host': host['output'], 'selector': selector['output']})
            if qualification['status'] == 'passed' and qualification['output'].get('qualified') is True:
                selected = qualification['output']['configuration']
                selector_status = 'retrained'
        if teachers['status'] != 'passed':
            return self.finish('insufficient_qualified_data', selector_status)
        collection, pilot_data = self.dataset('collection_pilot', {**common, 'teachers': teachers['output'], 'selector': selected})
        if pilot_data['status'] != 'passed':
            return self.finish('insufficient_qualified_data', selector_status)
        plan = self.stage('evaluation_plan', {**common, 'measurements': collection['output']})
        if plan['status'] != 'passed':
            return self.finish('evaluation_plan_unavailable', selector_status)
        frozen = {**common, 'plan': plan['output'], 'selector': selected, 'teachers': teachers['output']}
        pilots = {}
        for student in self.students:
            pilots[student] = self.stage('pilot_' + student, {**frozen, 'student': student, 'dataset': pilot_data['output']})
        _, data = self.dataset('collection_full', {**frozen, 'pilot_measurements': {
            name: result['output'] for name, result in pilots.items() if result['status'] == 'passed'}})
        outcomes, evaluated = {}, {}
        for student in self.students:
            base = self.stage('evaluate_base_' + student, {**frozen, 'student': student})
            if data['status'] != 'passed' or pilots[student]['status'] != 'passed':
                outcomes[student] = {'status': 'execution_incomplete', 'base': base}
                continue
            trained = self.stage('train_' + student, {**frozen, 'student': student, 'dataset': data['output'],
                                                      'pilot': pilots[student]['output']})
            if trained['status'] != 'passed':
                outcomes[student] = {'status': 'execution_incomplete', 'base': base, 'training': trained}
                continue
            exported = self.stage('export_' + student, {**frozen, 'student': student, 'training': trained['output']})
            if exported['status'] != 'passed':
                outcomes[student] = {'status': 'export_incomplete', 'base': base, 'training': trained, 'export': exported}
                continue
            evaluation = self.stage('evaluate_trained_' + student, {**frozen, 'student': student, 'artifact': exported['output']})
            if base['status'] != 'passed' or evaluation['status'] != 'passed':
                outcomes[student] = {'status': 'evaluation_incomplete', 'base': base, 'evaluation': evaluation,
                                     'artifact': exported['output']}
                continue
            evaluated[student] = {'base': base['output'], 'trained': evaluation['output'], 'artifact': exported['output']}
        if evaluated:
            review = self.gate('review_evaluation', {'evaluations': {student: {'base': item['base'], 'trained': item['trained']}
                                                                      for student, item in evaluated.items()}})
            for student, item in evaluated.items():
                if review['status'] != 'passed':
                    outcomes[student] = {'status': 'evaluation_incomplete', 'review': review, 'artifact': item['artifact']}
                    continue
                grade = self.stage('compare_' + student, {**frozen, 'student': student, 'base': item['base'],
                                                          'trained': item['trained'], 'artifact': item['artifact'],
                                                          'review': review['output']})
                outcomes[student] = {'status': grade['output']['status'] if grade['status'] == 'passed' else 'evaluation_incomplete',
                                     'grade': grade, 'artifact': item['artifact']}
        status = 'qualified' if all(outcomes[student]['status'] == 'pass' for student in self.students) else 'no_go'
        return self.finish(status, selector_status, {student: outcomes[student] for student in self.students})

    def finish(self, status, selector=None, students=None):
        snapshot = self.journal.snapshot()
        result = {'schema': 'pw-training-campaign-result/1', 'status': status, 'selector': selector,
                  'students': students or {name: {'status': 'not_executed'} for name in self.students},
                  'journal_sha256': digest(snapshot), 'journal': snapshot}
        path = self.journal.directory / 'result.json'
        if path.exists():
            if json.loads(path.read_text()) != result:
                raise CampaignFault('campaign_terminal_result_changed')
        else:
            atomic_json(path, result)
        return result
