"""Run a small, non-promoting training campaign through real host episodes."""
import argparse
import json
import time
from pathlib import Path

from .execution import Executor, ReceiptBackend, atomic_json
from .journal import CampaignFault, Journal, digest


class Pilot:
    def __init__(self, journal, handlers, plan, students):
        self.journal, self.plan, self.students = journal, plan, students
        self.executor = Executor(journal, ReceiptBackend(journal.directory / 'receipts', handlers), time.sleep)

    def stage(self, name, inputs):
        result = self.executor.run(name, inputs)
        atomic_json(self.journal.directory / 'status.json', self.journal.snapshot())
        print(json.dumps({'stage': name, 'status': result['status'], 'fault': result.get('fault')}), flush=True)
        if result['status'] != 'passed':
            raise CampaignFault('pilot_stage_failed', details={'stage': name, 'fault': result.get('fault')})
        return result['output']

    def run(self):
        outcomes = {}
        with self.journal.controller_lock():
            try:
                host = self.stage('host_qualification', {})
                self.stage('reviewer_qualification', {'host': host})
                teachers = self.stage('teacher_baseline', {'host': host})
                common = {'host': host, 'selector': host['conservative_reference'], 'teachers': teachers}
                data = self.stage('collection_pilot', common)
                for student in self.students:
                    inputs = {**common, 'student': student, 'plan': self.plan}
                    baseline = self.stage('baseline_' + student, inputs)
                    base = self.stage('evaluate_base_' + student, {**inputs, 'artifact': baseline})
                    trained = self.stage('pilot_' + student, {**inputs, 'dataset': data})
                    artifact = self.stage('export_' + student, {**inputs, 'training': trained})
                    evaluated = self.stage('evaluate_trained_' + student, {**inputs, 'artifact': artifact})
                    outcomes[student] = {'artifact': artifact, 'base': base, 'trained': evaluated,
                                         'base_successes': sum(row['success'] for row in base['attempts']),
                                         'trained_successes': sum(row['success'] for row in evaluated['attempts'])}
                complete = all(row['stage_status'] == 'passed' and row['effective_thinking'] is False
                               for outcome in outcomes.values() for arm in ('base', 'trained') for row in outcome[arm]['attempts'])
                result = {'status': 'mechanical_pipeline_complete' if complete else 'evaluation_incomplete', 'students': outcomes}
            except CampaignFault as exc:
                result = {'status': 'incomplete', 'fault': exc.record(), 'students': outcomes}
            result.update(schema='pw-training-pilot/1', promotion_qualified=False, plan=self.plan,
                          journal_sha256=digest(self.journal.snapshot()),
                          limitations=['Small synthetic bank; no statistical quality conclusion.',
                                       'Multi-turn retention probe; full-context behavior is unqualified.',
                                       'The selected tool surface is recorded in the host qualification.'])
            atomic_json(self.journal.directory / 'result.json', result)
            return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--configuration', required=True, type=Path)
    args = parser.parse_args(argv)
    configuration = json.loads(args.configuration.read_text())
    from .pilot_runtime import compose
    journal = Journal(configuration['directory'], {'schema': 'pw-training-pilot-inputs/1', 'configuration': configuration})
    try:
        handlers, plan = compose(journal, configuration)
        result = Pilot(journal, handlers, plan, configuration['students']).run()
        print(json.dumps(result, indent=2))
        return 0 if result['status'] == 'mechanical_pipeline_complete' else 1
    finally:
        journal.close()


if __name__ == '__main__':
    raise SystemExit(main())
