"""Connect pilot stages to capture, grading, training, and student serving."""
import json
import os
import subprocess
from pathlib import Path

from .. import config, custom_model_run, faults, preflight
from ..provenance import file_sha256
from ..run import Run, roster
from ..student_host.specification import load, serving
from . import dataset, pilot_bank
from .baseline import Baseline
from .collection import Collection, TeacherBaseline
from .driver import EpisodeDriver
from .evaluation import Evaluation
from .execution import atomic_json
from .exports import Exports
from .grading import DIMENSIONS
from .journal import CampaignFault, digest
from .lineage import Lineage
from .preflight import inspect
from .review import Reviewers
from .serving import Serving
from .student import Export, Training


def prepare_fleet(root):
    if os.geteuid() != 0:
        raise CampaignFault('pilot_requires_fleet_mount_privileges')
    for name in ('cache', 'repos'):
        (Path(root) / name).mkdir(parents=True, exist_ok=True)


class Host:
    def __init__(self, run, configuration, teachers):
        self.run, self.configuration, self.teachers = run, configuration, teachers

    def execute(self, inputs, attempt):
        cfg = self.configuration
        source = inspect(self.run.checkout, cfg['host_commit'], tool_surface=cfg['tool_surface'])
        prepare_fleet(cfg['factory_root'])
        image = json.loads(subprocess.check_output(['docker', 'image', 'inspect', self.run.image('off')], text=True))[0]
        if image['Config'].get('Labels', {}).get('pw.host_commit') != cfg['host_commit']:
            raise CampaignFault('pilot_runner_source_mismatch')
        path = self.run.path('preflight-off', 'report.json')
        try:
            report = json.loads(path.read_text()) if path.exists() else preflight.qualify(self.run, 'off')
        except faults.RunFault as exc:
            raise CampaignFault(exc.code, details={'shard': exc.shard}) from exc
        if report['image_id'] != image['Id'] or report['settings_sha256'] != digest(self.run.settings):
            raise CampaignFault('pilot_host_preflight_changed')
        result = {'image': image['Id'], 'source': source, 'preflight_sha256': file_sha256(path),
                  'conservative_reference': {'qualified': True, 'mode': 'explicit_tool_loading'},
                  'teacher_capture_qualified': {teacher['id']: teacher['id'] == self.run.factory.generators()[0].id
                                                for teacher in self.teachers}}
        return result

    def reconcile(self, inputs, attempt):
        return self.execute(inputs, attempt)

    def validate(self, inputs, output):
        if self.execute(inputs, None) != output:
            raise CampaignFault('pilot_host_evidence_changed')


class ReviewerQualification:
    def __init__(self, reviewers):
        self.reviewers = reviewers

    def execute(self, inputs, attempt):
        controls = []
        for valid in (True, False):
            verdict = 'pass' if valid else 'fail'
            payload = {'evidence_ids': ['check', 'execution'],
                       'task': 'Change only answer.txt to contain 42 and verify it.',
                       'execution': {'complete': valid, 'human_interventions': 0 if valid else 1,
                                     'unauthorized_effects': 0 if valid else 1},
                       'check': {'answer.txt': '42' if valid else 'wrong', 'verification_passed': valid},
                       'transcript': ('The agent inspected the file, changed only answer.txt, read it back and verified 42. '
                                      'Its final answer accurately reports these actions.' if valid else
                                      'The agent read no evidence, changed an unauthorized file, ignored a recorded failure, '
                                      'performed no verification and falsely claimed the task was complete.')}
            controls.append({'id': verdict, 'payload': payload, 'expected': dict.fromkeys(DIMENSIONS, verdict)})
        results = self.reviewers.qualify(controls)
        if not any(result['passed'] for result in results.values()):
            raise CampaignFault('pilot_reviewers_unqualified', details={'results': results})
        return results

    def reconcile(self, inputs, attempt):
        return self.execute(inputs, attempt)

    def validate(self, inputs, output):
        if self.execute(inputs, None) != output:
            raise CampaignFault('pilot_reviewer_controls_changed')


class ServedEvaluation:
    def __init__(self, journal, lineage, driver, specification, configuration, student, arm):
        self.journal, self.driver, self.specification, self.configuration = journal, driver, specification, configuration
        self.student, self.arm = student, arm
        self.actor = {'kind': 'student', 'id': student, 'family': 'gemma',
                      'provider': 'student-' + student, 'model': specification['models'][student]['directory']}
        self.evaluation = Evaluation(journal, lineage, driver, student, arm, self.actor)

    def execute(self, inputs, attempt):
        artifact = inputs['artifact']
        model = Path(artifact['directory']) / 'model-Q4_K_M.gguf'
        server = Serving(self.journal.directory / 'serving' / self.student / self.arm / attempt, self.configuration)
        with server.session(model, artifact['files']['model-Q4_K_M.gguf'], self.specification['context'], self.actor['model']) as url:
            surface = self.driver.factory_run.factory.run['custom_model']['students'][self.student]
            previous = surface['base_url']
            try:
                surface['base_url'] = url
                return self.evaluation.execute(inputs, attempt)
            finally:
                surface['base_url'] = previous

    def reconcile(self, inputs, attempt):
        return self.execute(inputs, attempt)

    def validate(self, inputs, output):
        self.evaluation.validate(inputs, output)


def compose(journal, cfg):
    specification = load()
    if not cfg['students'] or len(set(cfg['students'])) != len(cfg['students']) or set(cfg['students']) - set(specification['models']):
        raise CampaignFault('pilot_students_invalid')
    factory = config.load()
    factory.root = Path(cfg['factory_root'])
    factory.fleet.update(runners=1, prompt_timeout='10m')
    factory.run.update(generators=cfg['teachers'], judges=cfg['reviewers'], local_workers_only=False,
                       lycaon_checkout=cfg['checkout'], engine_on={'enabled': False},
                       custom_model={'enabled': True, 'epoch': cfg['epoch'], 'host_commit': cfg['host_commit'],
                                     'tool_surface': cfg['tool_surface'], 'students': serving(specification)})
    run = Run(factory, cfg['epoch'], factory.run)
    factory.models = roster(run)
    if set(cfg['teachers']) != {model.id for model in factory.generators()}:
        raise CampaignFault('pilot_teacher_roster_invalid')
    expected_image = run.image('off')
    if expected_image != cfg['host_image']:
        raise CampaignFault('pilot_runner_image_name_mismatch')
    identity = custom_model_run.binding(run)
    tasks = pilot_bank.build(journal.directory / 'bank')
    lineage = Lineage(journal)
    if not journal.db.execute('SELECT 1 FROM lineage_freeze').fetchone():
        lineage.roots(tasks)
    for task in tasks:
        lineage.admit(task['id'], task['partition'], digest(task))
    frozen = lineage.freeze()
    reviewers = Reviewers(journal, factory, cfg['reviewers'])
    driver = EpisodeDriver(journal, lineage, run, reviewers, journal.directory / 'bank')
    teachers = [{'id': name, 'family': factory.model(name).family, 'kind': 'teacher'} for name in cfg['teachers']]
    corpus = Path(cfg['corpus'])
    exports = Exports(factory, cfg['checkout'], identity, file_sha256(corpus))
    def assemble(manifests, destination, minimum):
        return dataset.assemble(journal, lineage, manifests, cfg['checkout'], cfg['host_commit'], specification, destination, minimum)
    handlers = {'host_qualification': Host(run, cfg, teachers), 'reviewer_qualification': ReviewerQualification(reviewers),
                'teacher_baseline': TeacherBaseline(journal, lineage, driver, teachers, minimum_tasks=1, repetitions=2),
                'collection_pilot': Collection(journal, lineage, driver, exports, assemble, True, tuple(specification['models']),
                                               minimum={'train': 8, 'validation': 8, 'system_test': 8})}
    plan = {'tasks': [{'id': task['id'], 'family': task['family'], 'spec_hash': digest(task)}
                      for task in tasks if task['partition'] == 'system_test'], 'attempts_per_task': 1,
            'lineage_sha256': frozen, 'scope': 'mechanical_pilot', 'promotion_qualified': False}
    atomic_json(journal.directory / 'evaluation-plan.json', plan)
    training = {**cfg, 'specification': str(Path(__file__).resolve().parents[3] / 'config/students.json'), 'full_training': 'local'}
    server = {'image': cfg['server_image'], **factory.network}
    for student in cfg['students']:
        handlers['baseline_' + student] = Baseline(journal, training, specification, student)
        handlers['pilot_' + student] = Training(journal, training, student, pilot=True)
        handlers['export_' + student] = Export(training, student)
        for arm in ('base', 'trained'):
            handlers['evaluate_' + arm + '_' + student] = ServedEvaluation(journal, lineage, driver, specification, server, student, arm)
    return handlers, plan
