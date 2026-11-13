"""`bialy campaign plan|run|resume|status|report`: one handler graph for every profile.

`plan` is offline: it freezes the configuration, the profile, the source commits, the
rosters, the scenario bank and its split, the fallback rate and the evaluation policy into
a new, read-only plan directory. `run` journals under that directory's run/.
"""
import json
import shlex
import subprocess
import time
from pathlib import Path

import yaml

from .. import config, custom_model_run
from .. import run as runmod
from ..provenance import file_sha256
from ..run import Run, roster
from ..student_host.specification import load as specification_load
from ..student_host.specification import serving
from . import ancestry, configuration, dataset, evaluation_policy, profiles, scenarios, selector
from .baseline import Baseline
from .collection import Collection
from .dataset import Assembly
from .driver import EpisodeDriver
from .evaluation import Comparison, EvaluationPlan
from .execution import ReceiptBackend, atomic_json
from .exports import Exports
from .handlers import Adoption, Cleanup, HeadRows, Host, ReviewerQualification, ServedEvaluation, SuppliedSelector
from .human_review import CollectionGate, EvaluationGate, Review
from .journal import CampaignFault, Journal, digest
from .lineage import Lineage
from .orchestration import Campaign
from .review_contract import Reviewers
from .student import Export, Training
from .teacher_qualification import TeacherQualification

PLAN = 'pw-campaign-plan/1'
ROOT = Path(__file__).resolve().parents[3]


def factory_commit():
    result = subprocess.run(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else None


def plan(path, out):
    """Freeze a campaign into a new plan directory, without any network call."""
    out = Path(out)
    if out.exists():
        raise CampaignFault('campaign_plan_directory_exists', details={'directory': str(out)})
    raw = yaml.safe_load(Path(path).read_text()) or {}
    profile = profiles.load(raw.get('profile'))
    cfg = configuration.load(path, profile)
    out.mkdir(parents=True)
    catalog = scenarios.load()
    tasks = scenarios.tasks(catalog, out / 'bank', lambda name: '/work/' + name)
    assignment = scenarios.assignment(catalog, tasks, cfg['epoch'], profile['quota'])
    assignment.save(out / 'ancestry.json')
    body = {'schema': PLAN, 'profile': profile, 'configuration': cfg,
            'sources': {'host_commit': cfg['host_commit'], 'factory_commit': factory_commit(), 'llama_revision': cfg['llama_revision']},
            'students': cfg['students'], 'teachers': cfg['teachers'], 'reviewers': cfg['reviewers'],
            'split': {'assignment_sha256': assignment.sha256, 'tasks_sha256': file_sha256(out / 'bank' / 'tasks.json'),
                      'catalog_sha256': catalog.sha256},
            'fallback_rate': selector.load()['fallback_rate'], 'selector_sha256': digest(selector.load()),
            'evaluation_policy_sha256': digest(evaluation_policy.load())}
    body['plan_sha256'] = digest(body)
    atomic_json(out / 'plan.json', body)
    for frozen in (out / 'plan.json', out / 'ancestry.json', out / 'bank' / 'tasks.json'):
        frozen.chmod(0o444)
    return body


def read_plan(directory):
    directory = Path(directory)
    body = json.loads((directory / 'plan.json').read_text())
    if (body.get('schema') != PLAN or digest({key: value for key, value in body.items() if key != 'plan_sha256'}) != body['plan_sha256']
            or ancestry.load(directory / 'ancestry.json').sha256 != body['split']['assignment_sha256']
            or file_sha256(directory / 'bank' / 'tasks.json') != body['split']['tasks_sha256']):
        raise CampaignFault('campaign_plan_changed')
    return body


def factory_for(cfg):
    factory = config.load()
    factory.root = Path(cfg['factory_root'])
    factory.fleet.update(runners=1, prompt_timeout='10m')
    return factory


def host_run(cfg, specification, epoch):
    """The factory run that drives application episodes against one host commit."""
    factory = factory_for(cfg)
    factory.run.update(generators=cfg['teachers'], judges=cfg['reviewers'], local_workers_only=False,
                       lycaon_checkout=cfg['checkout'], engine_on={'enabled': False},
                       custom_model={'enabled': True, 'epoch': epoch, 'host_commit': cfg['host_commit'],
                                     'students': serving(specification)})
    run = Run(factory, epoch, factory.run)
    factory.models = roster(run)
    if set(cfg['teachers']) != {model.id for model in factory.generators()}:
        raise CampaignFault('campaign_teacher_roster_invalid')
    if run.image('off') != cfg['host_image']:
        raise CampaignFault('campaign_runner_image_name_mismatch', details={'expected': run.image('off')})
    return run


def head_run(cfg):
    factory = factory_for(cfg)
    head = cfg['head_rows']
    settings = runmod.settings_for(factory, {**head.get('overrides', {}), 'engine_on.enabled': False})
    settings.update(lycaon_checkout=cfg['checkout'])
    return Run(factory=factory, name=head['run'], settings=settings)


def compose(journal, directory, body, adopted=None):
    """Every handler of the campaign graph. The student segment runs on the adopted commit
    when the profile has a selector segment, and on the planned commit otherwise."""
    cfg, profile = body['configuration'], body['profile']
    specification = specification_load()
    run_command = shlex.join(['bialy', 'campaign', 'resume', '--run', str(journal.directory)])
    teachers_run = host_run(cfg, specification, cfg['epoch'])
    teachers = [{'id': name, 'family': teachers_run.factory.model(name).family, 'kind': 'teacher'} for name in cfg['teachers']]
    tasks = json.loads((Path(directory) / 'bank' / 'tasks.json').read_text())
    assignment = ancestry.load(Path(directory) / 'ancestry.json')
    lineage = Lineage(journal, scenarios.load().focuses)
    if not journal.db.execute('SELECT 1 FROM lineage_freeze').fetchone():
        lineage.roots(tasks, assignment)
    for task in tasks:
        lineage.admit(task['id'], lineage.task(task['id'])['partition'], digest(task))
    lineage.freeze()
    reviewers = Reviewers(journal, teachers_run.factory, cfg['reviewers'])
    bank = Path(directory) / 'bank'
    teacher_driver = EpisodeDriver(journal, lineage, teachers_run, reviewers, bank)
    handlers = {'host_qualification': Host(teachers_run, cfg, teachers), 'reviewer_qualification': ReviewerQualification(reviewers),
                'teacher_qualification': TeacherQualification(journal, lineage, teacher_driver, teachers,
                                                              minimum_tasks=profile.get('teacher_tasks_per_focus', 1), repetitions=2)}
    student_cfg = cfg
    if 'selector' in profile['segments']:
        rows_run = head_run(cfg)
        handlers.update(head_rows=HeadRows(rows_run), selector_training=selector.SelectorTraining(journal, rows_run),
                        selector_qualification=selector.SelectorQualification(journal, rows_run),
                        selector_holdout=selector.SelectorHoldout(journal, rows_run),
                        host_adoption=Adoption(cfg['checkout'], adopted, run_command + ' --host-commit COMMIT --host-image IMAGE'))
        if adopted is None:
            return handlers
        student_cfg = {**cfg, 'host_commit': adopted['host_commit'], 'host_image': adopted['host_image'],
                       'epoch': cfg['epoch'] + '-' + adopted['host_commit'][:12]}
    else:
        handlers['selector_supplied'] = SuppliedSelector(cfg.get('selector_policy'))
    students = tuple(cfg['students'])
    run = host_run(student_cfg, specification, student_cfg['epoch']) if student_cfg is not cfg else teachers_run
    driver = teacher_driver if run is teachers_run else EpisodeDriver(journal, lineage, run, reviewers, bank)
    if student_cfg is not cfg:
        handlers['student_host_qualification'] = Host(run, student_cfg, teachers)
    exports = Exports(run.factory, cfg['checkout'], custom_model_run.binding(run), file_sha256(cfg['corpus']))
    measure = dataset.Measurement(cfg['checkout'], cfg['models'], specification)
    coverage = profile['coverage']
    groups = 0 if coverage == 'none' else coverage['groups_per_focus']
    selection = {'rows': profile['rows'], 'max_example_tokens': profile['max_example_tokens']} if profile.get('rows') else None

    def assemble(manifests, decisions, destination, minimum, chosen, strata):
        return dataset.assemble(journal, lineage, manifests, decisions, cfg['checkout'], student_cfg['host_commit'], specification,
                                destination, minimum, chosen, strata, measure, groups, selection)
    review = Review(journal.directory) if cfg['human_review'] else None
    command = shlex.join(['bialy', 'review', '--run', str(journal.directory)])
    gate = CollectionGate(review, 'collection', command) if review else None
    handlers.update(collection=Collection(journal, lineage, driver, exports, students, profile['collection'], groups, review=gate),
                    dataset=Assembly(journal, students, assemble),
                    evaluation_plan=EvaluationPlan(journal, lineage, collection_episodes=profile['collection'],
                                                   judged=profile['judged'], tasks=profile.get('evaluation_tasks')),
                    cleanup=Cleanup(cfg))
    training = {**student_cfg, 'specification': str(ROOT / 'config/students.json')}
    server = {'image': cfg['server_image'], 'gateway': cfg['gateway'], **run.factory.network}
    for student in students:
        handlers['baseline_' + student] = Baseline(journal, training, specification, student)
        handlers['train_' + student] = Training(journal, training, student, pilot=profile['recipe'] != 'full',
                                                dry_run=profile['recipe'] == 'dry_run')
        handlers['export_' + student] = Export(training, student)
        handlers['compare_' + student] = Comparison(journal)
        for arm in ('base', 'trained'):
            handlers['evaluate_' + arm + '_' + student] = ServedEvaluation(journal, lineage, driver, specification, server, student, arm)
    if review:
        handlers['review_collection'] = gate
        handlers['review_evaluation'] = EvaluationGate(review, 'evaluation', journal, command)
    return handlers


def adoption_of(journal):
    stage = journal.stage('host_adoption')
    return stage['output'] if stage and stage['status'] == 'passed' else None


def drive(directory, adopted=None, sleep=time.sleep, composer=None):
    """Run or resume the campaign planned in `directory`."""
    directory = Path(directory)
    body = read_plan(directory)
    journal = Journal(directory / 'run', {'schema': 'pw-campaign-run/1', 'plan_sha256': body['plan_sha256']})
    try:
        handlers = (composer or compose)(journal, directory, body, adoption_of(journal) or adopted)
        resume = shlex.join(['bialy', 'campaign', 'resume', '--run', str(journal.directory)])
        return Campaign(journal, ReceiptBackend(journal.directory / 'receipts', handlers), sleep, body['profile'],
                        body['students'], review=body['configuration']['human_review'], resume=resume).run()
    finally:
        journal.close()


def status(run_directory):
    run_directory = Path(run_directory)
    journal = Journal(run_directory, None)
    try:
        stages = {row['name']: row['status'] for row in journal.db.execute('SELECT name,status FROM stages ORDER BY rowid')}
    finally:
        journal.close()
    out = {'stages': stages}
    for name in ('waiting', 'result'):
        path = run_directory / (name + '.json')
        if path.exists():
            value = json.loads(path.read_text())
            out[name] = value if name == 'waiting' else {key: value[key] for key in value if key != 'journal'}
    return out


def report(run_directory):
    """The terminal result without the journal, with the reports it produced."""
    run_directory = Path(run_directory)
    path = run_directory / 'result.json'
    if not path.exists():
        raise CampaignFault('campaign_result_unavailable')
    result = json.loads(path.read_text())
    return {**{key: value for key, value in result.items() if key != 'journal'},
            'reports': sorted(str(item.relative_to(run_directory)) for pattern in ('datasets/*/data-report.md', 'evaluation-policy.json',
                                                                                   'selector/p0.json')
                              for item in run_directory.glob(pattern))}


def main(args):
    if args.action == 'plan':
        body = plan(args.config, args.out)
        print(json.dumps({'plan': str(args.out), 'plan_sha256': body['plan_sha256'], 'profile': body['profile']['name']}, indent=2))
        return 0
    if args.action == 'run':
        result = drive(args.plan)
    elif args.action == 'resume':
        if bool(args.host_commit) != bool(args.host_image):
            raise CampaignFault('adoption_needs_commit_and_image')
        adopted = {'host_commit': args.host_commit, 'host_image': args.host_image} if args.host_commit else None
        result = drive(Path(args.run).parent, adopted)
    elif args.action == 'decide':
        from . import decisions
        print(json.dumps(decisions.decide(args.run, json.loads(args.probes.read_text()), args.by), indent=2))
        return 0
    elif args.action == 'record':
        from . import decisions
        print(json.dumps(decisions.record(args.run, args.ledger, args.campaign, json.loads(args.publications.read_text())), indent=2))
        return 0
    elif args.action == 'status':
        print(json.dumps(status(args.run), indent=2))
        return 0
    else:
        print(json.dumps(report(args.run), indent=2))
        return 0
    printable = {key: value for key, value in result.items() if key != 'journal'}
    print(json.dumps(printable, indent=2))
    if result['status'] in ('waiting', 'awaiting_host_adoption'):
        print('Resume with: ' + result['waiting'].get('command', result['waiting']['resume']))
        return 3
    return 0 if result['status'] in ('qualified', 'mechanical_pipeline_complete') else 1
