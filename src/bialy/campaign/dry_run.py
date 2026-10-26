"""Exercise training, recovery, export, and serving with a small admitted dataset."""
import argparse
import json
import subprocess
from pathlib import Path

from ..provenance import file_sha256
from .dry_run_data import Data, import_evidence, qualified, read_source
from .execution import atomic_json
from .journal import CampaignFault, Journal, digest
from .pilot import Pilot
from .pilot_runtime import compose, prepare_fleet
from .student import Training

LIMITS = {'training_rows': 2, 'validation_rows': 1, 'holdout_rows': 1,
          'max_example_tokens': 10240, 'updates_before_resume': 1, 'evaluation_tasks': 1}


def runtime_configuration(configuration, directory, host, process=subprocess.run):
    epoch = configuration['epoch'][:50] + '-dry-' + digest(str(directory))[:12]
    image = configuration['host_image'].rsplit(':', 1)[0] + ':' + epoch + '-off'
    inspected = process(['docker', 'image', 'inspect', host['image']],
                        check=True, capture_output=True, text=True)
    source = json.loads(inspected.stdout)[0]
    if (source['Id'] != host['image']
            or source['Config'].get('Labels', {}).get('pw.host_commit') != configuration['host_commit']):
        raise CampaignFault('dry_run_source_image_changed')
    existing = process(['docker', 'image', 'inspect', image], capture_output=True, text=True)
    if existing.returncode == 0:
        if json.loads(existing.stdout)[0]['Id'] != host['image']:
            raise CampaignFault('dry_run_image_binding_changed')
    else:
        process(['docker', 'tag', host['image'], image], check=True, capture_output=True, text=True)
    result = {**configuration, 'directory': str(directory), 'epoch': epoch, 'host_image': image,
              'factory_root': str(directory / 'factory')}
    prepare_fleet(result['factory_root'])
    return result


def run(source, directory):
    source, directory = Path(source).resolve(), Path(directory).resolve()
    with read_source(source) as db:
        manifest = json.loads(db.execute('SELECT body FROM manifest WHERE id=1').fetchone()[0])
        cfg = manifest['configuration']
        host = qualified(db, 'host_qualification')
        teachers = qualified(db, 'teacher_baseline')
        baseline = qualified(db, 'baseline_e4b')
        base = qualified(db, 'evaluate_base_e4b')
    cfg = runtime_configuration(cfg, directory, host)
    binding = {'schema': 'pw-training-dry-run-inputs/1', 'source': str(source),
               'source_journal_sha256': file_sha256(source / 'campaign.sqlite'),
               'configuration': cfg, 'limits': LIMITS}
    journal = Journal(directory, binding)
    try:
        handlers, full_plan = compose(journal, {**cfg, 'directory': str(directory)})
        lineage = import_evidence(journal, source)
        plan = {**full_plan, 'scope': 'automation_dry_run',
                'tasks': [task for task in full_plan['tasks'] if task['id'] == 'system_test-structured_closeout']}
        if len(plan['tasks']) != 1:
            raise CampaignFault('dry_run_evaluation_task_missing')
        training_cfg = {**cfg, 'specification': str(Path(__file__).resolve().parents[3] / 'config/students.json'),
                        'full_training': 'local'}
        handlers['dry_run_data'] = Data(journal, lineage, source, cfg)
        handlers['pilot_e4b'] = Training(journal, training_cfg, 'e4b', pilot=True, dry_run=True)
        stages = Pilot(journal, handlers, plan, ['e4b'])
        result = {'schema': 'pw-training-dry-run/1', 'promotion_qualified': False,
                  'source': str(source), 'limits': LIMITS, 'plan': plan,
                  'limitations': ['Mechanical rehearsal only; no model quality or full-context qualification.']}
        common = {'host': host, 'selector': host['conservative_reference'], 'teachers': teachers,
                  'student': 'e4b', 'plan': plan}
        with journal.controller_lock():
            try:
                data = stages.stage('dry_run_data', common)
                training = stages.stage('pilot_e4b', {**common, 'dataset': data})
                artifact = stages.stage('export_e4b', {**common, 'training': training})
                evaluated = stages.stage('evaluate_trained_e4b', {**common, 'artifact': artifact})
                complete = all(row['stage_status'] == 'passed' and row['effective_thinking'] is False
                               for row in evaluated['attempts'])
                result.update(status='mechanical_pipeline_complete' if complete else 'evaluation_incomplete',
                              artifact=artifact, training=training, evaluation=evaluated,
                              reused_baseline=baseline,
                              base_attempts=[row for row in base['attempts']
                                             if row['task_id'] == plan['tasks'][0]['id']])
            except CampaignFault as exc:
                result.update(status='incomplete', fault=exc.record())
            result['journal_sha256'] = digest(journal.snapshot())
            atomic_json(directory / 'result.json', result)
        return result
    finally:
        journal.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run', required=True, type=Path)
    parser.add_argument('--directory', required=True, type=Path)
    args = parser.parse_args()
    result = run(args.source_run, args.directory)
    print(json.dumps(result, indent=2))
    return 0 if result['status'] == 'mechanical_pipeline_complete' else 1


if __name__ == '__main__':
    raise SystemExit(main())
