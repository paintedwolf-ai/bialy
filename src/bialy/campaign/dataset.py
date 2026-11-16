"""Assemble whole rows by frozen task partition after episode admission."""
import json
import os
import subprocess
from collections import Counter
from pathlib import Path

from ..provenance import file_sha256
from ..student_host.specification import identity
from ..training_rows import training_ready
from . import coverage, data_report
from .episodes import Episodes
from .execution import atomic_json
from .grading import decided
from .journal import CampaignFault, canonical, digest
from .selector import plan_record
from .sources import exported_episode, seal_dataset, verified_exports

FILES = {'train': 'train.jsonl', 'validation': 'val.jsonl', 'system_test': 'holdout.jsonl'}
MAX_TOKENS = 65536


class Measurement:
    """Token counts from the host's encoder under each student's pinned tokenizer."""

    def __init__(self, checkout, models, specification, process=subprocess.run):
        self.checkout, self.models, self.specification, self.process = Path(checkout), Path(models), specification, process

    def __call__(self, student, rows):
        out = Path(rows).with_suffix('.measured.json')
        script = Path(__file__).resolve().parents[1] / 'student_host/measure_rows.py'
        result = self.process(['uv', 'run', '--project', str(self.checkout / 'scripts/bialy/custom_model'), '--locked', 'python',
                               str(script), '--checkout', str(self.checkout),
                               '--model', str(self.models / self.specification['models'][student]['directory']),
                               '--rows', str(rows), '--max-tokens', str(MAX_TOKENS), '--out', str(out)],
                              capture_output=True, text=True)
        if result.returncode:
            raise CampaignFault('dataset_measurement_failed', details={'student': student, 'stderr': result.stderr[-2000:]})
        return json.loads(out.read_text())


def admission_exclusions(manifests):
    """Excluded admissions by reason, from the admissions each export binds."""
    counts = Counter()
    for manifest in map(Path, manifests):
        exported = json.loads(manifest.read_text())
        path = manifest.parent / 'admissions.jsonl'
        if file_sha256(path) != exported.get('admissions_sha256'):
            raise CampaignFault('host_export_admissions_changed')
        for record in (json.loads(line) for line in path.read_text().splitlines() if line.strip()):
            if record['admission']['eligibility'] == 'excluded':
                counts[record['admission']['reason']] += 1
    return counts


def infrastructure_faults(journal):
    """Collection trials that failed before they settled, by fault code."""
    counts = Counter()
    for (body,) in journal.db.execute("SELECT body FROM events WHERE status='failed' AND stage LIKE 'collection_trial_%' ORDER BY sequence"):
        counts[(json.loads(body).get('fault') or {}).get('code', 'unknown')] += 1
    return counts


def check_row(row):
    capture = row['request_capture']
    if (digest({key: value for key, value in capture.items() if key != 'sha256'}) != capture['sha256']
            or digest(row['target']) != row['response_sha256']):
        raise CampaignFault('row_hash_mismatch', details={'session': row['session'], 'message': row['message_id']})
    return capture['catalog_sha256']


def collect(lineage, episodes, manifests, decisions, host_commit, model, strata):
    """A student's admitted rows with their ledger facts, before measurement."""
    binding, inputs, candidates = verified_exports(manifests, host_commit)
    if binding['student_model'] != model:
        raise CampaignFault('dataset_student_identity_mismatch')
    entries, exclusions, catalogs, compaction = [], admission_exclusions(manifests), set(), {}
    for candidate in candidates.values():
        task = lineage.session_task(binding['epoch'], candidate['session'])
        if task['partition'] not in FILES:
            exclusions['development_partition'] += 1
            continue
        try:
            episode = exported_episode(inputs, candidate['source']['bundle_sha256'])
            verdict = episodes.admit(binding['epoch'], candidate['session'], episode.manifest['store_sha256'], decisions)
        except CampaignFault as exc:
            if exc.code != 'episode_not_qualified':
                raise
            exclusions[exc.code] += 1
            continue
        if not training_ready(candidate):
            exclusions['response_not_qualified'] += 1
            continue
        if not verdict['episode_sha256']:
            raise CampaignFault('episode_identity_missing')
        catalogs.add(check_row(candidate))
        if episode.manifest_sha256 not in compaction:
            compaction[episode.manifest_sha256] = coverage.compacted(episode)
        root = lineage.task(task['root'])
        row = {**candidate, 'meta': {**candidate.get('meta', {}), 'task_id': task['id']}, 'split_group': task['root']}
        entries.append({'row': row, 'partition': task['partition'], 'task': {**task, 'spec': root['spec']},
                        'episode': verdict['episode_sha256'], 'stratum': coverage.stratum(strata[root['id']], candidate),
                        'compacted': compaction[episode.manifest_sha256], 'worker': candidate['session'] != episode.root_session})
    groups = {}
    for entry in entries:
        groups.setdefault(entry['task']['root'], set()).add(entry['partition'])
    if any(len(partitions) > 1 for partitions in groups.values()):
        raise CampaignFault('dataset_component_crosses_split')
    return binding, entries, exclusions, catalogs


def select(entries, rows, max_tokens):
    """The `rows[partition]` shortest whole rows within `max_tokens`, for a rehearsal."""
    out = []
    for partition, count in rows.items():
        fitting = sorted((entry for entry in entries if entry['partition'] == partition and entry['tokens'] <= max_tokens),
                         key=lambda entry: (entry['tokens'], digest(entry['row'])))
        out += fitting[:count]
    return out


def assemble(journal, lineage, manifests, decisions, checkout, host_commit, specification, destination, minimum_episodes,
             selector, strata, measure, groups_per_focus=1, selection=None, seal=seal_dataset):
    """Each student's splits from one collection. Every row is measured before any split is
    written; an oversize row is refused whole. A short focus is reported with its exact
    deficit as data_incomplete, and nothing is written. A rehearsal's `selection` keeps only
    the shortest whole rows of each partition."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    if (set(minimum_episodes) != set(FILES)
            or any(type(value) is not int or value < 1 for value in minimum_episodes.values())):
        raise CampaignFault('dataset_episode_requirements_invalid')
    episodes = Episodes(journal, lineage)
    faults = infrastructure_faults(journal)
    collected, catalogs, epochs = {}, set(), set()
    for student in specification['models']:
        binding, entries, exclusions, used = collect(lineage, episodes, manifests[student], decisions, host_commit,
                                                     specification['models'][student]['directory'], strata)
        pending = destination / (student + '-candidates.jsonl')
        pending.write_text(''.join(canonical(entry['row']) + '\n' for entry in entries))
        measured = measure(student, pending)
        if len(measured) != len(entries):
            raise CampaignFault('dataset_measurement_incomplete', details={'student': student})
        kept = []
        for entry, result in zip(entries, measured, strict=True):
            if 'refused' in result:
                exclusions[result['refused']] += 1
            elif type(result.get('tokens')) is not int or not 0 < result['tokens'] <= MAX_TOKENS:
                raise CampaignFault('dataset_measurement_invalid', details={'student': student})
            else:
                kept.append({**entry, 'tokens': result['tokens']})
        if selection:
            kept = select(kept, selection['rows'], selection['max_example_tokens'])
        collected[student] = (kept, exclusions)
        catalogs |= used
        epochs.add(binding['epoch'])
    if len(catalogs) > 1 or len(epochs) != 1:
        raise CampaignFault('dataset_epoch_mixed', details={'catalogs': sorted(catalogs), 'epochs': sorted(epochs)})
    assembly_binding = {'manifests': {student: {str(path): file_sha256(path) for path in paths} for student, paths in manifests.items()},
                        'host_commit': host_commit, 'specification': identity(specification), 'minimum_episodes': minimum_episodes,
                        'lineage_sha256': lineage.freeze(), 'decisions_sha256': digest(decisions), 'selector': plan_record(selector),
                        'catalog_sha256': next(iter(catalogs), None), 'strata_sha256': digest(strata),
                        'groups_per_focus': groups_per_focus, 'selection': selection}
    binding_path = destination / 'binding.json'
    if binding_path.exists() and json.loads(binding_path.read_text()) != assembly_binding:
        raise CampaignFault('dataset_assembly_inputs_changed')
    atomic_json(binding_path, assembly_binding)
    ledgers = {student: coverage.ledger(kept, exclusions, faults) for student, (kept, exclusions) in collected.items()}
    short = {student: found for student, (kept, _) in collected.items()
             if (found := coverage.deficits(kept, minimum_episodes, lineage.focuses, groups_per_focus))}
    if short:
        data_report.write(destination, data_report.build(assembly_binding, ledgers, 'data_incomplete', short))
        raise CampaignFault('data_incomplete', details={'deficits': short})
    report = data_report.build(assembly_binding, ledgers, 'data_ready', {})
    outputs = {}
    for student, (kept, exclusions) in collected.items():
        output = destination / student
        output.mkdir(exist_ok=True)
        paths = []
        for partition, name in FILES.items():
            path = output / name
            body = ''.join(canonical(entry['row']) + '\n' for entry in kept if entry['partition'] == partition)
            if path.exists():
                if path.read_text() != body:
                    raise CampaignFault('dataset_output_changed')
            else:
                temporary = path.with_suffix('.pending')
                with temporary.open('w') as stream:
                    stream.write(body)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary, path)
            paths.append(path)
        seal(journal, lineage, paths, manifests[student], decisions, checkout, host_commit)
        outputs[student] = {'directory': str(output), 'specification_sha256': identity(specification),
                            'files': {path.name: file_sha256(path) for path in [*paths, output / 'source.json']},
                            'episodes': {partition: counts['groups'] for partition, counts in ledgers[student]['independent'].items()},
                            'rows': {partition: counts['encodings'] for partition, counts in ledgers[student]['independent'].items()},
                            'exclusions': dict(exclusions)}
    data_report.write(destination, report)
    result = {'students': outputs, 'lineage_sha256': lineage.freeze(), 'assembly_sha256': digest(assembly_binding),
              'data_status': 'data_ready', 'data_report_sha256': file_sha256(destination / 'data-report.json')}
    atomic_json(destination / 'dataset.json', result)
    return result


class Assembly:
    """Assemble each student's splits from the first admitted attempt of every collected task.

    A human decision replaces the reviewers' verdict for the episode it names; skipped and
    unreviewed episodes keep the automated outcome.
    """
    def __init__(self, journal, students, assemble):
        self.journal, self.students, self.assemble = journal, tuple(students), assemble

    def chosen(self, collection, decisions):
        selected = {}
        for episode in sorted(collection['episodes'], key=lambda item: (item['task_id'], item['repetition'])):
            decision = decisions.get(episode['episode_sha256'])
            admitted = episode['admitted'] if decision is None else (
                decided(decision) and episode['exports'] is not None
                and all(exported['rows'] > 0 for exported in episode['exports'].values()))
            if admitted and episode['task_id'] not in selected:
                selected[episode['task_id']] = episode
        return [selected[task] for task in sorted(selected)]

    def execute(self, inputs, attempt):
        collection = inputs['collection']
        decisions = inputs['review']['decisions'] if inputs['review'] else {}
        episodes = self.chosen(collection, decisions)
        manifests = {student: [episode['exports'][student]['manifest'] for episode in episodes] for student in self.students}
        strata = {episode['task_id']: episode['stratum'] for episode in episodes}
        result = self.assemble(manifests, decisions, self.journal.directory / 'datasets' / attempt, collection['minimum'],
                               inputs['selector'], strata)
        return {**result, 'episodes': [episode['stage'] for episode in episodes],
                'retained_rows': {student: sum(episode['exports'][student]['rows'] for episode in episodes)
                                  for student in self.students}}

    def reconcile(self, inputs, attempt):
        return self.execute(inputs, attempt)

    def validate(self, inputs, output):
        decisions = inputs['review']['decisions'] if inputs['review'] else {}
        if output['episodes'] != [episode['stage'] for episode in self.chosen(inputs['collection'], decisions)]:
            raise CampaignFault('dataset_episode_selection_changed')
        for student in self.students:
            dataset = output['students'][student]
            if any(file_sha256(Path(dataset['directory']) / name) != checksum for name, checksum in dataset['files'].items()):
                raise CampaignFault('dataset_files_changed')
