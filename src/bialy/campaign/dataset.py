"""Assemble whole rows by frozen task partition after episode admission."""
import json
import os
from collections import Counter
from pathlib import Path

from ..dpo import training_eligible
from ..provenance import file_sha256
from ..student_host.specification import identity
from .episodes import Episodes
from .execution import atomic_json
from .journal import CampaignFault, canonical, digest
from .lineage import FAMILIES
from .sources import seal_dataset, verified_exports

FILES = {'train': 'train.jsonl', 'validation': 'val.jsonl', 'system_test': 'holdout.jsonl'}


def assemble(journal, lineage, manifests, checkout, host_commit, specification, destination, minimum_episodes, seal=seal_dataset):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    assembly_binding = {'manifests': {student: {str(path): file_sha256(path) for path in paths} for student, paths in manifests.items()},
               'host_commit': host_commit, 'specification': identity(specification), 'minimum_episodes': minimum_episodes,
               'lineage_sha256': lineage.freeze()}
    binding_path = destination / 'binding.json'
    if binding_path.exists() and json.loads(binding_path.read_text()) != assembly_binding:
        raise CampaignFault('dataset_assembly_inputs_changed')
    atomic_json(binding_path, assembly_binding)
    episodes = Episodes(journal, lineage)
    outputs = {}
    for student in specification['models']:
        binding, _, candidates = verified_exports(manifests[student], host_commit)
        if binding['student_model'] != specification['models'][student]['directory']:
            raise CampaignFault('dataset_student_identity_mismatch')
        rows = {partition: [] for partition in FILES}
        independent = {partition: set() for partition in FILES}
        families = {partition: set() for partition in FILES}
        exclusions = Counter()
        for candidate in candidates.values():
            task = lineage.session_task(binding['epoch'], candidate['session'])
            partition = task['partition']
            if partition not in rows:
                exclusions['development_partition'] += 1
                continue
            try:
                verdict = episodes.admit(binding['epoch'], candidate['session'], candidate['source']['store_sha256'])
            except CampaignFault as exc:
                if exc.code != 'episode_not_qualified':
                    raise
                exclusions[exc.code] += 1
                continue
            if not training_eligible(candidate):
                exclusions['response_not_qualified'] += 1
                continue
            row = {**candidate, 'meta': {'task_id': task['id']}, 'split_group': task['root']}
            rows[partition].append(row)
            independent[partition].add(task['root'])
            families[partition].add(task['family'])
            if not verdict['episode_sha256']:
                raise CampaignFault('episode_identity_missing')
        counts = {partition: len(values) for partition, values in independent.items()}
        if (set(minimum_episodes) != set(FILES)
                or any(type(value) is not int or value < 1 for value in minimum_episodes.values())):
            raise CampaignFault('dataset_episode_requirements_invalid')
        if any(counts[key] < minimum_episodes[key] or families[key] != set(FAMILIES) for key in FILES):
            atomic_json(destination / (student + '-insufficient.json'), {'episodes': counts, 'exclusions': dict(exclusions)})
            raise CampaignFault('insufficient_qualified_data', details={'student': student, 'episodes': counts})
        output = destination / student
        output.mkdir(exist_ok=True)
        paths = []
        for partition, name in FILES.items():
            path = output / name
            body = ''.join(canonical(row) + '\n' for row in rows[partition])
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
        seal(journal, lineage, paths, manifests[student], checkout, host_commit)
        outputs[student] = {'directory': str(output), 'specification_sha256': identity(specification),
                            'files': {path.name: file_sha256(path) for path in [*paths, output / 'source.json']},
                            'episodes': counts, 'rows': {key: len(value) for key, value in rows.items()},
                            'exclusions': dict(exclusions)}
    result = {'students': outputs, 'lineage_sha256': lineage.freeze(), 'assembly_sha256': digest(assembly_binding)}
    atomic_json(destination / 'dataset.json', result)
    return result


class Assembly:
    """Assemble each student's splits from the first admitted attempt of every collected task."""
    def __init__(self, journal, students, assemble):
        self.journal, self.students, self.assemble = journal, tuple(students), assemble

    def chosen(self, collection):
        selected = {}
        for episode in sorted(collection['episodes'], key=lambda item: (item['task_id'], item['repetition'])):
            if episode['admitted'] and episode['task_id'] not in selected:
                selected[episode['task_id']] = episode
        return [selected[task] for task in sorted(selected)]

    def execute(self, inputs, attempt):
        collection = inputs['collection']
        episodes = self.chosen(collection)
        manifests = {student: [episode['exports'][student]['manifest'] for episode in episodes] for student in self.students}
        result = self.assemble(manifests, self.journal.directory / 'datasets' / attempt, collection['minimum'])
        return {**result, 'episodes': [episode['stage'] for episode in episodes],
                'retained_rows': {student: sum(episode['exports'][student]['rows'] for episode in episodes)
                                  for student in self.students}}

    def reconcile(self, inputs, attempt):
        return self.execute(inputs, attempt)

    def validate(self, inputs, output):
        if output['episodes'] != [episode['stage'] for episode in self.chosen(inputs['collection'])]:
            raise CampaignFault('dataset_episode_selection_changed')
        for student in self.students:
            dataset = output['students'][student]
            if any(file_sha256(Path(dataset['directory']) / name) != checksum for name, checksum in dataset['files'].items()):
                raise CampaignFault('dataset_files_changed')
