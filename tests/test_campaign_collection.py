import json

import pytest

from bialy.campaign.collection import Collection, TeacherBaseline
from bialy.campaign.journal import CampaignFault, Journal
from bialy.campaign.lineage import FAMILIES, PARTITIONS, Lineage
from bialy.provenance import file_sha256


class Driver:
    def __init__(self):
        self.calls = []
    def execute(self, name, inputs, attempt):
        self.calls.append(inputs)
        return {'success': inputs['repetition'] > 0, 'task': inputs['task']['id']}
    def validate(self, name, inputs, output):
        assert output['task'] == inputs['task']['id']


def test_collection_recovers_failed_tasks_and_reuses_pilot_episodes(tmp_path):
    journal = Journal(tmp_path / 'campaign', {})
    lineage = Lineage(journal)
    lineage.roots([{'id': partition + family, 'partition': partition, 'family': family, 'source_class': 'synthetic'}
                   for partition in PARTITIONS for family in FAMILIES])
    lineage.freeze()
    driver = Driver()
    exported = []
    def export(inputs, output):
        assert output['success']
        exported.append(output['task'])
        return {student: {'manifest': output['task'], 'rows': 2} for student in ('e4b', '12b')}
    def assemble(manifests, directory, minimum):
        students = {}
        for student in manifests:
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / (student + '.jsonl')
            path.write_text(json.dumps(manifests[student]))
            students[student] = {'directory': str(directory), 'files': {path.name: file_sha256(path)}}
        return {'students': students, 'lineage_sha256': lineage.freeze()}
    minimum = dict.fromkeys(('train', 'validation', 'system_test'), 8)
    inputs = {'host': {}, 'selector': {}, 'teachers': {
        'teachers': [{'id': 'teacher'}], 'families': {family: {'teacher_id': 'teacher'} for family in FAMILIES}},
        'plan': {'collection_episodes': minimum}}
    pilot = Collection(journal, lineage, driver, export, assemble, True, minimum=minimum)
    result = pilot.execute(inputs, 'pilot')
    pilot.validate(inputs, result)
    assert result['episode_counts'] == minimum
    assert len(driver.calls) == 48 and len(exported) == 24
    full = Collection(journal, lineage, driver, export, assemble, False)
    full_result = full.execute(inputs, 'full')
    full.validate(inputs, full_result)
    assert len(driver.calls) == 48
    assert full_result['evidence'] == result['evidence']
    path = next((journal.directory / 'datasets' / 'full').glob('*.jsonl'))
    path.write_text('changed')
    with pytest.raises(CampaignFault, match='collection_dataset_changed'):
        full.validate(inputs, full_result)
    journal.close()


def test_teacher_floors_come_from_development_attempts(tmp_path):
    journal = Journal(tmp_path, {})
    lineage = Lineage(journal)
    lineage.roots([{'id': f'{partition}-{family}-{index}', 'partition': partition, 'family': family, 'source_class': 'synthetic'}
                   for partition in PARTITIONS for family in FAMILIES for index in range(4)])
    lineage.freeze()
    driver = Driver()
    baseline = TeacherBaseline(journal, lineage, driver, [{'id': 'teacher'}])
    inputs = {'host': {'conservative_reference': {}, 'teacher_capture_qualified': {'teacher': True}}}
    result = baseline.execute(inputs, 'baseline')
    baseline.validate(inputs, result)
    assert len(driver.calls) == 64
    for route in result['families'].values():
        assert route['qualified'] == 4 and route['attempts'] == 8
        assert 0 < route['floor'] < 0.5
    assert baseline.reconcile(inputs, 'baseline') == result
    assert len(driver.calls) == 64
    journal.close()
