import json
from pathlib import Path

import pytest

from bialy.campaign.collection import Collection, TeacherBaseline
from bialy.campaign.dataset import Assembly
from bialy.campaign.journal import CampaignFault, Journal
from bialy.campaign.lineage import FAMILIES, PARTITIONS, Lineage
from bialy.provenance import file_sha256
from bialy.student_host.specification import students


class Driver:
    def __init__(self):
        self.calls = []
    def execute(self, name, inputs, attempt):
        self.calls.append(inputs)
        return {'success': inputs['repetition'] > 0, 'task': inputs['task']['id'],
                'verdict': {'episode_sha256': inputs['task']['id'] + str(inputs['repetition'])}}
    def validate(self, name, inputs, output):
        assert output['task'] == inputs['task']['id']


def test_collection_records_every_episode_and_reuses_pilot_attempts(tmp_path):
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
        manifest = tmp_path / (output['task'] + '.json')
        manifest.write_text(output['task'])
        return {student: {'manifest': str(manifest), 'rows': 2} for student in students()}
    minimum = dict.fromkeys(('train', 'validation', 'system_test'), 8)
    inputs = {'host': {}, 'selector': {}, 'teachers': {
        'teachers': [{'id': 'teacher'}], 'families': {family: {'teacher_id': 'teacher'} for family in FAMILIES}},
        'plan': {'collection_episodes': minimum}}
    pilot = Collection(journal, lineage, driver, export, students(), True, minimum=minimum)
    result = pilot.execute(inputs, 'pilot')
    pilot.validate(inputs, result)
    assert len(driver.calls) == 48 and len(exported) == 24
    assert [episode['admitted'] for episode in result['episodes']] == [False, True] * 24
    assert result['minimum'] == minimum
    full = Collection(journal, lineage, driver, export, students(), False)
    full_result = full.execute(inputs, 'full')
    full.validate(inputs, full_result)
    assert len(driver.calls) == 48
    assert full_result['evidence'] == result['evidence']
    Path(result['episodes'][1]['exports'][students()[0]]['manifest']).write_text('changed')
    with pytest.raises(CampaignFault, match='collection_export_changed'):
        full.validate(inputs, full_result)
    journal.close()


def test_assembly_uses_the_first_admitted_attempt_of_each_task(tmp_path):
    journal = Journal(tmp_path / 'campaign', {})
    def episode(task, repetition, admitted):
        exports = {student: {'manifest': f'{task}-{repetition}', 'manifest_sha256': '', 'rows': 3} for student in students()}
        return {'stage': f'{task}-{repetition}', 'task_id': task, 'repetition': repetition, 'admitted': admitted,
                'episode_sha256': f'{task}-{repetition}',
                'exports': exports if admitted else None}
    collection = {'minimum': {'train': 1}, 'episodes': [
        episode('b', 0, False), episode('b', 1, True), episode('a', 1, True), episode('a', 0, True), episode('c', 0, False)]}
    seen = {}
    def assemble(manifests, decisions, directory, minimum):
        seen.update(manifests=manifests, minimum=minimum, decisions=decisions)
        files = {}
        for student in manifests:
            path = directory / (student + '.jsonl')
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(manifests[student]))
            files[student] = {'directory': str(directory), 'files': {path.name: file_sha256(path)}}
        return {'students': files}
    stage = Assembly(journal, students(), assemble)
    inputs = {'collection': collection, 'review': None}
    result = stage.execute(inputs, 'attempt')
    stage.validate(inputs, result)
    assert result['episodes'] == ['a-0', 'b-1']
    assert seen['manifests'] == {student: ['a-0', 'b-1'] for student in students()}
    assert seen['minimum'] == {'train': 1} and result['retained_rows'] == dict.fromkeys(students(), 6)
    with pytest.raises(CampaignFault, match='selection_changed'):
        stage.validate({'collection': {**collection, 'episodes': collection['episodes'][:2]}, 'review': None}, result)
    next((journal.directory / 'datasets' / 'attempt').glob('*.jsonl')).write_text('changed')
    with pytest.raises(CampaignFault, match='dataset_files_changed'):
        stage.validate(inputs, result)
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


@pytest.mark.parametrize('kind', ['teacher', 'collection'])
def test_infrastructure_failure_stops_paid_episodes_without_grading_teacher(tmp_path, kind):
    journal = Journal(tmp_path, {})
    lineage = Lineage(journal)
    lineage.roots([{'id': partition + family, 'partition': partition, 'family': family, 'source_class': 'synthetic'}
                   for partition in PARTITIONS for family in FAMILIES])
    lineage.freeze()
    driver = Driver()
    def broken(name, inputs, attempt):
        driver.calls.append(inputs)
        raise CampaignFault('evidence_unavailable')
    driver.execute = broken
    inputs = {'host': {'conservative_reference': {}, 'teacher_capture_qualified': {'teacher': True}},
              'selector': {}, 'teachers': {'teachers': [{'id': 'teacher'}],
                                          'families': {family: {'teacher_id': 'teacher'} for family in FAMILIES}}}
    if kind == 'teacher':
        stage = TeacherBaseline(journal, lineage, driver, [{'id': 'teacher'}], minimum_tasks=1, repetitions=1)
    else:
        stage = Collection(journal, lineage, driver, None, students(), True)
    with pytest.raises(CampaignFault, match='episode_incomplete') as failure:
        stage.execute(inputs, 'run')
    assert failure.value.details['fault']['code'] == 'evidence_unavailable'
    assert len(driver.calls) == 1
    journal.close()
