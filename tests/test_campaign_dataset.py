import json

import pytest

from bialy.campaign.dataset import assemble
from bialy.campaign.episodes import Episodes
from bialy.campaign.grading import DIMENSIONS
from bialy.campaign.journal import CampaignFault, Journal, digest
from bialy.campaign.lineage import FAMILIES, PARTITIONS, Lineage
from bialy.provenance import file_sha256
from bialy.student_host.specification import load, students


def bank(tmp_path, monkeypatch):
    journal = Journal(tmp_path / 'campaign', {})
    lineage = Lineage(journal)
    specifications = [{'id': partition + family, 'partition': partition, 'family': family,
                       'source_class': 'synthetic', 'required_checks': ['outcome']}
                      for partition in sorted(PARTITIONS) for family in FAMILIES]
    lineage.roots(specifications)
    lineage.freeze()
    rows = []
    for task in specifications:
        session = task['id']
        lineage.bind_session('epoch', session, task['id'])
        receipt = {'task_spec_sha256': digest(task), 'grader_identity': 'grader', 'complete': True,
                   'root_session': session, 'epoch': 'epoch', 'store_sha256': 'store',
                   'checks': [{'id': 'outcome', 'passed': True, 'source_sha256': 'final'}],
                   'final_source_sha256': 'final', 'evidence_ids': ['check'], 'human_interventions': 0,
                   'unauthorized_effects': 0, 'unresolved_effects': 0, 'teacher_family': 'teacher'}
        reviews = [{'reviewer_slot': name, 'judge_id': name, 'judge_family': name, 'episode_sha256': digest(receipt), 'controls_passed': True,
                    'evidence_ids': ['check'], 'dimensions': dict.fromkeys(DIMENSIONS, 'pass')} for name in ('first', 'second')]
        Episodes(journal, lineage).record('epoch', session, task['id'], [session], task, receipt, reviews, 'grader')
        rows.append({'session': session, 'source': {'store_sha256': 'store'}, 'target': {'content': 'complete row ' * 10000}})
    manifests = {}
    for name, model in load()['models'].items():
        folder = tmp_path / name
        folder.mkdir()
        candidates = folder / 'sft.receipts.jsonl'
        candidates.write_text(''.join(json.dumps(row) + '\n' for row in rows))
        manifest = folder / 'manifest.json'
        manifest.write_text(json.dumps({'schema': 'pw-receipt-export/1', 'identity': {
            'host_commit': 'host', 'epoch': 'epoch', 'student_model': model['directory']},
            'inputs': {}, 'rows_sha256': file_sha256(candidates)}))
        manifests[name] = [manifest]
    monkeypatch.setattr('bialy.campaign.dataset.training_eligible', lambda row: True)
    return journal, lineage, manifests


def test_paired_datasets_use_registered_partitions_and_preserve_whole_rows(tmp_path, monkeypatch):
    journal, lineage, manifests = bank(tmp_path, monkeypatch)
    seals = []
    def seal(journal, lineage, paths, manifests, decisions, checkout, host):
        seals.append(paths)
        (paths[0].parent / 'source.json').write_text('sealed')
    result = assemble(journal, lineage, manifests, {}, tmp_path, 'host', load(), tmp_path / 'dataset',
                      {'train': 8, 'validation': 8, 'system_test': 8}, seal=seal)
    assert len(seals) == len(students())
    for name in students():
        assert result['students'][name]['episodes'] == {'train': 8, 'validation': 8, 'system_test': 8}
        rows = [json.loads(line) for line in (tmp_path / 'dataset' / name / 'train.jsonl').read_text().splitlines()]
        assert all(row['split_group'].startswith('train') for row in rows)
        assert all(len(row['target']['content']) == 130000 for row in rows)
    journal.close()


def test_missing_episode_coverage_never_publishes_training_permits(tmp_path, monkeypatch):
    journal, lineage, manifests = bank(tmp_path, monkeypatch)
    calls = []
    with pytest.raises(CampaignFault, match='insufficient_qualified_data'):
        assemble(journal, lineage, manifests, {}, tmp_path, 'host', load(), tmp_path / 'dataset',
                 {'train': 32, 'validation': 16, 'system_test': 8}, seal=lambda *args: calls.append(True))
    assert not calls
    journal.close()
