import json
from pathlib import Path

import pytest

from bialy.campaign.episodes import Episodes
from bialy.campaign.grading import DIMENSIONS
from bialy.campaign.journal import CampaignFault, Journal, digest
from bialy.campaign.lineage import FAMILIES, PARTITIONS, Lineage
from bialy.campaign.sources import authority_key, seal_dataset
from bialy.provenance import file_sha256


def fixture(tmp_path):
    journal = Journal(tmp_path / 'campaign', {})
    lineage = Lineage(journal)
    lineage.roots([{'id': partition + family, 'family': family, 'partition': partition, 'source_class': 'synthetic', 'required_checks': ['outcome']}
                   for partition in PARTITIONS for family in FAMILIES])
    lineage.freeze()
    lineage.bind_session('epoch', 'session', 'train' + FAMILIES[0])
    capture, store = tmp_path / 'capture.json', tmp_path / 'store.db'
    capture.write_text(json.dumps({'session_id': 'session', 'epoch': 'epoch'}))
    store.write_bytes(b'fixture database')
    candidate = {'session': 'session', 'source': {'capture_sha256': file_sha256(capture), 'store_sha256': file_sha256(store)},
                 'request_capture': {'host_revision': 'host'}, 'target': {'content': 'answer'}}
    candidates = tmp_path / 'sft.receipts.jsonl'
    candidates.write_text(json.dumps(candidate) + '\n')
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps({'schema': 'pw-receipt-export/1', 'identity': {'host_commit': 'host', 'epoch': 'epoch'},
                                   'rows_sha256': file_sha256(candidates),
                                   'inputs': {str(capture): file_sha256(capture), str(store): file_sha256(store)}}))
    row = dict(candidate, meta={'task_id': 'train' + FAMILIES[0]}, split_group='train' + FAMILIES[0])
    dataset = tmp_path / 'train.jsonl'
    dataset.write_text(json.dumps(row) + '\n')
    specification = json.loads(lineage.task('train' + FAMILIES[0])['spec'])
    receipt = {'task_spec_sha256': digest(specification), 'grader_identity': 'grader', 'complete': True,
               'checks': [{'id': 'outcome', 'source_sha256': 'final', 'passed': True}],
               'final_source_sha256': 'final', 'evidence_ids': ['check'], 'human_interventions': 0,
               'unauthorized_effects': 0, 'unresolved_effects': 0, 'teacher_family': 'teacher',
               'epoch': 'epoch', 'root_session': 'session', 'store_sha256': file_sha256(store)}
    reviews = [{'reviewer_slot': name, 'judge_id': name, 'judge_family': name, 'episode_sha256': digest(receipt),
                'controls_passed': True, 'evidence_ids': ['check'], 'dimensions': dict.fromkeys(DIMENSIONS, 'pass')}
               for name in ('first', 'second')]
    Episodes(journal, lineage).record('epoch', 'session', 'train' + FAMILIES[0], ['session'], specification, receipt, reviews, 'grader')
    return journal, lineage, dataset, manifest, store


def test_issuance_uses_frozen_ancestry_and_exported_bytes(tmp_path):
    journal, lineage, dataset, manifest, store = fixture(tmp_path)
    requests = []
    def run(args, **kwargs):
        request = json.loads(Path(args[args.index('--request') + 1]).read_text())
        requests.append(request)
        (tmp_path / 'source.json').write_text('sealed')
    seal_dataset(journal, lineage, [dataset], [manifest], tmp_path, 'host', run=run)
    binding = next(iter(requests[0]['approved'].values()))
    assert binding['partition'] == 'train' and binding['root'] == 'train' + FAMILIES[0]
    assert authority_key(journal.directory).stat().st_mode & 0o077 == 0
    store.write_bytes(b'changed')
    with pytest.raises(CampaignFault, match='input_changed'):
        seal_dataset(journal, lineage, [dataset], [manifest], tmp_path, 'host', run=run)
    assert len(requests) == 1
    journal.close()


def test_changed_training_target_cannot_receive_a_source_permit(tmp_path):
    journal, lineage, dataset, manifest, _ = fixture(tmp_path)
    row = json.loads(dataset.read_text())
    row['target']['content'] = 'invented answer'
    dataset.write_text(json.dumps(row) + '\n')
    with pytest.raises(CampaignFault, match='row_not_in_host_export'):
        seal_dataset(journal, lineage, [dataset], [manifest], tmp_path, 'host')
    journal.close()


def test_metadata_cannot_reassign_a_session_to_another_partition(tmp_path):
    journal, lineage, dataset, manifest, _ = fixture(tmp_path)
    row = json.loads(dataset.read_text())
    row['meta']['task_id'] = 'system_test' + FAMILIES[0]
    row['split_group'] = row['meta']['task_id']
    dataset.write_text(json.dumps(row) + '\n')
    with pytest.raises(CampaignFault, match='row_lineage_mismatch'):
        seal_dataset(journal, lineage, [dataset], [manifest], tmp_path, 'host')
    with pytest.raises(CampaignFault, match='session_task_changed'):
        lineage.bind_session('epoch', 'session', row['meta']['task_id'])
    journal.close()


def test_successful_turn_without_a_qualified_episode_cannot_receive_a_permit(tmp_path):
    journal, lineage, dataset, manifest, _ = fixture(tmp_path)
    journal.db.execute('DELETE FROM episodes')
    journal.db.commit()
    with pytest.raises(CampaignFault, match='episode_missing'):
        seal_dataset(journal, lineage, [dataset], [manifest], tmp_path, 'host')
    journal.close()
