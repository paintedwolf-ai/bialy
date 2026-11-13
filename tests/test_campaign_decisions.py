import json
from datetime import datetime, timezone

import pytest

from bialy.campaign import decisions
from bialy.campaign.journal import CampaignFault
from bialy.provenance import file_sha256
from bialy.student_host import exporting

AT = datetime(2026, 11, 13, 12, tzinfo=timezone.utc)


def export(root, student):
    job = root / student
    for name in ('gguf', 'mlx', 'lora_adapter'):
        (job / name).mkdir(parents=True)
        (job / name / 'payload.bin').write_bytes(('%s-%s' % (student, name)).encode())
    receipt = {'outputs': {key: {'directory': name, 'files': exporting.output_files(job, name)}
                           for key, name in (('vulkan', 'gguf'), ('mlx', 'mlx'), ('lora_adapter', 'lora_adapter'))}}
    (job / 'export_receipt.json').write_text(json.dumps(receipt))
    return {'directory': str(job / 'gguf'), 'export_receipt_sha256': file_sha256(job / 'export_receipt.json')}


def campaign(tmp_path, outcomes=None, profile='full'):
    run = tmp_path / 'run'
    run.mkdir(parents=True)
    artifacts = {student: export(tmp_path / 'exports', student) for student in ('e4b', '12b', '26b-a4b')}
    students = {student: {'execution_status': 'complete', 'behavioral_status': 'pass', 'artifact': artifact}
                for student, artifact in artifacts.items()}
    for student, changes in (outcomes or {}).items():
        students[student].update(changes)
    (run / 'result.json').write_text(json.dumps({'profile': profile, 'dry_run': profile == 'dry-run', 'journal_sha256': 'j' * 64,
                                                 'students': students}))
    return run, artifacts


def probes(artifacts, skip=()):
    return [{'schema': decisions.PROBE, 'student': student, 'bundle': bundle, 'passed': True,
             'export_receipt_sha256': artifact['export_receipt_sha256'], 'runtime': 'host-native'}
            for student, artifact in artifacts.items() for bundle in decisions.BUNDLES if (student, bundle) not in skip]


def test_only_a_tier_meeting_every_condition_is_published(tmp_path):
    run, artifacts = campaign(tmp_path, {'12b': {'behavioral_status': 'inconclusive'}})
    decision = decisions.decide(run, probes(artifacts, skip={('26b-a4b', 'vulkan')}), 'operator', lambda: AT)
    assert decision['tiers']['e4b'] == {'publish': True, 'failed': [], 'export_receipt_sha256': artifacts['e4b']['export_receipt_sha256']}
    assert decision['tiers']['12b']['failed'] == ['behavioral_inconclusive']
    assert decision['tiers']['26b-a4b']['failed'] == ['load_probe_vulkan']
    assert decision['decided_by'] == 'operator' and decision['decided_at'] == AT.isoformat()
    assert decisions.decide(run, probes(artifacts, skip={('26b-a4b', 'vulkan')}), 'operator', lambda: AT) == decision
    with pytest.raises(CampaignFault, match='decision_changed'):
        decisions.decide(run, probes(artifacts), 'operator', lambda: AT)


def test_an_export_changed_after_its_receipt_is_not_published(tmp_path):
    run, artifacts = campaign(tmp_path)
    (tmp_path / 'exports' / 'e4b' / 'mlx' / 'payload.bin').write_bytes(b'replaced')
    decision = decisions.decide(run, probes(artifacts), 'operator', lambda: AT)
    assert decision['tiers']['e4b']['failed'] == ['export_mlx_changed']
    (tmp_path / 'exports' / '12b' / 'export_receipt.json').write_text('{}')
    assert decisions.export_failures(artifacts['12b']) == ['export_receipt_changed']
    assert decisions.export_failures(None) == ['no_artifact']


def test_a_probe_of_another_export_does_not_count(tmp_path):
    run, artifacts = campaign(tmp_path)
    stale = [dict(probe, export_receipt_sha256='0' * 64) if probe['student'] == 'e4b' else probe for probe in probes(artifacts)]
    assert decisions.decide(run, stale, 'operator', lambda: AT)['tiers']['e4b']['failed'] == ['load_probe_mlx', 'load_probe_vulkan']


def test_a_rehearsal_cannot_be_decided_and_a_decision_needs_a_decider(tmp_path):
    run, artifacts = campaign(tmp_path, profile='dry-run')
    with pytest.raises(CampaignFault, match='decision_needs_full_campaign'):
        decisions.decide(run, probes(artifacts), 'operator')
    full, artifacts = campaign(tmp_path / 'full')
    with pytest.raises(CampaignFault, match='decision_needs_a_decider'):
        decisions.decide(full, probes(artifacts), '')


def test_the_ledger_records_published_bundles_and_why_the_rest_were_not(tmp_path):
    run, artifacts = campaign(tmp_path, {'26b-a4b': {'behavioral_status': 'behavioral_failure'}})
    decisions.decide(run, probes(artifacts), 'operator', lambda: AT)
    bundle = {'revision': 'r' * 40, 'manifest_sha256': 'm' * 64}
    published = {'e4b': {'mlx': bundle, 'vulkan': bundle}, '12b': {'mlx': bundle, 'vulkan': bundle}}
    with pytest.raises(CampaignFault, match='publication_incomplete'):
        decisions.record(run, tmp_path / 'ledger', 'campaign-1', {'e4b': {'mlx': bundle}, '12b': published['12b']})
    with pytest.raises(CampaignFault, match='publication_not_decided'):
        decisions.record(run, tmp_path / 'ledger', 'campaign-1', {**published, '26b-a4b': published['e4b']})
    entry = decisions.record(run, tmp_path / 'ledger', 'campaign-1', published)
    assert entry['unpublished'] == {'26b-a4b': ['behavioral_behavioral_failure']}
    assert entry['published']['e4b']['bundles']['vulkan'] == bundle
    assert json.loads((tmp_path / 'ledger' / 'students-campaign-1.json').read_text()) == entry
    with pytest.raises(CampaignFault, match='release_ledger_changed'):
        decisions.record(run, tmp_path / 'ledger', 'campaign-1', {**published, 'e4b': {'mlx': bundle, 'vulkan': dict(bundle, revision='s' * 40)}})
