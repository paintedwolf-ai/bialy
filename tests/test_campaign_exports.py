import json

import pytest
from host_bundles import write_episode
from test_training_rows import call, candidate

from bialy.campaign.exports import Exports
from bialy.campaign.journal import CampaignFault
from bialy.provenance import file_sha256


def test_each_student_gets_admitted_reviewed_rows_and_every_admission_record(tmp_path, monkeypatch):
    evidence = tmp_path / 'episode'
    bundle = write_episode(evidence / 'host-bundle', sessions=[{'id': 'root', 'messages': []}])
    students = {name: {'provider': name, 'kind': 'llamacpp', 'model': name, 'context_window': 65536} for name in ('e4b', '12b')}
    identity = {'host_commit': 'host', 'epoch': 'epoch', 'students': students, 'contract_sha256': 'd' * 64}
    calls, reviews = [], []

    def collect(argv, **kwargs):
        calls.append(argv)
        out = tmp_path / argv[argv.index('--out') + 1]
        out.mkdir()
        binding = json.loads((tmp_path / argv[argv.index('--binding') + 1]).read_text())
        rows = [candidate('m1', calls=[call()]), candidate('m2', '2026-11-12T10:01:00Z', calls=[call(call_id='c2', effect_state='unknown')]),
                candidate('m3', '2026-11-12T10:02:00Z', calls=[call(call_id='c3')])]
        (out / 'candidates.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))
        (out / 'manifest.json').write_text(json.dumps({'schema': 'pw-candidate-export/1', 'identity': binding,
                                                     'bundles': {str(bundle): file_sha256(bundle / 'manifest.json')},
                                                     'candidates_sha256': file_sha256(out / 'candidates.jsonl')}))

    class Reviewers:
        def judge(self, row):
            assert row['meta']['model'] == 'teacher' and row['admission']['eligibility'] == 'eligible'
            reviews.append(row)
            return {'verdict': 'pass' if len(reviews) % 2 else 'fail', 'reviewer_slot': '1', 'judge_id': 'reviewer',
                    'judge_revision': 'pinned', 'controls_sha256': 'e' * 64, 'request_sha256': row['request_capture']['sha256']}

    exporter = Exports(Reviewers(), tmp_path, identity, 'catalog', collect)
    inputs = {'actor': {'id': 'teacher'}, 'task': {}}
    output = {'directory': str(evidence), 'bundle': str(bundle), 'receipt': {'checks': []}}
    result = exporter(inputs, output)
    assert {name: item['rows'] for name, item in result.items()} == {'e4b': 1, '12b': 1}
    manifest = json.loads((evidence / 'exports/e4b/rows-manifest.json').read_text())
    assert manifest['schema'] == 'pw-training-rows/1' and manifest['counts'] == {'ordinary': 2, 'excluded:unknown_effects': 1}
    assert manifest['reviewers'] == [{'reviewer_slot': '1', 'judge_id': 'reviewer', 'judge_revision': 'pinned', 'controls_sha256': 'e' * 64}]
    assert not (evidence / 'exports/transcript.jsonl').exists()
    rows = [json.loads(line) for line in (evidence / 'exports/e4b/rows.jsonl').read_text().splitlines()]
    assert len(rows) == 1 and rows[0]['quality']['verdict'] == 'pass' and rows[0]['admission']['category'] == 'ordinary'
    admissions = (evidence / 'exports/e4b/admissions.jsonl').read_text().splitlines()
    assert len(admissions) == 3 and manifest['admissions_sha256'] == file_sha256(evidence / 'exports/e4b/admissions.jsonl')
    assert exporter(inputs, output) == result
    assert len(calls) == 2 and len(reviews) == 8
    manifest = json.loads((bundle / 'manifest.json').read_text())
    (bundle / 'manifest.json').write_text(json.dumps(dict(manifest, store_sha256='9' * 64)))
    with pytest.raises(CampaignFault, match='host_export_input_changed'):
        exporter(inputs, output)
