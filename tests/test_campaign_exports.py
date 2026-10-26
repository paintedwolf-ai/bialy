import json
from types import SimpleNamespace

import pytest

from bialy.campaign.exports import Exports
from bialy.campaign.journal import CampaignFault
from bialy.provenance import file_sha256


def test_paired_export_resumes_review_receipts_and_rejects_changed_source(tmp_path, monkeypatch):
    evidence = tmp_path / 'episode'
    evidence.mkdir()
    (evidence / 'host-evidence.json').write_text(json.dumps({'sessions': [{'id': 'session', 'messages': []}]}))
    source = evidence / 'store.db'
    source.write_text('closed store')
    students = {name: {'provider': name, 'kind': 'llamacpp', 'model': name, 'context_window': 65536}
                for name in ('e4b', '12b')}
    identity = {'host_commit': 'host', 'epoch': 'epoch', 'students': students, 'tool_surface': 'command'}
    calls, reviews = [], []
    def process(argv, **kwargs):
        calls.append(argv)
        out = tmp_path / argv[argv.index('--out') + 1]
        out.mkdir()
        binding = json.loads((tmp_path / argv[argv.index('--binding') + 1]).read_text())
        rows = out / 'sft.receipts.jsonl'
        rows.write_text(json.dumps({'session': 'session', 'student': binding['student_model'],
                                   'validation': {'receipt_id': 'receipt'}, 'reviewed': '--quality' in argv}) + '\n')
        (out / 'manifest.json').write_text(json.dumps({'schema': 'pw-receipt-export/1', 'identity': binding,
                                                     'inputs': {str(source): file_sha256(source)}, 'rows_sha256': file_sha256(rows)}))
    def judge(factory, rows, transcript, destination):
        row = json.loads(rows.read_text())
        assert row['meta']['model'] == 'teacher'
        assert json.loads(transcript.read_text())['session_id'] == 'session'
        reviews.append(row)
        destination.write_text(json.dumps({'verdict': 'pass'}) + '\n')
    monkeypatch.setattr('bialy.campaign.exports.custom_quality.judge_rows', judge)
    monkeypatch.setattr('bialy.campaign.exports.training_eligible', lambda row: row['reviewed'])
    factory = SimpleNamespace(reviewer=lambda: SimpleNamespace(id='reviewer', revision='pinned'))
    exporter = Exports(factory, tmp_path, identity, 'catalog', process)
    inputs, output = {'actor': {'id': 'teacher'}}, {'directory': str(evidence), 'fleet': str(tmp_path / 'fleet')}
    result = exporter(inputs, output)
    assert {name: item['rows'] for name, item in result.items()} == {'e4b': 1, '12b': 1}
    assert exporter(inputs, output) == result
    assert len(calls) == 4 and len(reviews) == 2
    source.write_text('mutated store')
    with pytest.raises(CampaignFault, match='host_export_input_changed'):
        exporter(inputs, output)
