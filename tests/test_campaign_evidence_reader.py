import json
from types import SimpleNamespace

import pytest

from bialy.campaign.evidence_reader import read
from bialy.campaign.journal import CampaignFault


def test_evidence_export_is_readonly_and_reuses_retained_facts(tmp_path):
    calls = []
    facts = {'version': 1, 'root_session_id': 'session'}
    def process(args, **kwargs):
        calls.append(args)
        assert '--read-only' in args and args[args.index('--network') + 1] == 'none'
        assert str(tmp_path.resolve()) + ':/capture:ro' in args
        return SimpleNamespace(returncode=0, stdout=json.dumps(facts))
    destination = tmp_path / 'host-evidence.json'
    for _ in range(2):
        assert read(tmp_path, tmp_path, 'session', 'sha256:image', destination, process) == facts
    assert len(calls) == 1


def test_reader_failure_preserves_a_private_diagnostic_without_inventing_facts(tmp_path):
    destination = tmp_path / 'host-evidence.json'
    with pytest.raises(CampaignFault, match='host_evidence_export_failed') as fault:
        read(tmp_path, tmp_path, 'session', 'sha256:image', destination,
             lambda *args, **kwargs: SimpleNamespace(returncode=14, stderr='cannot open capture'))
    assert fault.value.details['exit_status'] == 14
    assert not destination.exists()
    diagnostic = tmp_path / 'evidence-reader-failure.json'
    assert json.loads(diagnostic.read_text()) == {'exit_status': 14, 'stderr': 'cannot open capture'}
    assert diagnostic.stat().st_mode & 0o777 == 0o600


@pytest.mark.parametrize('body, code', [('not json', 'invalid_json'), ('[]', 'invalid_shape')])
def test_reader_refuses_malformed_output(tmp_path, body, code):
    with pytest.raises(CampaignFault, match=code):
        read(tmp_path, tmp_path, 'session', 'sha256:image', tmp_path / 'host-evidence.json',
             lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout=body))
