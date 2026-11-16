import json
from types import SimpleNamespace

import pytest
from host_bundles import Image

from bialy.campaign.journal import CampaignFault
from bialy.campaign.preflight import inspect, main


def git(revision='pinned', dirty=''):
    image = Image(commit='pinned')

    def process(args, **kwargs):
        if args[0] == 'git':
            return SimpleNamespace(stdout=revision + '\n' if 'rev-parse' in args else dirty)
        return image(args, **kwargs)
    return process


def test_inspection_requires_a_clean_pinned_source_and_its_images_contract(tmp_path):
    evidence = inspect(tmp_path, 'pinned', 'runner:off', tmp_path / 'contract', git())
    assert evidence['host_commit'] == 'pinned' and len(evidence['contract_sha256']) == 64 and evidence['sources']
    with pytest.raises(CampaignFault, match='host_source_revision_mismatch'):
        inspect(tmp_path, 'other', 'runner:off', tmp_path / 'other', git())
    with pytest.raises(CampaignFault, match='host_source_not_clean'):
        inspect(tmp_path, 'pinned', 'runner:off', tmp_path / 'dirty', git(dirty=' M file'))
    stale = Image(commit='stale')
    with pytest.raises(CampaignFault, match='host_contract_revision_mismatch') as failure:
        inspect(tmp_path, 'pinned', 'runner:off', tmp_path / 'stale',
                lambda args, **kwargs: SimpleNamespace(stdout='pinned\n' if 'rev-parse' in args else '') if args[0] == 'git' else stale(args, **kwargs))
    assert failure.value.details == {'expected': 'pinned', 'actual': 'stale'}


def test_preflight_retains_terminal_failure_and_never_overwrites_run(tmp_path, monkeypatch):
    def unavailable(*args, **kwargs):
        raise CampaignFault('host_command_receipts_unavailable')
    monkeypatch.setattr('bialy.campaign.preflight.inspect', unavailable)
    out = tmp_path / 'attempt'
    args = ['--host-checkout', str(tmp_path), '--host-commit', 'pinned', '--host-image', 'runner:off', '--out', str(out)]
    assert main(args) == 1
    result = json.loads((out / 'result.json').read_text())
    assert result['status'] == 'blocked'
    assert not result['collection_started'] and not result['training_started'] and not result['live_host_qualified']
    with pytest.raises(FileExistsError):
        main(args)


def test_preflight_source_success_does_not_claim_live_qualification(tmp_path, monkeypatch):
    monkeypatch.setattr('bialy.campaign.preflight.inspect', lambda *args, **kwargs: {'host_commit': 'pinned'})
    out = tmp_path / 'attempt'
    assert main(['--host-checkout', str(tmp_path), '--host-commit', 'pinned', '--host-image', 'runner:off', '--out', str(out)]) == 0
    result = json.loads((out / 'result.json').read_text())
    assert result['status'] == 'source_prerequisites_passed' and not result['live_host_qualified']


def test_episode_source_failure_precedes_container_or_provider_calls(tmp_path, monkeypatch):
    from bialy.campaign.driver import EpisodeDriver
    def unavailable(*args, **kwargs):
        raise CampaignFault('host_command_receipts_unavailable')
    monkeypatch.setattr('bialy.campaign.driver.inspect', unavailable)
    run = SimpleNamespace(checkout=tmp_path, setting=lambda key: {'host_commit': 'pinned'}, path=lambda *parts: tmp_path.joinpath(*parts))
    driver = EpisodeDriver(SimpleNamespace(directory=tmp_path), None, run, None, tmp_path,
                           process=lambda *args, **kwargs: pytest.fail('process started before source qualification'))
    with pytest.raises(CampaignFault, match='host_command_receipts_unavailable'):
        driver.execute('trial', {'host': {'image': 'runner:off'}}, 'attempt')
    assert not (tmp_path / 'trials').exists()
