import json
from types import SimpleNamespace

import pytest

from bialy.campaign.journal import CampaignFault
from bialy.campaign.preflight import COMMAND_COLUMNS, command_schema, inspect, main


def schema():
    return '\n'.join('CREATE TABLE ' + table + '(' + ','.join(column + ' TEXT' for column in sorted(columns)) + ');'
                     for table, columns in COMMAND_COLUMNS.items())


def test_schema_checks_actual_columns_and_rejects_comments_as_evidence():
    assert command_schema(schema())['missing_columns'] == {}
    result = command_schema('-- invocation_programs invocation_program_executions\nCREATE TABLE unrelated(id TEXT);')
    assert set(result['missing_columns']) == set(COMMAND_COLUMNS)
    result = command_schema(schema().replace('effect_certainty TEXT,', ''))
    assert result['missing_columns'] == {'invocation_program_executions': ['effect_certainty']}
    with pytest.raises(CampaignFault, match='host_schema_unreadable'):
        command_schema('invalid SQL')


def test_inspection_requires_clean_pinned_source_before_accepting_schema(tmp_path):
    path = tmp_path / 'lycaon/internal/db/schema.sql'
    path.parent.mkdir(parents=True)
    path.write_text(schema())
    def process(args, **kwargs):
        return SimpleNamespace(stdout='pinned\n' if 'rev-parse' in args else '')
    assert inspect(tmp_path, 'pinned', process)['missing_columns'] == {}
    with pytest.raises(CampaignFault, match='host_source_revision_mismatch'):
        inspect(tmp_path, 'other', process)
    with pytest.raises(CampaignFault, match='host_source_not_clean'):
        inspect(tmp_path, 'pinned', lambda *args, **kwargs: SimpleNamespace(stdout='pinned'))
    path.write_text('CREATE TABLE unrelated(id TEXT);')
    with pytest.raises(CampaignFault, match='host_command_receipts_unavailable') as failure:
        inspect(tmp_path, 'pinned', process)
    assert set(failure.value.details['missing_columns']) == set(COMMAND_COLUMNS)
    legacy = inspect(tmp_path, 'pinned', process, tool_surface='legacy')
    assert legacy['tool_surface'] == 'legacy'
    assert legacy['missing_columns']
    with pytest.raises(CampaignFault, match='host_tool_surface_unknown'):
        inspect(tmp_path, 'pinned', process, tool_surface='unknown')


def test_preflight_retains_terminal_failure_and_never_overwrites_run(tmp_path, monkeypatch):
    def unavailable(*args, **kwargs):
        raise CampaignFault('host_command_receipts_unavailable')
    monkeypatch.setattr('bialy.campaign.preflight.inspect', unavailable)
    out = tmp_path / 'attempt'
    args = ['--host-checkout', str(tmp_path), '--host-commit', 'pinned', '--out', str(out)]
    assert main(args) == 1
    result = json.loads((out / 'result.json').read_text())
    assert result['status'] == 'blocked'
    assert not result['collection_started'] and not result['training_started'] and not result['live_host_qualified']
    with pytest.raises(FileExistsError):
        main(args)


def test_preflight_source_success_does_not_claim_live_qualification(tmp_path, monkeypatch):
    monkeypatch.setattr('bialy.campaign.preflight.inspect', lambda *args, **kwargs: {'host_commit': 'pinned'})
    out = tmp_path / 'attempt'
    assert main(['--host-checkout', str(tmp_path), '--host-commit', 'pinned', '--out', str(out)]) == 0
    result = json.loads((out / 'result.json').read_text())
    assert result['status'] == 'source_prerequisites_passed' and not result['live_host_qualified']


def test_episode_source_failure_precedes_container_or_provider_calls(tmp_path, monkeypatch):
    from bialy.campaign.driver import EpisodeDriver
    def unavailable(*args, **kwargs):
        raise CampaignFault('host_command_receipts_unavailable')
    monkeypatch.setattr('bialy.campaign.driver.inspect', unavailable)
    run = SimpleNamespace(checkout=tmp_path, setting=lambda key: {'host_commit': 'pinned', 'tool_surface': 'command-v1'})
    driver = EpisodeDriver(SimpleNamespace(directory=tmp_path), None, run, None, tmp_path,
                           process=lambda *args, **kwargs: pytest.fail('process started before source qualification'))
    with pytest.raises(CampaignFault, match='host_command_receipts_unavailable'):
        driver.execute('trial', {}, 'attempt')
    assert not (tmp_path / 'trials').exists()
