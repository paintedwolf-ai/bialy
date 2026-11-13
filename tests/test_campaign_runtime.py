import json
import socket
from argparse import Namespace
from types import SimpleNamespace

import pytest
import yaml

from bialy import run as runmod
from bialy.campaign import adoption, configuration, profiles, runtime
from bialy.campaign.dataset import select
from bialy.campaign.handlers import Adoption, Cleanup, HeadRows, SuppliedSelector
from bialy.campaign.journal import CampaignFault, Journal, StageWaiting
from bialy.campaign.orchestration import BOUNDARY

POLICY = {'identity': 'p0', 'decision_contract': 3, 'heads': {'turn-load': {'sha256': 'a' * 64}, 'unit-rank': {'sha256': 'b' * 64}},
          'thresholds': {'turn.tools.load_at': 0.55, 'request.max_loads': 3}}
RELEASE = {'release': 'r1', 'heads': {'turn-load': {'sha256': 'a' * 64}, 'unit-rank': {'sha256': 'b' * 64}}}
DECISIONS = {'version': 3, 'turn': {'tools': {'load_at': 0.55}}, 'request': {'max_loads': 3}}


def settings(**changes):
    value = {'profile': 'dry-run', 'epoch': 'rehearsal', 'checkout': '/host', 'host_commit': 'a' * 40,
             'host_image': 'bialy-runner:rehearsal-off', 'factory_root': '/factory', 'corpus': '/corpus.json', 'models': '/models',
             'teachers': ['teacher'], 'reviewers': ['first', 'second'], 'server_image': 'sha256:' + 'c' * 64, 'gateway': '192.0.2.1',
             'llama': '/llama', 'llama_revision': 'b' * 40, 'full_training': 'local', 'device': 'cpu', 'mirror': '/mirror'}
    value.update(changes)
    return {key: item for key, item in value.items() if item is not None}


def written(tmp_path, **changes):
    path = tmp_path / 'campaign.yaml'
    path.write_text(yaml.safe_dump(settings(**changes)))
    return path


def offline(monkeypatch):
    def refuse(*args, **kwargs):
        raise OSError('network unavailable')
    monkeypatch.setattr(socket, 'socket', refuse)
    monkeypatch.setattr(socket, 'create_connection', refuse)


def test_plan_is_offline_and_writes_an_immutable_plan(tmp_path, monkeypatch):
    offline(monkeypatch)
    body = runtime.plan(written(tmp_path), tmp_path / 'plan')
    assert body['profile']['name'] == 'dry-run' and body['students'] == ['e4b'] and body['fallback_rate'] == 0.1
    assert (tmp_path / 'plan' / 'plan.json').stat().st_mode & 0o222 == 0
    assert runtime.read_plan(tmp_path / 'plan') == body
    with pytest.raises(CampaignFault, match='campaign_plan_directory_exists'):
        runtime.plan(written(tmp_path), tmp_path / 'plan')
    tasks = tmp_path / 'plan' / 'bank' / 'tasks.json'
    tasks.chmod(0o644)
    tasks.write_text('[]')
    with pytest.raises(CampaignFault, match='campaign_plan_changed'):
        runtime.read_plan(tmp_path / 'plan')


@pytest.mark.parametrize('changes,code', [
    ({'teachers': None}, 'campaign_configuration_keys'), ({'surprise': 1}, 'campaign_configuration_keys'),
    ({'host_commit': 'main'}, 'campaign_configuration_invalid'), ({'students': ['e4b', 'e4b']}, 'campaign_configuration_invalid'),
    ({'selector_policy': '/run'}, 'campaign_configuration_invalid'), ({'full_training': 'remote'}, 'campaign_configuration_invalid'),
    ({'profile': 'full'}, 'campaign_configuration_invalid')])
def test_a_configuration_that_cannot_be_planned_is_refused(tmp_path, changes, code):
    with pytest.raises(CampaignFault, match=code):
        configuration.load(written(tmp_path, **changes), profiles.load('dry-run'))


def test_the_full_profile_needs_its_head_row_pass(tmp_path):
    full = profiles.load('full')
    with pytest.raises(CampaignFault, match='campaign_configuration_invalid'):
        configuration.load(written(tmp_path, profile='full'), full)
    loaded = configuration.load(written(tmp_path, profile='full', head_rows={'run': 'heads'}), full)
    assert loaded['students'] == ['e4b', '12b', '26b-a4b'] and loaded['human_review'] is False


def test_an_unknown_or_inconsistent_profile_is_refused(tmp_path):
    with pytest.raises(CampaignFault, match='campaign_profile_unknown'):
        profiles.load('weekend')
    document = yaml.safe_load(profiles.CONFIG.read_text())
    document['profiles']['pilot']['segments'] = ['selector', 'student']
    path = tmp_path / 'campaign.yaml'
    path.write_text(yaml.safe_dump(document))
    with pytest.raises(CampaignFault, match='campaign_profile_invalid'):
        profiles.load('pilot', path)


def git(files):
    def process(command, **kwargs):
        path = command[-1].split(':', 1)[1]
        body = files.get(path)
        return SimpleNamespace(returncode=0 if body is not None else 128, stdout=body or '', stderr='')
    return process


def test_adoption_accepts_only_a_commit_with_p0s_pins_and_thresholds():
    from bialy.campaign import selector
    decisions_path = selector.load()['decisions']
    same = git({adoption.RELEASE: json.dumps(RELEASE), decisions_path: yaml.safe_dump(DECISIONS)})
    assert adoption.check('/host', 'c' * 40, POLICY, same) == {'host_commit': 'c' * 40, 'policy': 'p0', 'release': 'r1'}
    drifted = dict(DECISIONS, request={'max_loads': 4})
    with pytest.raises(CampaignFault, match='adopted_commit_differs_from_p0') as raised:
        adoption.check('/host', 'c' * 40, POLICY, git({adoption.RELEASE: json.dumps(RELEASE), decisions_path: yaml.safe_dump(drifted)}))
    assert raised.value.details['differences'] == [{'threshold': 'request.max_loads', 'expected': 3, 'found': 4}]
    with pytest.raises(CampaignFault, match='adopted_commit_invalid'):
        adoption.check('/host', 'main', POLICY, same)
    with pytest.raises(CampaignFault, match='adopted_commit_unreadable'):
        adoption.check('/host', 'c' * 40, POLICY, git({}))


def test_the_boundary_waits_and_keeps_waiting_on_a_refused_commit(monkeypatch):
    with pytest.raises(StageWaiting) as waiting:
        Adoption('/host', None, 'resume --host-commit COMMIT').execute({'policy': POLICY}, 'a')
    assert waiting.value.reason == BOUNDARY and waiting.value.details['command'] == 'resume --host-commit COMMIT'
    def refuse(checkout, commit, policy, process):
        raise CampaignFault('adopted_commit_differs_from_p0', details={'differences': []})
    monkeypatch.setattr(adoption, 'check', refuse)
    with pytest.raises(StageWaiting) as refused:
        Adoption('/host', {'host_commit': 'c' * 40, 'host_image': 'image'}, 'resume').execute({'policy': POLICY}, 'a')
    assert refused.value.details['refused']['code'] == 'adopted_commit_differs_from_p0'
    monkeypatch.setattr(adoption, 'check', lambda checkout, commit, policy, process: {'host_commit': commit, 'policy': 'p0'})
    accepted = Adoption('/host', {'host_commit': 'c' * 40, 'host_image': 'image'}, 'resume').execute({'policy': POLICY}, 'a')
    assert accepted == {'host_commit': 'c' * 40, 'policy': 'p0', 'host_image': 'image'}


def test_cleanup_reaps_remote_leases_once(tmp_path):
    assert Cleanup({'full_training': 'local'}).execute({}, 'a') == {'reaped': False, 'reason': 'local_training'}
    remote = tmp_path / 'machine.json'
    remote.write_text('{}')
    calls = []
    output = Cleanup({'full_training': 'remote', 'remote': str(remote)}, reap=lambda path, once: calls.append((path, once))).execute({}, 'a')
    assert output['reaped'] is True and calls == [(str(remote), True)]


def test_the_head_row_pass_reports_its_split_assignment(tmp_path):
    run = SimpleNamespace(name='heads', path=lambda *parts: tmp_path.joinpath(*parts))
    def stages(run, until):
        assert until == 'coderank'
        (tmp_path / 'ancestry.json').write_text(json.dumps({'assignment_sha256': 'e' * 64}))
    handler = HeadRows(run, stages)
    output = handler.execute({}, 'a')
    assert output == {'run': 'heads', 'assignment_sha256': 'e' * 64}
    handler.validate({}, output)
    def failing(run, until):
        raise runmod.RunError('judge failed')
    with pytest.raises(CampaignFault, match='head_rows_failed'):
        HeadRows(run, failing).execute({}, 'a')


def test_a_supplied_selector_is_read_from_the_campaign_that_qualified_it(tmp_path):
    assert SuppliedSelector(None).execute({}, 'a') == {'qualified': False}
    pilot = tmp_path / 'pilot'
    (pilot / 'heads').mkdir(parents=True)
    (pilot / 'overlay' / 'config').mkdir(parents=True)
    source = Journal(tmp_path / 'source', {})
    attempt = source.begin('selector_qualification', {})['attempt']
    source.settle('selector_qualification', attempt, 'passed', output={'qualified': True, 'configuration': {'pilot': str(pilot)}})
    source.close()
    output = SuppliedSelector(tmp_path / 'source').execute({}, 'a')
    assert output['qualified'] is True and output['configuration'] == {'pilot': str(pilot)}


class Stage:
    def __init__(self, name):
        self.name = name

    def execute(self, inputs, attempt):
        if self.name == 'host_qualification':
            return {'conservative_reference': {'qualified': True, 'mode': 'explicit_tool_loading'}}
        return {'stage': self.name}

    reconcile = execute

    def validate(self, inputs, output):
        assert output


def test_a_planned_rehearsal_runs_and_reports(tmp_path, monkeypatch, capsys):
    body = runtime.plan(written(tmp_path), tmp_path / 'plan')
    names = ['host_qualification', 'reviewer_qualification', 'teacher_qualification', 'selector_supplied', 'collection', 'dataset',
             'evaluation_plan', 'baseline_e4b', 'evaluate_base_e4b', 'train_e4b', 'export_e4b', 'evaluate_trained_e4b', 'cleanup']
    def compose(journal, directory, plan, adopted):
        assert plan == body and adopted is None
        return {name: Stage(name) for name in names}
    result = runtime.drive(tmp_path / 'plan', composer=compose)
    assert result['status'] == 'mechanical_pipeline_complete' and result['dry_run'] is True and result['fallback'] is True
    run = tmp_path / 'plan' / 'run'
    status = runtime.status(run)
    assert set(status['stages'].values()) == {'passed'} and status['result']['status'] == 'mechanical_pipeline_complete'
    assert 'journal' not in runtime.report(run)
    monkeypatch.setattr(runtime, 'compose', compose)
    assert runtime.main(Namespace(action='run', plan=tmp_path / 'plan')) == 0
    assert runtime.main(Namespace(action='report', run=run)) == 0
    assert '"profile": "dry-run"' in capsys.readouterr().out
    with pytest.raises(CampaignFault, match='adoption_needs_commit_and_image'):
        runtime.main(Namespace(action='resume', run=run, host_commit='c' * 40, host_image=None))


def test_a_rehearsal_keeps_the_shortest_whole_rows_within_its_token_limit():
    entries = [{'partition': 'train', 'tokens': tokens, 'row': {'n': tokens}} for tokens in (9000, 500, 12000, 700)]
    entries.append({'partition': 'validation', 'tokens': 300, 'row': {'n': 300}})
    chosen = select(entries, {'train': 2, 'validation': 1, 'system_test': 1}, 10240)
    assert [entry['tokens'] for entry in chosen] == [500, 700, 300]
