from argparse import Namespace

import pytest
import yaml
from campaign_world import Stage, World, adopted_files, git, outputs

from bialy.campaign import adoption, profiles, runtime, selector
from bialy.campaign.handlers import Adoption
from bialy.campaign.journal import CampaignFault

STUDENTS = ['e4b', '12b', '26b-a4b']
GOOD, BAD = 'c' * 40, 'd' * 40


@pytest.fixture(autouse=True)
def bank_sized_profiles(tmp_path, monkeypatch):
    """The full profile on the pilot quota, which the current scenario bank satisfies."""
    document = yaml.safe_load(profiles.CONFIG.read_text())
    document['profiles']['full']['quota'] = 'pilot'
    path = tmp_path / 'profiles.yaml'
    path.write_text(yaml.safe_dump(document))
    monkeypatch.setattr(profiles, 'CONFIG', path)


def test_the_full_quota_is_refused_with_the_banks_exact_deficit(tmp_path, monkeypatch):
    monkeypatch.undo()
    with pytest.raises(CampaignFault, match='coverage_short') as raised:
        runtime.plan(configuration(tmp_path), tmp_path / 'plan')
    assert raised.value.details['deficits']


def configuration(tmp_path, profile='full'):
    value = {'profile': profile, 'epoch': 'campaign', 'checkout': '/host', 'host_commit': 'a' * 40,
             'host_image': 'bialy-runner:campaign-off', 'factory_root': '/factory', 'corpus': '/corpus.json', 'models': '/models',
             'teachers': ['teacher'], 'reviewers': ['first', 'second'], 'server_image': 'sha256:' + 'c' * 64, 'gateway': '192.0.2.1',
             'llama': '/llama', 'llama_revision': 'b' * 40, 'full_training': 'local', 'device': 'cpu', 'mirror': '/mirror'}
    if profile == 'full':
        value['head_rows'] = {'run': 'heads'}
    path = tmp_path / 'campaign.yaml'
    path.write_text(yaml.safe_dump(value))
    return path


def composer(world, values, files):
    process = git(files)

    def compose(journal, directory, body, adopted):
        handlers = {name: Stage(world, name, output) for name, output in values.items()}
        handlers['host_adoption'] = Adoption('/host', adopted, 'bialy campaign resume --run run --host-commit COMMIT', process)
        return handlers
    return compose


def files():
    decisions = selector.load()['decisions']
    return {GOOD: adopted_files(decisions, adoption.RELEASE, 0.55), BAD: adopted_files(decisions, adoption.RELEASE, 0.6)}


def test_a_full_campaign_plans_offline_stops_at_the_boundary_and_finishes_on_the_adopted_host(tmp_path):
    runtime.plan(configuration(tmp_path), tmp_path / 'plan')
    world = World()
    compose = composer(world, outputs(STUDENTS), files())
    first = runtime.drive(tmp_path / 'plan', composer=compose)
    assert first['status'] == 'awaiting_host_adoption' and world.provider_requests == [] and world.rentals == []
    run = tmp_path / 'plan' / 'run'
    assert runtime.status(run)['waiting']['reason'] == 'awaiting_host_adoption'
    refused = runtime.drive(tmp_path / 'plan', {'host_commit': BAD, 'host_image': 'image'}, composer=compose)
    assert refused['status'] == 'awaiting_host_adoption'
    assert refused['waiting']['refused']['details']['differences'] == [
        {'threshold': 'turn.tools.load_at', 'expected': 0.55, 'found': 0.6}]
    assert world.provider_requests == []
    result = runtime.drive(tmp_path / 'plan', {'host_commit': GOOD, 'host_image': 'image'}, composer=compose)
    assert result['status'] == 'qualified' and result['selector'] == 'p0' and result['promotion_qualified'] is True
    assert len(world.provider_requests) == 1 and len(world.rentals) == len(STUDENTS)
    assert all(result['students'][student]['artifact'] == {'gguf': student, 'mlx': student, 'lora_adapter': student}
               for student in STUDENTS)
    report = runtime.report(run)
    assert report['status'] == 'qualified' and report['cleanup']['status'] == 'passed'
    again = runtime.drive(tmp_path / 'plan', composer=compose)
    assert again == result and len(world.provider_requests) == 1


def test_a_larger_tier_fault_keeps_the_smaller_tiers_artifacts_and_the_shared_collection(tmp_path):
    runtime.plan(configuration(tmp_path), tmp_path / 'plan')
    world = World()
    world.fail = {'train_26b-a4b'}
    compose = composer(world, outputs(STUDENTS), files())
    runtime.drive(tmp_path / 'plan', composer=compose)
    result = runtime.drive(tmp_path / 'plan', {'host_commit': GOOD, 'host_image': 'image'}, composer=compose)
    assert result['status'] == 'no_go' and result['data_status'] == 'data_ready'
    assert result['students']['26b-a4b']['execution_status'] == 'training_incomplete'
    assert result['students']['e4b']['artifact_status'] == result['students']['12b']['artifact_status'] == 'exported'
    assert len(world.provider_requests) == 1


def test_a_failed_selector_ends_with_no_policy_and_no_boundary(tmp_path):
    runtime.plan(configuration(tmp_path), tmp_path / 'plan')
    world = World()
    result = runtime.drive(tmp_path / 'plan', composer=composer(world, outputs(STUDENTS, selector_qualified=False), files()))
    assert result['status'] == 'selector_unqualified' and result['selector'] == 'unqualified'
    stages = runtime.status(tmp_path / 'plan' / 'run')['stages']
    assert 'host_adoption' not in stages and 'collection' not in stages
    assert world.provider_requests == [] and world.rentals == []


def test_resume_waits_and_tells_the_operator_what_to_run(tmp_path, monkeypatch, capsys):
    runtime.plan(configuration(tmp_path), tmp_path / 'plan')
    monkeypatch.setattr(runtime, 'compose', composer(World(), outputs(STUDENTS), files()))
    run = tmp_path / 'plan' / 'run'
    assert runtime.main(Namespace(action='run', plan=tmp_path / 'plan')) == 3
    assert runtime.main(Namespace(action='resume', run=run, host_commit=None, host_image=None)) == 3
    assert 'Resume with: bialy campaign resume --run run --host-commit COMMIT' in capsys.readouterr().out
    assert runtime.main(Namespace(action='resume', run=run, host_commit=GOOD, host_image='image')) == 0
