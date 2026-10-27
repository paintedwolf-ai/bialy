from bialy.campaign.journal import CampaignFault, Journal
from bialy.campaign.pilot import Pilot


class Handler:
    def __init__(self, output=None, fault=None):
        self.output, self.fault, self.calls = output or {}, fault, 0

    def execute(self, inputs, attempt):
        self.calls += 1
        if self.fault:
            raise CampaignFault(self.fault)
        return self.output

    reconcile = execute

    def validate(self, inputs, output):
        assert output == self.output


def handlers():
    return {name: Handler() for name in ('host_qualification', 'reviewer_qualification', 'teacher_baseline',
        'collection_pilot', 'dataset_pilot', 'baseline_e4b', 'evaluate_base_e4b', 'pilot_e4b', 'export_e4b', 'evaluate_trained_e4b')}


def test_pilot_completes_data_path_without_claiming_behavioral_success(tmp_path):
    journal = Journal(tmp_path, {'pilot': 'one'})
    stages = handlers()
    stages['host_qualification'].output = {'conservative_reference': {'qualified': True}}
    for arm in ('base', 'trained'):
        stages['evaluate_' + arm + '_e4b'].output = {'attempts': [
            {'stage_status': 'passed', 'effective_thinking': False, 'success': False}]}
    pilot = Pilot(journal, stages, {'tasks': ['frozen']}, ['e4b'])
    result = pilot.run()
    assert result['status'] == 'mechanical_pipeline_complete'
    assert result['promotion_qualified'] is False
    assert result['students']['e4b']['trained_successes'] == 0
    assert pilot.run()['status'] == result['status']
    assert all(stage.calls == 1 for stage in stages.values())
    journal.close()


def test_failed_stage_stops_later_side_effects_and_retains_failure(tmp_path):
    journal = Journal(tmp_path, {'pilot': 'one'})
    stages = handlers()
    stages['host_qualification'].fault = 'source_unavailable'
    result = Pilot(journal, stages, {}, ['e4b']).run()
    assert result['status'] == 'incomplete'
    assert result['fault']['details']['stage'] == 'host_qualification'
    assert stages['pilot_e4b'].calls == 0
    assert journal.stage('host_qualification')['fault']['code'] == 'source_unavailable'
    journal.close()


def test_pilot_checks_mount_privileges_before_creating_fleet_state(tmp_path, monkeypatch):
    import pytest

    from bialy.campaign.pilot_runtime import prepare_fleet
    monkeypatch.setattr('bialy.campaign.pilot_runtime.os.geteuid', lambda: 1000)
    with pytest.raises(CampaignFault, match='pilot_requires_fleet_mount_privileges'):
        prepare_fleet(tmp_path / 'fleet')
    assert not (tmp_path / 'fleet').exists()
    monkeypatch.setattr('bialy.campaign.pilot_runtime.os.geteuid', lambda: 0)
    prepare_fleet(tmp_path / 'fleet')
    assert (tmp_path / 'fleet/cache').is_dir()
    assert (tmp_path / 'fleet/repos').is_dir()


def test_pilot_composition_uses_only_the_declared_teacher(tmp_path, monkeypatch):
    from bialy.campaign.pilot_runtime import compose
    monkeypatch.setattr('bialy.campaign.pilot_runtime.custom_model_run.binding', lambda run: {})
    corpus = tmp_path / 'corpus.json'
    corpus.write_text('{}')
    cfg = {'students': ['e4b'], 'teachers': ['deepseek-v4.1-flash'], 'reviewers': ['deepseek-v4.1-flash'],
           'factory_root': str(tmp_path / 'factory'), 'checkout': str(tmp_path / 'host'), 'epoch': 'test',
           'host_commit': 'a' * 40, 'tool_surface': 'legacy', 'host_image': 'bialy-runner:test-off',
           'corpus': str(corpus), 'server_image': 'server'}
    journal = Journal(tmp_path / 'journal', {'configuration': cfg})
    stages, _ = compose(journal, cfg)
    assert stages['teacher_baseline'].repetitions == 2
    factory = stages['host_qualification'].run.factory
    assert [model.id for model in factory.generators()] == cfg['teachers']
    assert factory.generators()[0].hosted is not None
    assert factory.reviewer().id == cfg['reviewers'][0]
    journal.close()
