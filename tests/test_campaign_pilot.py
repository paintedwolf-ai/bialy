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
        'collection_pilot', 'baseline_e4b', 'evaluate_base_e4b', 'pilot_e4b', 'export_e4b', 'evaluate_trained_e4b')}


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
