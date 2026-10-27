import json

from bialy.campaign import pilot as pilot_module
from bialy.campaign.grading import DIMENSIONS
from bialy.campaign.journal import Journal, StageWaiting
from bialy.campaign.orchestration import Campaign
from bialy.campaign.pilot import Pilot


class Handler:
    def __init__(self, output=None):
        self.output, self.calls = output if output is not None else {}, 0

    def execute(self, inputs, attempt):
        self.calls += 1
        return self.output

    reconcile = execute

    def validate(self, inputs, output):
        assert output == self.output


class Gate(Handler):
    """Waits until the test finishes its queue, like a review gate."""
    def __init__(self, output):
        super().__init__(output)
        self.finished = False

    def execute(self, inputs, attempt):
        self.calls += 1
        if not self.finished:
            raise StageWaiting('human_review', {'queue': 'collection_pilot', 'left': 3, 'command': 'bialy review --run here'})
        return self.output


ATTEMPT = {'stage_status': 'passed', 'effective_thinking': False, 'success': False, 'review': 'split',
           'episode_sha256': 'episode', 'unsupported_completion_claims': 1}


def handlers():
    stages = {name: Handler() for name in ('host_qualification', 'reviewer_qualification', 'teacher_baseline',
              'collection_pilot', 'dataset_pilot', 'baseline_e4b', 'evaluate_base_e4b', 'pilot_e4b', 'export_e4b')}
    stages['host_qualification'].output = {'conservative_reference': {'qualified': True}}
    stages['evaluate_base_e4b'].output = {'attempts': [ATTEMPT]}
    stages['evaluate_trained_e4b'] = Handler({'attempts': [ATTEMPT]})
    stages['review_collection_pilot'] = Gate({'decisions': {}})
    stages['review_evaluation'] = Gate({'decisions': {'episode': {'dimensions': dict.fromkeys(DIMENSIONS, 'pass')}}})
    return stages


def test_the_pilot_pauses_at_a_gate_and_resumes_from_the_journal(tmp_path):
    journal = Journal(tmp_path, {'pilot': 'one'})
    stages = handlers()
    pilot = Pilot(journal, stages, {'tasks': []}, ['e4b'], review=True, resume='python -m bialy.campaign.pilot --configuration c')
    first = pilot.run()
    assert first['status'] == 'waiting' and not (tmp_path / 'result.json').exists()
    waiting = json.loads((tmp_path / 'waiting.json').read_text())
    assert waiting == {'schema': 'pw-campaign-waiting/1', 'reason': 'human_review', 'queue': 'collection_pilot', 'left': 3,
                       'command': 'bialy review --run here', 'stage': 'review_collection_pilot',
                       'resume': 'python -m bialy.campaign.pilot --configuration c'}
    assert stages['dataset_pilot'].calls == 0 and journal.stage('review_collection_pilot')['status'] == 'waiting'
    assert pilot.run()['status'] == 'waiting' and stages['collection_pilot'].calls == 1
    stages['review_collection_pilot'].finished = True
    assert pilot.run()['waiting']['stage'] == 'review_evaluation'
    stages['review_evaluation'].finished = True
    result = pilot.run()
    assert result['status'] == 'mechanical_pipeline_complete' and not (tmp_path / 'waiting.json').exists()
    assert result['students']['e4b']['trained_successes'] == 1 and result['students']['e4b']['base_successes'] == 1
    assert all(stage.calls == 1 for name, stage in stages.items() if not name.startswith('review_'))
    journal.close()


def test_without_review_the_pilot_never_runs_a_gate(tmp_path):
    journal = Journal(tmp_path, {'pilot': 'one'})
    stages = handlers()
    result = Pilot(journal, stages, {'tasks': []}, ['e4b']).run()
    assert result['status'] == 'mechanical_pipeline_complete' and result['students']['e4b']['trained_successes'] == 0
    assert stages['review_collection_pilot'].calls == 0 and stages['review_evaluation'].calls == 0
    journal.close()


def test_the_pilot_command_exits_three_while_waiting(tmp_path, monkeypatch, capsys):
    configuration = tmp_path / 'pilot.json'
    configuration.write_text(json.dumps({'directory': str(tmp_path / 'campaign'), 'students': ['e4b'], 'human_review': True}))
    monkeypatch.setattr('bialy.campaign.pilot_runtime.compose', lambda journal, cfg: (handlers(), {'tasks': []}))
    assert pilot_module.main(['--configuration', str(configuration)]) == pilot_module.WAITING
    error = capsys.readouterr().err
    assert 'Waiting on human review: collection_pilot. Run: bialy review --run here' in error
    assert '--configuration ' + str(configuration) in error


def test_the_pilot_refuses_a_non_boolean_review_setting(tmp_path):
    import pytest
    configuration = tmp_path / 'pilot.json'
    configuration.write_text(json.dumps({'directory': str(tmp_path), 'students': ['e4b'], 'human_review': 'yes'}))
    with pytest.raises(SystemExit):
        pilot_module.main(['--configuration', str(configuration)])


class Backend:
    def __init__(self, waiting=()):
        self.waiting, self.calls = set(waiting), []

    def execute(self, name, inputs, attempt):
        self.calls.append((name, inputs))
        if name in self.waiting:
            raise StageWaiting('human_review', {'queue': name, 'left': 1, 'command': 'c'})
        if name == 'host_qualification':
            return {'conservative_reference': {'qualified': True, 'configuration': 'reference'}}
        if name.startswith('review_'):
            return {'decisions': {}, 'queue': name}
        if name.startswith('compare_'):
            return {'status': 'pass'}
        return {'identity': name}

    def validate(self, name, inputs, output):
        assert output


def test_the_campaign_waits_at_each_review_and_passes_decisions_on(tmp_path):
    journal = Journal(tmp_path, {})
    backend = Backend({'review_collection_pilot'})
    first = Campaign(journal, backend, lambda _: None, ['e4b'], review=True, resume='resume').run()
    assert first['status'] == 'waiting' and first['waiting']['stage'] == 'review_collection_pilot'
    assert not (tmp_path / 'result.json').exists()
    backend.waiting = {'review_evaluation'}
    assert Campaign(journal, backend, lambda _: None, ['e4b'], review=True).run()['waiting']['stage'] == 'review_evaluation'
    backend.waiting = set()
    result = Campaign(journal, backend, lambda _: None, ['e4b'], review=True).run()
    assert result['status'] == 'qualified'
    names = [name for name, _ in backend.calls]
    assert names.index('review_collection_full') < names.index('dataset_full') < names.index('train_e4b')
    assert names.index('review_evaluation') < names.index('compare_e4b')
    dataset = next(inputs for name, inputs in backend.calls if name == 'dataset_pilot')
    compare = next(inputs for name, inputs in backend.calls if name == 'compare_e4b')
    assert dataset['review']['queue'] == 'review_collection_pilot' and compare['review']['queue'] == 'review_evaluation'
    journal.close()


def test_a_campaign_without_review_passes_no_decisions(tmp_path):
    journal = Journal(tmp_path, {})
    backend = Backend()
    assert Campaign(journal, backend, lambda _: None, ['e4b']).run()['status'] == 'qualified'
    assert not any(name.startswith('review_') for name, _ in backend.calls)
    assert next(inputs for name, inputs in backend.calls if name == 'compare_e4b')['review'] is None
    journal.close()


def test_a_failed_review_gate_stops_before_assembly(tmp_path):
    from bialy.campaign.journal import CampaignFault
    class Failing(Backend):
        def execute(self, name, inputs, attempt):
            if name == 'review_collection_pilot':
                raise CampaignFault('review_decisions_changed')
            return super().execute(name, inputs, attempt)
    journal = Journal(tmp_path, {})
    backend = Failing()
    assert Campaign(journal, backend, lambda _: None, ['e4b'], review=True).run()['status'] == 'insufficient_qualified_data'
    assert 'dataset_pilot' not in [name for name, _ in backend.calls]
    journal.close()


def test_a_failed_collection_stops_before_its_review(tmp_path):
    from bialy.campaign.journal import CampaignFault
    class Failing(Backend):
        def execute(self, name, inputs, attempt):
            if name == 'collection_pilot':
                raise CampaignFault('collection_episode_incomplete')
            return super().execute(name, inputs, attempt)
    journal = Journal(tmp_path, {})
    backend = Failing()
    assert Campaign(journal, backend, lambda _: None, ['e4b'], review=True).run()['status'] == 'insufficient_qualified_data'
    assert not any(name.startswith('review_') for name, _ in backend.calls)
    journal.close()
