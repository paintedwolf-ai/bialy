from bialy.campaign.journal import CampaignFault, Journal
from bialy.campaign.orchestration import Campaign
from bialy.student_host.specification import students


class Backend:
    def __init__(self, failures=()):
        self.failures, self.calls = set(failures), []
    def execute(self, name, inputs, attempt):
        self.calls.append((name, inputs))
        if name in self.failures:
            raise CampaignFault('fixture_failure')
        if name == 'host_qualification':
            return {'conservative_reference': {'qualified': True, 'configuration': 'reference'}}
        if name == 'selector_qualification':
            return {'qualified': True, 'configuration': {'configuration': 'retrained'}}
        if name.startswith('compare_'):
            return {'status': 'pass'}
        return {'identity': name}
    def validate(self, name, inputs, output):
        assert output


def test_failed_selector_keeps_both_students_on_qualified_reference(tmp_path):
    journal = Journal(tmp_path, {})
    backend = Backend({'selector_training'})
    result = Campaign(journal, backend, lambda _: None, students()).run()
    assert result['status'] == 'qualified' and result['selector'] == 'conservative'
    assert result['execution_status'] == 'complete' and result['data_status'] == 'data_ready'
    for student in students():
        tier = result['students'][student]
        assert (tier['execution_status'], tier['artifact_status'], tier['behavioral_status']) == ('complete', 'exported', 'pass')
        inputs = next(inputs for name, inputs in backend.calls if name == 'pilot_' + student)
        assert inputs['selector']['configuration'] == 'reference'
    calls = list(backend.calls)
    assert Campaign(journal, backend, lambda _: None, students()).run() == result
    assert backend.calls == calls
    journal.close()


def test_one_student_failure_does_not_cancel_the_other_or_erase_base_results(tmp_path):
    journal = Journal(tmp_path, {})
    backend = Backend({'train_e4b'})
    result = Campaign(journal, backend, lambda _: None, students()).run()
    assert result['status'] == 'no_go' and result['execution_status'] == 'incomplete'
    failed, other = result['students']['e4b'], result['students']['12b']
    assert (failed['execution_status'], failed['artifact_status'], failed['behavioral_status']) == (
        'training_incomplete', 'absent', 'not_measured')
    assert failed['base']['status'] == 'passed'
    assert (other['execution_status'], other['artifact_status'], other['behavioral_status']) == ('complete', 'exported', 'pass')
    journal.close()


def test_unqualified_host_never_starts_collection_or_model_jobs(tmp_path):
    journal = Journal(tmp_path, {})
    backend = Backend({'host_qualification'})
    result = Campaign(journal, backend, lambda _: None, students()).run()
    assert result['status'] == 'dependency_unqualified' and result['data_status'] == 'not_reached'
    assert all(tier['execution_status'] == 'not_executed' for tier in result['students'].values())
    assert [name for name, _ in backend.calls] == ['host_qualification']
    journal.close()


def test_a_behavioral_failure_in_one_tier_keeps_every_tiers_artifact(tmp_path):
    class Failing(Backend):
        def execute(self, name, inputs, attempt):
            if name == 'compare_12b':
                self.calls.append((name, inputs))
                return {'status': 'behavioral_failure'}
            return super().execute(name, inputs, attempt)
    journal = Journal(tmp_path, {})
    result = Campaign(journal, Failing(), lambda _: None, students()).run()
    assert result['status'] == 'no_go' and result['execution_status'] == 'complete'
    assert result['students']['12b']['behavioral_status'] == 'behavioral_failure'
    assert all(result['students'][student]['artifact'] == {'identity': 'export_' + student} for student in students())
    assert result['students']['e4b']['behavioral_status'] == 'pass'
    journal.close()


def test_incomplete_data_is_reported_as_the_data_status(tmp_path):
    class Short(Backend):
        def execute(self, name, inputs, attempt):
            if name == 'dataset_pilot':
                self.calls.append((name, inputs))
                raise CampaignFault('data_incomplete', details={'deficits': {}})
            return super().execute(name, inputs, attempt)
    journal = Journal(tmp_path, {})
    result = Campaign(journal, Short(), lambda _: None, students()).run()
    assert result['status'] == 'insufficient_qualified_data' and result['data_status'] == 'data_incomplete'
    journal.close()
