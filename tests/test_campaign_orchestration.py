from bialy.campaign.journal import CampaignFault, Journal
from bialy.campaign.orchestration import Campaign


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
    result = Campaign(journal, backend, lambda _: None).run()
    assert result['status'] == 'qualified' and result['selector'] == 'conservative'
    for student in ('e4b', '12b'):
        assert result['students'][student]['status'] == 'pass'
        inputs = next(inputs for name, inputs in backend.calls if name == 'pilot_' + student)
        assert inputs['selector']['configuration'] == 'reference'
    calls = list(backend.calls)
    assert Campaign(journal, backend, lambda _: None).run() == result
    assert backend.calls == calls
    journal.close()


def test_one_student_failure_does_not_cancel_the_other_or_erase_base_results(tmp_path):
    journal = Journal(tmp_path, {})
    backend = Backend({'train_e4b'})
    result = Campaign(journal, backend, lambda _: None).run()
    assert result['status'] == 'no_go'
    assert result['students']['e4b']['status'] == 'execution_incomplete'
    assert result['students']['e4b']['base']['status'] == 'passed'
    assert result['students']['12b']['status'] == 'pass'
    journal.close()


def test_unqualified_host_never_starts_collection_or_model_jobs(tmp_path):
    journal = Journal(tmp_path, {})
    backend = Backend({'host_qualification'})
    result = Campaign(journal, backend, lambda _: None).run()
    assert result['status'] == 'dependency_unqualified'
    assert [name for name, _ in backend.calls] == ['host_qualification']
    journal.close()
