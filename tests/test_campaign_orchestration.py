from campaign_backend import Backend, profile

from bialy.campaign.journal import CampaignFault, Journal, StageWaiting
from bialy.campaign.orchestration import BOUNDARY, Campaign
from bialy.student_host.specification import students


def campaign(journal, backend, name='pilot', roster=None, **options):
    return Campaign(journal, backend, lambda _: None, profile(name), roster or students(), **options)


def test_every_profile_runs_the_same_student_stages(tmp_path):
    seen = {}
    for name in ('dry-run', 'pilot', 'full'):
        journal = Journal(tmp_path / name, {})
        backend = Backend()
        campaign(journal, backend, name, ['e4b']).run()
        seen[name] = [stage for stage in backend.names() if stage not in (
            'head_rows', 'selector_training', 'selector_qualification', 'host_adoption', 'student_host_qualification',
            'selector_supplied', 'compare_e4b')]
        journal.close()
    assert seen['dry-run'] == seen['pilot'] == seen['full']


def test_a_pilot_without_a_supplied_policy_runs_with_the_engine_disabled(tmp_path):
    journal = Journal(tmp_path, {})
    backend = Backend(outputs={'selector_supplied': {'qualified': False}})
    result = campaign(journal, backend).run()
    assert result['status'] == 'qualified' and result['selector'] == 'engine_disabled' and result['fallback'] is True
    names = backend.names()
    assert all(names.index('dataset') < index for index, name in enumerate(names) if name.startswith(('baseline_', 'train_')))
    assert backend.inputs('collection')['selector'] == {'qualified': True, 'configuration': 'reference'}
    assert result['execution_status'] == 'complete' and result['data_status'] == 'data_ready'
    for student in students():
        tier = result['students'][student]
        assert (tier['execution_status'], tier['artifact_status'], tier['behavioral_status']) == ('complete', 'exported', 'pass')
    calls = list(backend.calls)
    assert campaign(journal, backend).run() == result and backend.calls == calls
    journal.close()


def test_the_full_profile_stops_at_the_adoption_boundary_and_resumes_on_the_adopted_host(tmp_path):
    journal = Journal(tmp_path, {})
    class Boundary(Backend):
        adopted = False
        def execute(self, name, inputs, attempt):
            if name == 'host_adoption' and not self.adopted:
                self.calls.append((name, inputs))
                raise StageWaiting(BOUNDARY, {'policy': 'p0', 'command': 'resume'})
            return super().execute(name, inputs, attempt)
    backend = Boundary()
    first = campaign(journal, backend, 'full').run()
    assert first['status'] == BOUNDARY and first['waiting']['stage'] == 'host_adoption'
    assert 'collection' not in backend.names() and not (tmp_path / 'result.json').exists()
    backend.adopted = True
    result = campaign(journal, backend, 'full').run()
    assert result['status'] == 'qualified' and result['selector'] == 'p0' and result['promotion_qualified'] is True
    assert backend.inputs('collection')['host']['stage'] == 'student_host_qualification'
    assert backend.inputs('collection')['selector']['policy'] == 'p0'
    assert backend.names().count('head_rows') == 1
    journal.close()


def test_an_unqualified_selector_ends_the_full_profile_with_no_boundary(tmp_path):
    journal = Journal(tmp_path, {})
    backend = Backend(outputs={'selector_qualification': {'qualified': False}})
    result = campaign(journal, backend, 'full').run()
    assert result['status'] == 'selector_unqualified' and 'host_adoption' not in backend.names()
    journal.close()


def test_one_student_failure_does_not_cancel_the_other_or_erase_base_results(tmp_path):
    journal = Journal(tmp_path, {})
    backend = Backend({'train_e4b'})
    result = campaign(journal, backend).run()
    assert result['status'] == 'no_go' and result['execution_status'] == 'incomplete'
    failed, other = result['students']['e4b'], result['students']['12b']
    assert (failed['execution_status'], failed['artifact_status'], failed['behavioral_status']) == (
        'training_incomplete', 'absent', 'not_measured')
    assert failed['base']['status'] == 'passed'
    assert (other['execution_status'], other['artifact_status'], other['behavioral_status']) == ('complete', 'exported', 'pass')
    journal.close()


def test_an_unjudged_rehearsal_reports_mechanics_only(tmp_path):
    journal = Journal(tmp_path, {})
    backend = Backend()
    result = campaign(journal, backend, 'dry-run', ['e4b']).run()
    assert result['status'] == 'mechanical_pipeline_complete' and result['dry_run'] is True
    assert 'compare_e4b' not in backend.names() and result['students']['e4b']['behavioral_status'] == 'not_measured'
    assert result['promotion_qualified'] is False and backend.names()[-1] == 'cleanup'
    journal.close()


def test_unqualified_host_never_starts_collection_or_model_jobs(tmp_path):
    journal = Journal(tmp_path, {})
    backend = Backend({'host_qualification'})
    result = campaign(journal, backend).run()
    assert result['status'] == 'dependency_unqualified' and result['data_status'] == 'not_reached'
    assert backend.names() == ['host_qualification']
    assert all(tier['execution_status'] == 'not_executed' for tier in result['students'].values())
    journal.close()


def test_a_behavioral_failure_in_one_tier_keeps_every_tiers_artifact(tmp_path):
    journal = Journal(tmp_path, {})
    result = campaign(journal, Backend(outputs={'compare_12b': {'status': 'behavioral_failure'}})).run()
    assert result['status'] == 'no_go' and result['execution_status'] == 'complete'
    assert result['students']['12b']['behavioral_status'] == 'behavioral_failure'
    assert all(result['students'][student]['artifact'] == {'identity': 'export_' + student} for student in students())
    assert result['students']['e4b']['behavioral_status'] == 'pass'
    journal.close()


def test_incomplete_data_is_reported_as_the_data_status(tmp_path):
    class Short(Backend):
        def execute(self, name, inputs, attempt):
            if name == 'dataset':
                self.calls.append((name, inputs))
                raise CampaignFault('data_incomplete', details={'deficits': {}})
            return super().execute(name, inputs, attempt)
    journal = Journal(tmp_path, {})
    result = campaign(journal, Short()).run()
    assert result['status'] == 'insufficient_qualified_data' and result['data_status'] == 'data_incomplete'
    journal.close()
