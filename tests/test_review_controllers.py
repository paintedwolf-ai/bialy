from campaign_backend import Backend, profile

from bialy.campaign.journal import CampaignFault, Journal
from bialy.campaign.orchestration import Campaign


def campaign(journal, backend, review=True, resume=None):
    return Campaign(journal, backend, lambda _: None, profile('pilot'), ['e4b'], review=review, resume=resume)


def test_the_campaign_waits_at_each_review_and_passes_decisions_on(tmp_path):
    journal = Journal(tmp_path, {})
    backend = Backend(waiting={'review_collection'})
    first = campaign(journal, backend, resume='resume').run()
    assert first['status'] == 'waiting' and first['waiting']['stage'] == 'review_collection'
    assert first['waiting']['resume'] == 'resume' and not (tmp_path / 'result.json').exists()
    backend.waiting = {'review_evaluation'}
    assert campaign(journal, backend).run()['waiting']['stage'] == 'review_evaluation'
    backend.waiting = set()
    result = campaign(journal, backend).run()
    assert result['status'] == 'qualified' and not (tmp_path / 'waiting.json').exists()
    names = backend.names()
    assert names.index('review_collection') < names.index('dataset') < names.index('train_e4b')
    assert names.index('review_evaluation') < names.index('compare_e4b')
    assert backend.inputs('dataset')['review']['queue'] == 'review_collection'
    assert backend.inputs('compare_e4b')['review']['queue'] == 'review_evaluation'
    assert names.count('collection') == 1
    journal.close()


def test_a_campaign_without_review_passes_no_decisions(tmp_path):
    journal = Journal(tmp_path, {})
    backend = Backend()
    assert campaign(journal, backend, review=False).run()['status'] == 'qualified'
    assert not any(name.startswith('review_') for name in backend.names())
    assert backend.inputs('compare_e4b')['review'] is None
    journal.close()


def test_a_failed_review_gate_stops_before_assembly(tmp_path):
    class Failing(Backend):
        def execute(self, name, inputs, attempt):
            if name == 'review_collection':
                raise CampaignFault('review_decisions_changed')
            return super().execute(name, inputs, attempt)
    journal = Journal(tmp_path, {})
    backend = Failing()
    assert campaign(journal, backend).run()['status'] == 'insufficient_qualified_data'
    assert 'dataset' not in backend.names()
    journal.close()


def test_a_failed_collection_stops_before_its_review(tmp_path):
    journal = Journal(tmp_path, {})
    backend = Backend({'collection'})
    assert campaign(journal, backend).run()['status'] == 'insufficient_qualified_data'
    assert not any(name.startswith('review_') for name in backend.names())
    journal.close()
