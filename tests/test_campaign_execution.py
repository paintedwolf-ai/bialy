import pytest

from bialy.campaign.execution import Executor, ReceiptBackend
from bialy.campaign.journal import CampaignFault, Journal


class Handler:
    def __init__(self):
        self.calls = 0
        self.output = {'artifact': 'bound'}

    def execute(self, inputs, attempt):
        self.calls += 1
        return self.output

    def reconcile(self, inputs, attempt):
        raise CampaignFault('execution_outcome_unknown')

    def validate(self, inputs, output):
        if output != self.output:
            raise CampaignFault('artifact_changed')


def test_crash_after_paid_work_uses_receipt_without_repeating_work(tmp_path):
    journal = Journal(tmp_path, {})
    handler = Handler()
    backend = ReceiptBackend(tmp_path / 'receipts', {'collect': handler})
    current = journal.begin('collect', {})
    backend.execute('collect', {}, current['attempt'])
    result = Executor(journal, backend).run('collect', {})
    assert result['status'] == 'passed' and handler.calls == 1
    assert Executor(journal, backend).run('collect', {})['status'] == 'passed'
    assert handler.calls == 1
    journal.close()


def test_unknown_execution_is_not_repeated(tmp_path):
    journal = Journal(tmp_path, {})
    handler = Handler()
    backend = ReceiptBackend(tmp_path / 'receipts', {'collect': handler})
    journal.begin('collect', {})
    result = Executor(journal, backend).run('collect', {})
    assert result['fault']['code'] == 'execution_outcome_unknown' and handler.calls == 0
    journal.close()


def test_successful_stage_revalidates_artifacts_on_restart(tmp_path):
    journal = Journal(tmp_path, {})
    handler = Handler()
    backend = ReceiptBackend(tmp_path / 'receipts', {'collect': handler})
    executor = Executor(journal, backend)
    executor.run('collect', {})
    handler.output = {'artifact': 'changed'}
    with pytest.raises(CampaignFault, match='artifact_changed'):
        executor.run('collect', {})
    journal.close()
