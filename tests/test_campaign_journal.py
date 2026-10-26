import pytest

from bialy.campaign.journal import CampaignFault, Journal


def test_resume_reconciles_same_attempt_and_rejects_changed_inputs(tmp_path):
    journal = Journal(tmp_path, {"students": ["e4b", "12b"]})
    first = journal.begin("collection", {"plan": "frozen"})
    journal.close()
    journal = Journal(tmp_path, {"students": ["e4b", "12b"]})
    assert journal.begin("collection", {"plan": "frozen"}) == first
    with pytest.raises(CampaignFault, match="stage_inputs_changed"):
        journal.begin("collection", {"plan": "changed"})
    journal.settle("collection", first["attempt"], "passed", output={"digest": "bytes"})
    assert journal.begin("collection", {"plan": "frozen"})["status"] == "passed"
    journal.close()
    with pytest.raises(CampaignFault, match="campaign_identity_changed"):
        Journal(tmp_path, {"students": ["e4b"]})


def test_only_transient_failures_get_two_retries(tmp_path):
    journal = Journal(tmp_path, {})
    for count in range(1, 4):
        attempt = journal.begin("download", {})
        assert attempt["tries"] == count
        journal.settle("download", attempt["attempt"], "failed", fault=CampaignFault("transport", retryable=True).record())
    assert journal.begin("download", {})["status"] == "failed"
    attempt = journal.begin("training", {})
    journal.settle("training", attempt["attempt"], "failed", fault=CampaignFault("out_of_memory").record())
    assert journal.begin("training", {})["tries"] == 1
    journal.close()


def test_one_owner_and_no_duplicate_accounting(tmp_path):
    journal = Journal(tmp_path, {})
    other = Journal(tmp_path, {})
    with journal.ownership():
        with pytest.raises(CampaignFault, match="already_owned"), other.ownership():
            pytest.fail("second controller acquired ownership")
    with other.ownership():
        pass
    journal.usage("request-1", {"tokens": 100, "charge": None})
    journal.usage("request-1", {"tokens": 100, "charge": None})
    with pytest.raises(CampaignFault, match="usage_identity_changed"):
        journal.usage("request-1", {"tokens": 200})
    assert len(journal.snapshot()["usage"]) == 1
    journal.close()
    other.close()
