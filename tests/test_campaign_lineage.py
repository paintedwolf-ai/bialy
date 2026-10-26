import pytest

from bialy.campaign.journal import CampaignFault, Journal
from bialy.campaign.lineage import FAMILIES, PARTITIONS, Lineage


def bank():
    return [{"id": partition + family, "family": family, "partition": partition, "source_class": "synthetic"}
            for partition in sorted(PARTITIONS) for family in FAMILIES]


def test_descendants_cannot_change_family_partition_or_source(tmp_path):
    journal = Journal(tmp_path, {})
    lineage = Lineage(journal)
    lineage.roots(bank())
    identity = lineage.freeze()
    child = lineage.descendant("worker", "system_test" + FAMILIES[0], {"partition": "train", "source_class": "private_dev"})
    assert child["partition"] == "system_test" and child["family"] == FAMILIES[0]
    assert child["source_class"] == "synthetic"
    assert lineage.freeze() == identity
    with pytest.raises(CampaignFault, match="lineage_mismatch"):
        lineage.admit("worker", "train", child["spec_hash"])
    with pytest.raises(CampaignFault, match="bank_frozen"):
        lineage.roots(bank())
    journal.close()


def test_mixed_private_bank_rolls_back_whole_admission(tmp_path):
    journal = Journal(tmp_path, {})
    lineage = Lineage(journal)
    tasks = bank()
    tasks[-1]["source_class"] = "private_dev"
    with pytest.raises(CampaignFault, match="source_not_permitted"):
        lineage.roots(tasks)
    assert journal.db.execute("SELECT count(*) FROM tasks").fetchone()[0] == 0
    with pytest.raises(CampaignFault, match="empty_task_bank"):
        lineage.freeze()
    journal.close()
