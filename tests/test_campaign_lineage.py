import pytest
from conftest import register_bank

from bialy.campaign import ancestry
from bialy.campaign.journal import CampaignFault, Journal
from bialy.campaign.lineage import FAMILIES, Lineage


def bank():
    return [{"id": partition + family, "family": family, "partition": partition, "source_class": "synthetic"}
            for partition in ancestry.PARTITIONS for family in FAMILIES]


def test_descendants_cannot_change_family_partition_or_source(tmp_path):
    journal = Journal(tmp_path, {})
    lineage = Lineage(journal)
    register_bank(lineage, bank())
    identity = lineage.freeze()
    child = lineage.descendant("worker", "system_test" + FAMILIES[0], {"partition": "train", "source_class": "private_dev"})
    assert child["partition"] == "system_test" and child["family"] == FAMILIES[0]
    assert child["source_class"] == "synthetic"
    assert lineage.freeze() == identity
    with pytest.raises(CampaignFault, match="lineage_mismatch"):
        lineage.admit("worker", "train", child["spec_hash"])
    with pytest.raises(CampaignFault, match="bank_frozen"):
        register_bank(lineage, bank())
    journal.close()


def test_a_root_takes_its_partition_from_the_assignment_and_may_not_declare_one(tmp_path):
    journal = Journal(tmp_path, {})
    lineage = Lineage(journal)
    tasks = bank()
    assignment = register_bank(Lineage(Journal(tmp_path / "other", {})), tasks)
    with pytest.raises(CampaignFault, match="task_declares_partition"):
        lineage.roots(tasks, assignment)
    stripped = [{k: v for k, v in task.items() if k != "partition"} for task in tasks]
    lineage.roots(stripped[:4], assignment)
    assert lineage.task(stripped[0]["id"])["partition"] == assignment.partition(ancestry.task_node(stripped[0]["id"]))
    with pytest.raises(CampaignFault, match="node_outside_ancestry"):
        lineage.roots([dict(stripped[0], id="unplanned")], assignment)
    journal.close()


def test_a_bank_registered_under_one_assignment_refuses_another(tmp_path):
    journal = Journal(tmp_path, {})
    lineage = Lineage(journal)
    first = bank()
    register_bank(lineage, first[:4])
    with pytest.raises(CampaignFault, match="ancestry_assignment_changed"):
        register_bank(lineage, first[4:8])
    journal.close()


def test_the_freeze_binds_the_assignment(tmp_path):
    def frozen(path, tasks):
        journal = Journal(path, {})
        lineage = Lineage(journal)
        register_bank(lineage, tasks)
        identity = lineage.freeze()
        journal.close()
        return identity
    tasks = bank()
    moved = [dict(t, partition="validation") if t["id"] == "train" + FAMILIES[0] else t for t in tasks]
    moved = [dict(t, partition="train") if t["id"] == "validation" + FAMILIES[0] else t for t in moved]
    assert frozen(tmp_path / "a", tasks) == frozen(tmp_path / "b", tasks)
    assert frozen(tmp_path / "a2", tasks) != frozen(tmp_path / "c", moved)


def test_a_private_source_never_enters_the_ancestry_or_the_bank(tmp_path):
    journal = Journal(tmp_path, {})
    lineage = Lineage(journal)
    tasks = bank()
    tasks[-1]["source_class"] = "private_dev"
    with pytest.raises(CampaignFault, match="source_not_permitted"):
        ancestry.Graph().add(ancestry.task_node(tasks[-1]["id"]), "task", source_class="private_dev")
    assignment = register_bank(Lineage(Journal(tmp_path / "other", {})), [dict(t, source_class="synthetic") for t in tasks])
    with pytest.raises(CampaignFault, match="source_not_permitted"):
        lineage.roots([{k: v for k, v in t.items() if k != "partition"} for t in tasks], assignment)
    assert journal.db.execute("SELECT count(*) FROM tasks").fetchone()[0] == 0
    with pytest.raises(CampaignFault, match="empty_task_bank"):
        lineage.freeze()
    journal.close()
