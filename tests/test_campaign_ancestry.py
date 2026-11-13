import json

import pytest

from bialy.campaign import ancestry
from bialy.campaign.journal import CampaignFault

FRACTIONS = {"train": 0.6, "validation": 0.2, "development": 0.2}


def chain(graph, root, *descendants, **facts):
    """A root task and descendants that each join the one before."""
    previous = graph.add(root, "task", source_class=facts.pop("source_class", "synthetic"), **facts)
    for node in descendants:
        graph.link(previous, graph.add(node, "derived"))
        previous = node
    return root


def test_everything_that_shares_ancestry_lands_in_one_partition():
    g = ancestry.Graph()
    for n in range(40):
        chain(g, "task:t%d" % n, "worker:t%d" % n, "followup:t%d" % n, "encoding:e4b:t%d" % n, "encoding:12b:t%d" % n)
    g.link("task:t0", g.add("template:shared", "template"))
    g.link("task:t1", "template:shared")
    a = ancestry.allocate(g, 7, FRACTIONS)
    for n in range(40):
        family = {a.partition(node) for node in ("task:t%d" % n, "worker:t%d" % n, "followup:t%d" % n, "encoding:e4b:t%d" % n, "encoding:12b:t%d" % n)}
        assert len(family) == 1
    assert a.partition("task:t0") == a.partition("task:t1") == a.partition("template:shared")
    assert a.component("task:t0") == a.component("task:t1")
    assert set(a.counts()) == set(ancestry.PARTITIONS) and sum(a.counts().values()) == 39


def test_allocation_is_deterministic_and_seeded():
    def build(seed):
        g = ancestry.Graph()
        for n in range(60):
            chain(g, "task:t%d" % n)
        return ancestry.allocate(g, seed, FRACTIONS)
    first, again, other = build(7), build(7), build(8)
    assert first.document == again.document and first.sha256 == again.sha256
    assert first.sha256 != other.sha256
    assert {first.partition("task:t%d" % n) for n in range(60)} == {"train", "validation", "development"}


def test_an_explicit_holdout_takes_its_whole_component():
    g = ancestry.Graph()
    repo = g.add("repository:zod", "repository", source_class="permitted_repository", permission_sha256="p", holdout="system_test")
    for n in range(5):
        g.link(chain(g, "task:z%d" % n, "worker:z%d" % n), repo)
    for n in range(20):
        chain(g, "task:t%d" % n)
    a = ancestry.allocate(g, 7, FRACTIONS)
    assert {a.partition(n) for n in ("repository:zod", "task:z0", "worker:z4")} == {"system_test"}
    assert all(a.partition("task:t%d" % n) != "system_test" for n in range(20))


def test_conflicting_holdouts_in_one_component_are_refused():
    g = ancestry.Graph()
    g.link(chain(g, "task:a", holdout="system_test"), chain(g, "task:b", holdout="development"))
    with pytest.raises(CampaignFault, match="holdout_conflict"):
        ancestry.allocate(g, 7, FRACTIONS)


def test_exact_duplicates_across_components_are_refused():
    g = ancestry.Graph()
    chain(g, "task:a", hashes={"prompt": "same"})
    chain(g, "task:b", hashes={"prompt": "same"})
    with pytest.raises(CampaignFault, match="duplicate_across_components"):
        ancestry.allocate(g, 7, FRACTIONS)
    g.link("task:a", "task:b")
    assert ancestry.allocate(g, 7, FRACTIONS).component("task:a") == ancestry.allocate(g, 7, FRACTIONS).component("task:b")


@pytest.mark.parametrize("facts, code", [
    ({"source_class": "private_log"}, "source_not_permitted"),
    ({"source_class": "permitted_repository"}, "repository_permission_missing"),
    ({"holdout": "elsewhere"}, "ancestry_holdout_invalid"),
    ({"hashes": {"embedding": "x"}}, "ancestry_hash_kind_invalid"),
])
def test_inadmissible_nodes_fail_before_any_allocation(facts, code):
    with pytest.raises(CampaignFault, match=code):
        ancestry.Graph().add("task:a", "task", **facts)


def test_a_component_without_a_declared_source_is_refused():
    g = ancestry.Graph()
    g.add("group:orphan", "prompt_group")
    with pytest.raises(CampaignFault, match="ancestry_source_undeclared"):
        ancestry.allocate(g, 7, FRACTIONS)


def test_a_node_declared_twice_must_agree():
    g = ancestry.Graph()
    g.add("task:a", "task", source_class="synthetic")
    g.add("task:a", "task", source_class="synthetic")
    with pytest.raises(CampaignFault, match="ancestry_node_conflict"):
        g.add("task:a", "task", source_class="synthetic", holdout="system_test")
    with pytest.raises(CampaignFault, match="ancestry_edge_unknown_node"):
        g.link("task:a", "task:missing")


def test_minimums_fill_every_partition_per_stratum_and_report_exact_deficits():
    g = ancestry.Graph()
    for focus in ("diagnosis", "repair"):
        for n in range(6):
            chain(g, "task:%s-%d" % (focus, n), stratum=focus)
    minimums = {"train": 2, "validation": 1, "development": 1, "system_test": 1}
    a = ancestry.allocate(g, 7, {"train": 1.0}, minimums)
    for focus in ("diagnosis", "repair"):
        placed = [a.partition("task:%s-%d" % (focus, n)) for n in range(6)]
        assert all(placed.count(p) >= n for p, n in minimums.items())
    g.add("task:lonely", "task", source_class="synthetic", stratum="capture")
    with pytest.raises(CampaignFault) as caught:
        ancestry.allocate(g, 7, {"train": 1.0}, minimums)
    assert caught.value.code == "coverage_short" and caught.value.details == {"deficits": {"capture": 4}}


def test_a_component_spanning_two_strata_is_refused():
    g = ancestry.Graph()
    g.link(chain(g, "task:a", stratum="x"), chain(g, "task:b", stratum="y"))
    with pytest.raises(CampaignFault, match="stratum_conflict"):
        ancestry.allocate(g, 7, FRACTIONS)


def test_fractions_must_name_partitions_and_sum_to_one():
    g = ancestry.Graph()
    chain(g, "task:a")
    for fractions in ({"train": 0.5}, {"train": 0.5, "holdout": 0.5}, {"train": 1.2, "validation": -0.2}):
        with pytest.raises(CampaignFault, match="ancestry_fractions_invalid"):
            ancestry.allocate(g, 7, fractions)


def test_an_assignment_is_written_once_and_refused_when_edited(tmp_path):
    g = ancestry.Graph()
    for n in range(10):
        chain(g, "task:t%d" % n)
    a = ancestry.allocate(g, 7, FRACTIONS)
    path = a.save(tmp_path / "ancestry.json")
    assert ancestry.load(path).document == a.document
    a.save(path)
    chain(g, "task:extra")
    with pytest.raises(CampaignFault, match="ancestry_assignment_changed"):
        ancestry.allocate(g, 7, FRACTIONS).save(path)
    document = json.loads(path.read_text())
    first = next(iter(document["components"]))
    document["components"][first]["partition"] = "system_test"
    path.write_text(json.dumps(document))
    with pytest.raises(CampaignFault, match="ancestry_assignment_changed"):
        ancestry.load(path)
    document["schema"] = "bialy-ancestry/9"
    path.write_text(json.dumps(document))
    with pytest.raises(CampaignFault, match="ancestry_schema_unknown"):
        ancestry.load(path)
    with pytest.raises(CampaignFault, match="node_outside_ancestry"):
        a.partition("task:unknown")
