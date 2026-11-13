import json

import pytest

from bialy import split
from bialy.campaign import ancestry
from bialy.campaign.journal import CampaignFault


def task(workspace, n, group=None, scenario=None, prompt=None):
    meta = {"workspace": workspace, "prompt_group": group or "%s:g%d" % (workspace, n)}
    if scenario:
        meta["scenario"] = scenario
    return {"id": "%s-%d" % (workspace, n), "prompt": prompt or "request %s %d" % (workspace, n), "meta": meta}


def bank():
    tasks = [task(w, n) for w in ("flask", "gin", "zod") for n in range(6)]
    tasks += [task(w, n) for w in ("react-vite", "vue-vite") for n in range(30)]
    return tasks


def test_repositories_stay_whole_and_held_out_workspaces_are_system_test(factory):
    a = split.assign(factory, bank())
    for repo in ("flask", "gin", "zod"):
        assert len({a.partition(ancestry.task_node("%s-%d" % (repo, n))) for n in range(6)}) == 1
    assert {a.partition(ancestry.task_node("zod-%d" % n)) for n in range(6)} == {"system_test"}
    assert {a.partition(ancestry.task_node("vue-vite-%d" % n)) for n in range(30)} == {"system_test"}
    stack = {a.partition(ancestry.task_node("react-vite-%d" % n)) for n in range(30)}
    assert stack == {"train", "validation", "development"}
    assert a.partition("repository:flask") == a.partition(ancestry.task_node("flask-0"))


def test_a_scenario_or_prompt_group_joins_tasks_across_workspaces(factory):
    tasks = bank() + [task("react-vite", 90, scenario="s1"), task("node-api", 90, scenario="s1"), task("react-vite", 91, group="shared"),
                      task("node-api", 91, group="shared")]
    a = split.assign(factory, tasks)
    assert a.component(ancestry.task_node("react-vite-90")) == a.component(ancestry.task_node("node-api-90"))
    assert a.partition(ancestry.task_node("react-vite-91")) == a.partition(ancestry.task_node("node-api-91"))


def test_a_scenario_shared_with_a_held_out_workspace_is_held_out(factory):
    a = split.assign(factory, bank() + [task("react-vite", 90, scenario="s1"), task("zod", 90, scenario="s1")])
    assert a.partition(ancestry.task_node("react-vite-90")) == "system_test"


def test_identical_requests_in_unrelated_tasks_are_refused(factory):
    with pytest.raises(CampaignFault, match="duplicate_across_components"):
        split.assign(factory, bank() + [task("react-vite", 90, prompt="same words"), task("node-api", 90, prompt="same words")])


def test_skill_request_families_join_their_repository(factory):
    families = [("family:a", "flask"), ("family:b", "zod"), ("family:c", "react-vite")]
    a = split.assign(factory, bank(), families)
    assert a.partition("family:a") == a.partition(ancestry.task_node("flask-0"))
    assert a.partition("family:b") == "system_test"
    assert a.component("family:c") not in {a.component(ancestry.task_node("react-vite-%d" % n)) for n in range(30)}


def test_the_assignment_is_deterministic_for_a_seed(factory):
    assert split.assign(factory, bank()).document == split.assign(factory, bank()).document


def rows(path, task_ids):
    path.write_text("".join(json.dumps({"receipt": n, "meta": {"task_id": t}}) + "\n" for n, t in enumerate(task_ids)))
    return path


def test_rows_split_by_their_task_partition(factory, tmp_path):
    a = split.assign(factory, bank())
    ids = [t["id"] for t in bank()] * 2
    counts = split.split(a, [rows(tmp_path / "rows.jsonl", ids)], tmp_path / "out")
    for partition, name in ancestry.FILES.items():
        written = [json.loads(line)["meta"]["task_id"] for line in (tmp_path / "out" / name).read_text().splitlines()]
        assert all(a.partition(ancestry.task_node(t)) == partition for t in written)
        assert counts[partition]["rows"] == len(written)
    assert sum(c["rows"] for c in counts.values()) == len(ids)


@pytest.mark.parametrize("meta, code", [({}, "row_without_task"), ({"task_id": "elsewhere"}, "node_outside_ancestry")])
def test_a_row_outside_the_ancestry_publishes_nothing(factory, tmp_path, meta, code):
    a = split.assign(factory, bank())
    out = tmp_path / "out"
    out.mkdir()
    for name in ancestry.FILES.values():
        (out / name).write_text("previous accepted dataset\n")
    source = rows(tmp_path / "rows.jsonl", ["flask-0"])
    source.write_text(source.read_text() + json.dumps({"receipt": 9, "meta": meta}) + "\n")
    with pytest.raises(CampaignFault, match=code):
        split.split(a, [source], out)
    assert all((out / name).read_text() == "previous accepted dataset\n" for name in ancestry.FILES.values())


def test_late_invalid_input_does_not_publish_partial_dataset(factory, tmp_path):
    a = split.assign(factory, bank())
    source = rows(tmp_path / "rows.jsonl", ["flask-0"])
    source.write_text(source.read_text() + "{invalid json\n")
    out = tmp_path / "out"
    with pytest.raises(json.JSONDecodeError):
        split.split(a, [source], out)
    assert list(out.iterdir()) == []
