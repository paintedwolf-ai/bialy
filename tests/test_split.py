import json

import pytest

from bialy import split


def test_split_keeps_prompt_groups_and_held_out_workspaces_whole(factory, tmp_path):
    rows = tmp_path / "rows.jsonl"
    workspaces = ["flask", "gin", "zod", "react-vite", "vue-vite"]
    with open(rows, "w") as fh:
        for i in range(500):
            name = workspaces[i % 5]
            fh.write(json.dumps({"project": name, "root_session": "s%d" % i, "meta": {"workspace": name, "prompt_group": "%s:g%d" % (name, i // 2)}}) + "\n")
    counts = split.split(factory, [rows], tmp_path / "out")
    read = lambda name: [json.loads(line) for line in open(tmp_path / "out" / name)]  # noqa: E731
    # A held-out repository and a held-out stack are both held out whole.
    assert {r["meta"]["workspace"] for r in read("holdout.jsonl")} == {"zod", "vue-vite"} and counts["holdout"]["rows"] == 200
    assert {r["meta"]["workspace"] for r in read("train.jsonl")} == {"flask", "gin", "react-vite"}
    train_groups = {r["meta"]["prompt_group"] for r in read("train.jsonl")}
    assert not train_groups & {r["meta"]["prompt_group"] for r in read("val.jsonl")}
    assert 0 < counts["val"]["rows"] < counts["train"]["rows"]


def test_split_is_deterministic(factory, tmp_path):
    rows = tmp_path / "rows.jsonl"
    rows.write_text("".join(json.dumps({"project": "flask", "root_session": "s%d" % i, "meta": {"workspace": "flask", "prompt_group": "g%d" % i}}) + "\n" for i in range(50)))
    split.split(factory, [rows], tmp_path / "a")
    split.split(factory, [rows], tmp_path / "b")
    assert (tmp_path / "a" / "val.jsonl").read_text() == (tmp_path / "b" / "val.jsonl").read_text()


def test_rows_without_a_known_workspace_are_refused(factory, tmp_path):
    rows = tmp_path / "rows.jsonl"
    rows.write_text(json.dumps({"project": "mystery", "root_session": "s", "meta": {}}) + "\n")
    with pytest.raises(ValueError, match="no task workspace"):
        split.split(factory, [rows], tmp_path / "out")
    rows.write_text(json.dumps({"project": "mystery", "root_session": "s", "meta": {"workspace": "mystery", "prompt_group": "g"}}) + "\n")
    with pytest.raises(ValueError, match="unknown workspace"):
        split.split(factory, [rows], tmp_path / "out")
