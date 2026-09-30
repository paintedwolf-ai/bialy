import json

import pytest

from bialy import split


def test_split_keeps_prompt_groups_and_holdout_repositories_whole(factory, tmp_path):
    rows = tmp_path / "rows.jsonl"
    with open(rows, "w") as fh:
        for i in range(300):
            repo = ["flask", "gin", "zod"][i % 3]
            fh.write(json.dumps({"project": repo, "root_session": "s%d" % i, "meta": {"repo": repo, "prompt_group": "%s:g%d" % (repo, i // 2)}}) + "\n")
    counts = split.split(factory, [rows], tmp_path / "out")
    read = lambda name: [json.loads(line) for line in open(tmp_path / "out" / name)]  # noqa: E731
    assert all(r["meta"]["repo"] == "zod" for r in read("holdout.jsonl")) and counts["holdout"]["rows"] == 100
    train_groups = {r["meta"]["prompt_group"] for r in read("train.jsonl")}
    assert not train_groups & {r["meta"]["prompt_group"] for r in read("val.jsonl")}
    assert 0 < counts["val"]["rows"] < counts["train"]["rows"]


def test_split_is_deterministic(factory, tmp_path):
    rows = tmp_path / "rows.jsonl"
    rows.write_text("".join(json.dumps({"project": "flask", "root_session": "s%d" % i, "meta": {"repo": "flask", "prompt_group": "g%d" % i}}) + "\n" for i in range(50)))
    split.split(factory, [rows], tmp_path / "a")
    split.split(factory, [rows], tmp_path / "b")
    assert (tmp_path / "a" / "val.jsonl").read_text() == (tmp_path / "b" / "val.jsonl").read_text()


def test_rows_from_unknown_repositories_are_refused(factory, tmp_path):
    rows = tmp_path / "rows.jsonl"
    rows.write_text(json.dumps({"project": "mystery", "root_session": "s", "meta": {}}) + "\n")
    with pytest.raises(ValueError):
        split.split(factory, [rows], tmp_path / "out")
