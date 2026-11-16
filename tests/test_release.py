import json
import shutil
from pathlib import Path

import pytest
from conftest import FIXTURES, layout_ancestry
from selector_bench import qualified

from bialy import anchor, audit, heads, hub, release
from bialy.campaign import selector


def retention_split(tmp_path):
    """Retention rows for two of the split fixture's requests; the holdout split wrote none."""
    out = tmp_path / "split" / "retention"
    out.mkdir(parents=True)
    row = lambda n, repo, group: {"schema": "pw-retention-row/1", "receipt": n, "session": "s%d" % n, "root_session": "s%d" % n,  # noqa: E731
                                  "candidates": [{"id": "r", "tool": "read", "tokens": 400, "age": 1, "facts": {}, "option": "read a, 1 turns ago, 400 tokens"}],
                                  "labels": {"needed": {"r": True}, "later": {"r": True}, "by": {"r": ["path"]}},
                                  "meta": {"workspace": repo, "prompt_group": group, "task_id": "t%d" % n}}
    (out / "train.jsonl").write_text(json.dumps(row(1, "flask", "flask:a")) + "\n" + json.dumps(row(3, "gin", "gin:b")) + "\n")
    (out / "val.jsonl").write_text(json.dumps(row(4, "gin", "gin:c")) + "\n")
    return out


def test_a_release_carries_the_retention_rows_it_was_given(factory, split_dir, corpus_file, tmp_path, driven_file):
    out = release.build(factory, split_dir, "v0r", tmp_path / "dist-retention", FIXTURES / "row.schema.json", corpus_file, "v0",
                        layout_ancestry(tmp_path / "ancestry.json"), driven=[driven_file],
                        retention=retention_split(tmp_path), retention_schema=FIXTURES / "row.schema.json")
    provenance = json.loads((out / "PROVENANCE.json").read_text())
    assert provenance["retention_rows"] == {"train": 2, "validation": 1}
    assert (out / "retention" / "train.jsonl").exists() and (out / "retention" / "retention-row.schema.json").exists()
    assert not (out / "retention" / "holdout.jsonl").exists() and "## Context retention rows" in (out / "README.md").read_text()
    assert "retention/train.jsonl" in (out / "SHA256SUMS").read_text()
    plain = build_release(factory, split_dir, corpus_file, tmp_path)
    assert json.loads((plain / "PROVENANCE.json").read_text())["retention_rows"] is None and not (plain / "retention").exists()


def build_release(factory, split_dir, corpus_file, tmp_path, pairs=((3, 3), (0, 1), (4, 4)), driven=None):
    agreement = tmp_path / "agreement.jsonl"
    agreement.write_text("".join(json.dumps({"session": "s", "receipt": 1, "unit": "skills", "candidate": str(i), "first_judge": "a",
                                             "second_judge": "b", "first_score": x, "second_score": y}) + "\n" for i, (x, y) in enumerate(pairs)))
    if driven is None:
        driven = tmp_path / "tasks-driven.jsonl"
        driven.write_text("".join(json.dumps({"id": "t%d" % n, "prompt": "p", "outcome": {"status": "settled", "run": "r"}}) + "\n" for n in range(1, 7)))
    stage = tmp_path / "fleet-provenance.json"
    stage.write_text(json.dumps({"stage": "fleet", "at": "2026-01-01T00:00:00Z", "platform": "Linux-x86_64",
                                 "factory": {"git_commit": "abc", "dirty": False}, "engine": {"lycaon_commit": "def"},
                                 "serving": {"vllm": "0.30.0"}, "config": {"factory.yaml": "root: /x\n"}, "image": "sha256:1"}))
    coderank = tmp_path / "coderank"
    coderank.mkdir(exist_ok=True)
    (coderank / "model-flask.jsonl").write_text(json.dumps({"repo": "flask", "file": "a.py", "line": 1, "symbol": "f", "task": "t"}) + "\n")
    return release.build(factory, split_dir, "v0", tmp_path / "dist", FIXTURES / "row.schema.json", corpus_file, "v0",
                         layout_ancestry(tmp_path / "ancestry.json"), [agreement],
                         [stage], driven=[driven], coderank=coderank, stopping="Stopped after 5 settled tasks")


def resum(out):
    """Checksum a release again after an edit."""
    hub.seal(out)


def test_a_release_passes_its_own_audit(factory, split_dir, corpus_file, tmp_path):
    out = build_release(factory, split_dir, corpus_file, tmp_path)
    card = (out / "README.md").read_text()
    assert card.startswith("---\nlicense: apache-2.0") and "split: validation, path: val.jsonl" in card and "`train.jsonl`: 3 rows" in card
    report = audit.dataset(out)
    assert report["splits"]["train"]["rows"] == 3 and report["judge_agreement"]["pairs"] == 3 and report["leakage"] == "none"
    assert report["tasks"] == {"driven": 6, "outcomes": {"settled": 6}} and "Stopped after 5 settled tasks." in card
    assert "split: development, path: development.jsonl" in card and report["splits"]["development"]["rows"] == 1
    assert json.loads((out / "PROVENANCE.json").read_text())["ancestry_sha256"] == json.loads((out / "ancestry.json").read_text())["assignment_sha256"]
    assert (out / "coderank" / "model-flask.jsonl").exists() and "Painted Wolf Code at commit def" in card


def test_provenance_is_a_recipe_without_where_or_when(factory, split_dir, corpus_file, tmp_path):
    out = build_release(factory, split_dir, corpus_file, tmp_path)
    text = (out / "PROVENANCE.json").read_text()
    recipe = json.loads(text)["recipe"]
    assert recipe["sessions"] == [{"engine": {"lycaon_commit": "def"}, "serving": {"vllm": "0.30.0"}, "decide_env": None,
                                   "config": {"factory.yaml": "root: /x\n"}}]
    for gone in ('"abc"', "2026-01-01", "Linux-x86_64", "sha256:1"):
        assert gone not in text


def test_audit_catches_a_row_from_an_unlisted_task(factory, split_dir, corpus_file, tmp_path):
    driven = tmp_path / "partial-driven.jsonl"
    driven.write_text("".join(json.dumps({"id": "t%d" % n, "prompt": "p", "outcome": {"status": "settled", "run": "r"}}) + "\n" for n in range(1, 5)))
    out = build_release(factory, split_dir, corpus_file, tmp_path, driven=driven)
    with pytest.raises(audit.AuditError, match="tasks.jsonl does not list"):
        audit.dataset(out)


def test_audit_catches_a_leaked_prompt_group(factory, split_dir, corpus_file, tmp_path):
    val = split_dir / "val.jsonl"
    leaked = json.loads(val.read_text())
    leaked["meta"]["prompt_group"] = "flask:a"
    val.write_text(json.dumps(leaked) + "\n")
    out = build_release(factory, split_dir, corpus_file, tmp_path)
    with pytest.raises(audit.AuditError, match="prompt groups"):
        audit.dataset(out)


def test_audit_catches_a_row_outside_its_tasks_partition(factory, split_dir, corpus_file, tmp_path):
    moved = (split_dir / "val.jsonl").read_text()
    (split_dir / "val.jsonl").write_text("")
    (split_dir / "development.jsonl").write_text((split_dir / "development.jsonl").read_text() + moved)
    out = build_release(factory, split_dir, corpus_file, tmp_path)
    with pytest.raises(audit.AuditError, match="the ancestry puts in validation"):
        audit.dataset(out)


def test_audit_catches_a_training_repository_in_the_test_split(factory, split_dir, corpus_file, tmp_path):
    shutil.copy(split_dir / "val.jsonl", split_dir / "holdout.jsonl")
    (split_dir / "val.jsonl").write_text("")
    out = build_release(factory, split_dir, corpus_file, tmp_path)
    with pytest.raises(audit.AuditError, match="training workspaces"):
        audit.dataset(out)


def test_audit_catches_statistics_that_differ_from_the_rows(factory, split_dir, corpus_file, tmp_path):
    out = build_release(factory, split_dir, corpus_file, tmp_path)
    doc = json.loads((out / "PROVENANCE.json").read_text())
    doc["stats"]["train"]["rows"] = 999
    (out / "PROVENANCE.json").write_text(json.dumps(doc))
    resum(out)
    with pytest.raises(audit.AuditError, match="statistics"):
        audit.dataset(out, anchored=False)


def test_audit_recomputes_the_judge_agreement(factory, split_dir, corpus_file, tmp_path):
    out = build_release(factory, split_dir, corpus_file, tmp_path)
    doc = json.loads((out / "PROVENANCE.json").read_text())
    doc["judge_agreement"]["weighted_kappa"] = 0.99
    (out / "PROVENANCE.json").write_text(json.dumps(doc))
    resum(out)
    with pytest.raises(audit.AuditError, match="agreement"):
        audit.dataset(out, anchored=False)


def test_audit_validates_rows_against_the_shipped_schema(factory, split_dir, corpus_file, tmp_path):
    bad = json.loads((split_dir / "train.jsonl").read_text().splitlines()[0])
    bad["host"] = "robot"
    (split_dir / "train.jsonl").write_text(json.dumps(bad) + "\n")
    out = build_release(factory, split_dir, corpus_file, tmp_path)
    with pytest.raises(audit.AuditError, match="row.schema.json"):
        audit.dataset(out)


def test_tampering_breaks_the_checksums(factory, split_dir, corpus_file, tmp_path):
    out = build_release(factory, split_dir, corpus_file, tmp_path)
    assert hub.plan("dataset", out)["repo_type"] == "dataset"
    (out / "train.jsonl").write_text("tampered\n")
    with pytest.raises(ValueError):
        hub.plan("dataset", out)


def test_a_rechecksummed_release_fails_against_its_anchor(factory, split_dir, corpus_file, tmp_path):
    out = build_release(factory, split_dir, corpus_file, tmp_path)
    hub.check("dataset", out)
    (out / "train.jsonl").write_text((out / "val.jsonl").read_text())
    resum(out)
    hub.check("dataset", out, anchored=False)
    with pytest.raises(anchor.AnchorError, match="no dataset anchor"):
        hub.check("dataset", out)


def test_the_holdout_set_comes_from_the_anchor(factory, split_dir, corpus_file, tmp_path, anchors):
    out = build_release(factory, split_dir, corpus_file, tmp_path)
    path = anchors / "dataset-v0.json"
    doc = json.loads(path.read_text())
    doc["holdout"] = ["flask"]
    path.write_text(json.dumps(doc))
    with pytest.raises(audit.AuditError, match="anchor holds out"):
        audit.dataset(out)


def test_an_anchored_version_is_never_rebuilt_differently(factory, split_dir, corpus_file, tmp_path):
    build_release(factory, split_dir, corpus_file, tmp_path)
    (split_dir / "train.jsonl").write_text((split_dir / "train.jsonl").read_text().splitlines()[0] + "\n")
    with pytest.raises(anchor.AnchorError, match="new version"):
        build_release(factory, split_dir, corpus_file, tmp_path)


def selector_directory(tmp_path, holdout=True, **changes):
    """A selector directory as a run or a campaign leaves it: fitted, qualified, and compared."""
    journal, run, workbench, trained = qualified(tmp_path, **changes)
    selector.SelectorQualification(journal, run, workbench).execute({"selector": trained}, "q")
    if holdout:
        run.path("split", "holdout.jsonl").write_text(json.dumps({"offered": {"loadable": ["read"]}, "labels": {"tools": ["read"]}}) + "\n")
        selector.SelectorHoldout(journal, run, workbench).execute({"selector": trained}, "h")
    journal.close()
    return journal.directory / "selector", trained


def test_a_heads_release_is_the_qualified_policy_with_its_evidence(tmp_path, anchors):
    directory, trained = selector_directory(tmp_path)
    out = heads.build(directory, "v1", "v1", "abc123", tmp_path / "dist")
    policy = json.loads((out / "policy.json").read_text())
    assert policy == trained["policy"] and policy["preload"]["options"]
    assert sorted(p.name for p in out.glob("*.safetensors")) == ["code-rank.safetensors", "turn-load.safetensors", "unit-rank.safetensors"]
    card = (out / "README.md").read_text()
    assert "| holdout | 0.70 / 0.90 / 0.79 |" in card and "| held-out | 240 | 0.410 / 0.520 | 0.300 / 0.400 | 60 / 12 |" in card
    assert "| `code-rank` | mrr_gain | 0.11 | 0.01 |" in card and "worst gap 0.0002" in card and "abc123" in card
    assert policy["identity"] in card and "`turn.tools.load_at` 0.55" in card
    provenance = json.loads((out / "PROVENANCE.json").read_text())
    assert provenance["engine_commit"] == "abc123" and provenance["policy"] == policy["identity"]
    assert provenance["heads"][0]["metadata"]["seed"] == "11"
    names = hub.check("heads", out)
    assert {"eval/qualification.json", "eval/development.json", "eval/holdout.json", "eval/holdout-comparison.json", "corpus.json"} <= set(names)
    assert json.loads((out / "corpus.json").read_text())["questions"]["tools"]["load_at"] == 0.55


def test_nothing_is_released_without_a_qualification_or_the_held_out_comparison(tmp_path, anchors):
    directory, _ = selector_directory(tmp_path / "a", holdout=False)
    with pytest.raises(heads.ReleaseError, match="held-out comparison"):
        heads.build(directory, "v1", "v1", "abc123", tmp_path / "dist")
    directory, _ = selector_directory(tmp_path / "b")
    (directory / "qualification.json").unlink()
    with pytest.raises(heads.ReleaseError, match="no recorded qualification"):
        heads.build(directory, "v1", "v1", "abc123", tmp_path / "dist")
    directory, _ = selector_directory(tmp_path / "c")
    (directory / "heads" / "unit-rank.safetensors").write_bytes(b"retrained after qualifying")
    with pytest.raises(heads.ReleaseError, match="no intact selector policy"):
        heads.build(directory, "v1", "v1", "abc123", tmp_path / "dist")
    with pytest.raises(heads.ReleaseError, match="no intact selector policy"):
        heads.build(tmp_path / "heads-only", "v1", "v1", "abc123", tmp_path / "dist")
    assert not (tmp_path / "dist").exists() and not list(anchors.glob("heads-*"))


def test_the_heads_audit_replays_as_the_release_was_measured(tmp_path, anchors, monkeypatch):
    directory, _ = selector_directory(tmp_path)
    out = heads.build(directory, "v1", "v1", "abc123", tmp_path / "dist")
    seen = {}

    def replay(args, cwd=None, check=None, stdout=None):
        seen.update(args=args, cwd=cwd, launcher=Path(args[args.index("--engine") + 1]).read_text())
        Path(args[args.index("--json") + 1]).write_text((out / "eval" / "holdout.json").read_text())
    monkeypatch.setattr("subprocess.run", replay)
    report = audit.heads(out, tmp_path / "dataset", "/lycaon", "/engine/bialy", "/model")
    args = seen["args"]
    assert args[1] == "/lycaon/scripts/bialy/replay_eval.py" and seen["cwd"] == "/lycaon" and args[args.index("--tool-truth") + 1] == "called"
    assert args[args.index("--corpus") + 1] == str(out / "corpus.json") and args[args.index("--examples") + 1] == str(tmp_path / "dataset" / "holdout.jsonl")
    assert "/engine/bialy serve --model /model" in seen["launcher"] and "--head-max-len 512" in seen["launcher"]
    assert "--head turn-load=%s" % (out / "turn-load.safetensors") in seen["launcher"]
    assert report["metrics"]["tools.recall"] == {"shipped": 0.9, "replayed": 0.9}


def test_a_release_names_its_stacks_and_anchors_every_held_out_workspace(factory, split_dir, corpus_file, tmp_path, anchors):
    out = build_release(factory, split_dir, corpus_file, tmp_path)
    provenance = json.loads((out / "PROVENANCE.json").read_text())
    assert [s["name"] for s in provenance["stacks"]] == [s.name for s in factory.stacks]
    assert json.loads((anchors / "dataset-v0.json").read_text())["holdout"] == sorted(factory.holdout())
    card = (out / "README.md").read_text()
    assert "- Greenfield stacks: web-static, react-vite" in card and "java-maven" in card.split("- Held out")[1]
    assert not (out / "licenses" / "react-vite").exists()
