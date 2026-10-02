import json
import shutil

import pytest
from conftest import FIXTURES

from bialy import anchor, audit, heads, hub, release


def build_release(factory, split_dir, corpus_file, tmp_path, pairs=((3, 3), (0, 1), (4, 4)), driven=None):
    agreement = tmp_path / "agreement.jsonl"
    agreement.write_text("".join(json.dumps({"session": "s", "receipt": 1, "unit": "skills", "candidate": str(i), "first_judge": "a",
                                             "second_judge": "b", "first_score": x, "second_score": y}) + "\n" for i, (x, y) in enumerate(pairs)))
    if driven is None:
        driven = tmp_path / "tasks-driven.jsonl"
        driven.write_text("".join(json.dumps({"id": "t%d" % n, "prompt": "p", "outcome": {"status": "settled", "run": "r"}}) + "\n" for n in range(1, 6)))
    stage = tmp_path / "fleet-provenance.json"
    stage.write_text(json.dumps({"stage": "fleet", "at": "2026-01-01T00:00:00Z", "platform": "Linux-x86_64",
                                 "factory": {"git_commit": "abc", "dirty": False}, "engine": {"lycaon_commit": "def"},
                                 "serving": {"vllm": "0.30.0"}, "config": {"factory.yaml": "root: /x\n"}, "image": "sha256:1"}))
    coderank = tmp_path / "coderank"
    coderank.mkdir(exist_ok=True)
    (coderank / "model-flask.jsonl").write_text(json.dumps({"repo": "flask", "file": "a.py", "line": 1, "symbol": "f", "task": "t"}) + "\n")
    return release.build(factory, split_dir, "v0", tmp_path / "dist", FIXTURES / "row.schema.json", corpus_file, "v0", [agreement],
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
    assert report["tasks"] == {"driven": 5, "outcomes": {"settled": 5}} and "Stopped after 5 settled tasks." in card
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


def fake_report(precision, recall, f1):
    return {"overall": {"tools": {"_all": {"precision": precision, "recall": recall, "f1": f1, "macro_recall": 0.5, "loads_per_turn": 2.1}},
                        "guides": {"_all": {"omission_precision": 0.98}}, "kind": {"accuracy": 0.6}, "requests": {"mrr": 0.6}}}


def test_heads_release_ships_its_evidence(tmp_path):
    heads_dir = tmp_path / "heads"
    heads_dir.mkdir()
    header = json.dumps({"__metadata__": {"label": "turn-load@test", "backbone": "enc", "seed": "11"}}).encode()
    (heads_dir / "turn-load.safetensors").write_bytes(len(header).to_bytes(8, "little") + header)
    (tmp_path / "new.json").write_text(json.dumps(fake_report(0.3, 0.7, 0.42)))
    (tmp_path / "off.json").write_text(json.dumps(fake_report(0.0, 0.0, 0.0)))
    (tmp_path / "rerank.json").write_text(json.dumps({"pairs": 10, "lexical": {"mrr": 0.4, "hit_1": 0.3}, "blended": {"mrr": 0.5, "hit_1": 0.4},
                                                      "improved": 3, "regressed": 1}))
    out = heads.build(heads_dir, "v1", "v1", "abc123", {"holdout": tmp_path / "new.json"}, {"holdout": tmp_path / "off.json"}, "base checkpoint",
                      {"definitions-zod": tmp_path / "rerank.json"}, tmp_path / "dist")
    card = (out / "README.md").read_text()
    assert "| holdout | this release | 0.30 / 0.70 / 0.42 |" in card and "| holdout | base checkpoint | 0.00 / 0.00 / 0.00 |" in card
    assert "| definitions-zod | 10 | 0.400 / 0.500 | 0.300 / 0.400 | 3 / 1 |" in card and "abc123" in card
    assert (out / "eval" / "holdout.baseline.json").exists() and (out / "eval" / "rerank-definitions-zod.json").exists()
    provenance = json.loads((out / "PROVENANCE.json").read_text())
    assert provenance["engine_commit"] == "abc123" and provenance["heads"][0]["metadata"]["seed"] == "11"
    assert "eval/holdout.json" in hub.check("heads", out)


def test_a_release_names_its_stacks_and_anchors_every_held_out_workspace(factory, split_dir, corpus_file, tmp_path, anchors):
    out = build_release(factory, split_dir, corpus_file, tmp_path)
    provenance = json.loads((out / "PROVENANCE.json").read_text())
    assert [s["name"] for s in provenance["stacks"]] == [s.name for s in factory.stacks]
    assert json.loads((anchors / "dataset-v0.json").read_text())["holdout"] == sorted(factory.holdout())
    card = (out / "README.md").read_text()
    assert "- Greenfield stacks: web-static, react-vite" in card and "java-maven" in card.split("- Held out")[1]
    assert not (out / "licenses" / "react-vite").exists()
