"""Derive a turn-load dataset release from a published one by relabelling its guide units.

The rows, order, sanitisation, and every other label come from the parent release; only
`labels.guides` changes, through `relabel_guides.py`, from a corpus whose units declare
`needed_with`. The output carries the parent's licence and notices, the new corpus and
its independent form, a trainer snapshot, a card, provenance naming the parent and the
rule, and checksums in the parent's layout.

Usage: derive_guides_release.py --parent DIR --corpus FILE --independent-corpus FILE
                                --trainer DIR --version NAME --out DIR
"""
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from datetime import date
from pathlib import Path

HERE = Path(__file__).resolve().parent


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path):
    return sum(1 for line in path.open() if line.strip())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ("parent", "corpus", "independent_corpus", "trainer", "version", "out"):
        ap.add_argument("--" + name.replace("_", "-"), dest=name, required=True)
    args = ap.parse_args()
    parent, out, trainer = Path(args.parent), Path(args.out), Path(args.trainer)
    out.mkdir(parents=True)
    for name in ("LICENSE", "NOTICE", "row.schema.json", "agreement.jsonl", "tasks.jsonl"):
        shutil.copy(parent / name, out / name)
    (out / "corpora").mkdir()
    shutil.copy(args.corpus, out / "corpus.json")
    shutil.copy(args.independent_corpus, out / "corpora" / "independent.json")
    files = []
    for split in ("train", "val", "holdout"):
        src, dst = parent / f"{split}.jsonl", out / f"{split}.jsonl"
        report = subprocess.run([sys.executable, str(HERE / "relabel_guides.py"), str(out / "corpus.json"), str(src), str(dst)],
                                check=True, capture_output=True, text=True).stdout
        files.append({"path": dst.name, "source_sha256": sha256(src), "sha256": sha256(dst), "source_bytes": src.stat().st_size,
                      "bytes": dst.stat().st_size, "rows": rows(dst), "relabel": report.splitlines()[0]})
    snapshot = out / "training" / "turn-load" / "scripts" / "decide"
    shutil.copytree(trainer / "scripts" / "decide", snapshot, ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copy(trainer / "scripts" / "artifact_paths.py", snapshot.parent / "artifact_paths.py")
    for name in ("relabel_guides.py", "turn_probe.py", "guide_audit.py", "train.sh"):
        shutil.copy(HERE / name, out / "training" / "turn-load" / name)
    parent_prov = json.loads((parent / "PROVENANCE.json").read_text())
    provenance = {
        "schema": parent_prov.get("schema"), "version": args.version, "derived_from": parent_prov.get("version"),
        "derived_on": date.today().isoformat(),
        "derivation": "labels.guides relabelled from tool calls: a unit is needed when its turn called a tool it attaches to "
                      "or one it declares needed_with (train-host/relabel_guides.py); rows, order, and other labels unchanged",
        "sources": parent_prov.get("sources"), "models": parent_prov.get("models"),
    }
    (out / "PROVENANCE.json").write_text(json.dumps(provenance, indent=2) + "\n")
    (out / "TRAINING.md").write_text(TRAINING.format(version=args.version, parent=parent_prov.get("version")))
    (out / "README.md").write_text(CARD.format(version=args.version, parent=parent_prov.get("version"),
                                               counts=", ".join("%s (%s)" % (f["path"], f["rows"]) for f in files)))
    (out / "FILES.json").write_text(json.dumps(files, indent=2) + "\n")
    sums = []
    for path in sorted(p for p in out.rglob("*") if p.is_file() and p.name != "SHA256SUMS"):
        sums.append("%s  %s" % (sha256(path), path.relative_to(out)))
    (out / "SHA256SUMS").write_text("\n".join(sums) + "\n")
    print("wrote %s: %s" % (out, ", ".join(f["path"] + " " + f["relabel"] for f in files)))


TRAINING = """# Training from the archived inputs

{version} derives from {parent}: the same rows in the same order, with `labels.guides`
relabelled from tool calls (`training/turn-load/relabel_guides.py`). Train the B7 and
B7G heads with `training/turn-load/train.sh`:

- `train.sh GPU B7 RUN`: `--families tools,guides` on `corpora/independent.json`,
  `--tool-truth consensus --tool-weight none --tool-negatives 24 --pos-weight 6
  --lr 0.0005 --seed 11 --batch-size 64 --epochs 45 --patience 8`.
- `train.sh GPU B7G RUN`: the same with `--families guides`.

Score a head on every tool and guide option with `training/turn-load/turn_probe.py` and
audit omissions with `training/turn-load/guide_audit.py`. The backbone is
`convaiinnovations/laya-multilingual` at revision
`e4e9ddf21a7b1903b7acffd8814ad4307bf63a67`; obtain that revision explicitly.
"""

CARD = """---
license: apache-2.0
pretty_name: Painted Wolf Decide training release {version}
tags:
- synthetic
- coding-agents
- tool-selection
configs:
- config_name: turn_load
  data_files:
  - split: train
    path: train.jsonl
  - split: validation
    path: val.jsonl
  - split: test
    path: holdout.jsonl
---

# Painted Wolf Decide training release {version}

The turn-load rows of {parent} with their guide-unit labels relabelled from tool calls,
the companion dataset for the guide-load head. Rows, order, sanitisation, and every other
label are those of the parent release; {parent} remains the source for the unit-rank and
code-rank inputs and the conversation exports.

| Configuration | Files | Meaning |
| --- | --- | --- |
| Turn-load | {counts}, `corpora/independent.json` | Parent rows; `labels.guides` is whether the turn called a tool the unit attaches to or is needed with. |
| Corpus | `corpus.json` | The host's unit, tool, and skill catalog, with each unit's `attaches` and `needed_with`. |
| Trainer | `training/turn-load/` | The trainer snapshot, recipes, relabelling, probe, and audit scripts. |

`FILES.json` records the parent and released row hashes and the relabelling counts;
`SHA256SUMS` covers the release. See [Training](TRAINING.md) and `PROVENANCE.json`.
"""


if __name__ == "__main__":
    main()
