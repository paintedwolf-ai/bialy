"""Assemble a heads release: weights, model card with results, notice, and checksums.

The heads are trained and replayed in the Painted Wolf Code repository
(`scripts/bialy/`); this packs what that produced for publishing. Every
number on the card comes from a report the engine produced, shipped in `eval/`,
and a baseline row is one a reader can reproduce from the release.
"""

import json
import shutil
import struct
from pathlib import Path

from . import anchor, config, hub
from .provenance import factory_source, file_sha256

NOTICE = """Bialy decision heads
Copyright 2026 Painted Wolf LLC

Licensed under the Apache License, Version 2.0. See LICENSE.

These heads run over a backbone they do not include:

- Laya multilingual (convaiinnovations/laya-multilingual, revision
  e4e9ddf21a7b1903b7acffd8814ad4307bf63a67), by Convai Innovations, under
  the Apache License, Version 2.0: https://huggingface.co/convaiinnovations/laya-multilingual
- mmBERT-base (jhu-clsp/mmBERT-base), the encoder Laya multilingual builds on,
  by the Johns Hopkins University Center for Language and Speech Processing,
  under the MIT License: https://huggingface.co/jhu-clsp/mmBERT-base

"Painted Wolf" and "Painted Wolf Code" are trademarks of Painted Wolf LLC.
This license grants no trademark rights.
"""

CARD = """---
license: apache-2.0
base_model: convaiinnovations/laya-multilingual
datasets: [{dataset_repo}]
tags: [coding-agents, tool-selection]
---

# Bialy decision heads {version}

Tuned heads built on [Laya](https://github.com/NandhaKishorM/laya) by Convai Innovations,
over its frozen multilingual encoder that Painted Wolf Code
asks as it works: which loadable tool schemas a turn will need, which
instruction units it can leave out, and what kind of work it is (`turn-load`);
how relevant each skill and tool card is to a request (`unit-rank`); and which
code units best answer a request, blended into the text-match order of code
summaries, repository maps, and project search (`code-rank`). The engine
(`bialy`) loads them beside the backbone; a head trained over another
backbone is refused.

Trained on [{dataset_repo}](https://huggingface.co/datasets/{dataset_repo}) {dataset_version}: `turn-load`
and `unit-rank` on its session rows, labeled by what open-weights models did
in coding-agent sessions and scored by an open-weights judge; `code-rank` on its
code-rank pairs, requests open-weights models wrote for code units in the same
repositories. Trained and replayed with Painted Wolf Code at commit
{engine_commit}; packaged with {code_repo} at commit {factory_commit} (tag `heads-{version}`).

## Heads

{heads}

## Results

Replayed through the shipped engine on sets neither head trained on. Tool
columns are precision / recall / F1 of the loadable tools a turn used, at
the catalog's load threshold; guide omission precision is the share of
omitted instruction units the turn did not need; need MRR ranks the tools a
`request_tools` need went on to use.

{results}

`code-rank`, on repositories it never trained on: the rank of the code unit a
request was written for, in the site's text-match order and blended with the
head.

{rerank}

{retention}

## Check it yourself

Each row of the table comes from a replay report shipped in `eval/`.
`bialy audit heads` (from {code_repo}) replays these heads through the
Painted Wolf Code engine on a CPU over the dataset's held-out split and
compares every metric with the shipped report.
"""

COLUMNS = (("tools P/R/F1", lambda r: "%.2f / %.2f / %.2f" % tuple(r["tools"]["_all"][k] for k in ("precision", "recall", "f1"))),
           ("macro R", lambda r: r["tools"]["_all"].get("macro_recall")),
           ("loads/turn", lambda r: r["tools"]["_all"].get("loads_per_turn")),
           ("guide omit P", lambda r: r["guides"]["_all"].get("omission_precision")),
           ("kind acc", lambda r: r["kind"].get("accuracy")),
           ("need MRR", lambda r: r["requests"].get("mrr")))


def head_header(path):
    """A safetensors head's metadata (label, backbone, training facts) without loading it."""
    with open(path, "rb") as fh:
        size = struct.unpack("<Q", fh.read(8))[0]
        return json.loads(fh.read(size)).get("__metadata__", {})


def results_table(evals, baselines, baseline_label):
    """One row per set: these heads, then the baseline when one was replayed."""
    header = "| Set | Heads | " + " | ".join(name for name, _ in COLUMNS) + " |"
    lines = [header, "|" + "---|" * (len(COLUMNS) + 2)]
    for name in sorted(evals):
        for label, path in (("this release", evals[name]), (baseline_label, baselines.get(name))):
            if not path:
                continue
            report = json.loads(Path(path).read_text(encoding="utf-8"))["overall"]
            cells = []
            for _, get in COLUMNS:
                try:
                    value = get(report)
                except (KeyError, TypeError):
                    value = None
                cells.append("-" if value is None else str(value))
            lines.append("| %s | %s | %s |" % (name, label, " | ".join(cells)))
    return "\n".join(lines)


def rerank_table(reports):
    """One row per decide-rerank report: text-match and blended MRR and hit@1, and how many
    requests the head moved up or down."""
    lines = ["| Report | Pairs | MRR text / blended | hit@1 text / blended | improved / regressed |", "|---|---|---|---|---|"]
    for name in sorted(reports):
        r = json.loads(Path(reports[name]).read_text(encoding="utf-8"))
        lines.append("| %s | %d | %.3f / %.3f | %.3f / %.3f | %d / %d |" % (
            name, r["pairs"], r["lexical"]["mrr"], r["blended"]["mrr"], r["lexical"]["hit_1"], r["blended"]["hit_1"],
            r["improved"], r["regressed"]))
    return "\n".join(lines)


def retention_section(report_path, verdict):
    """The card's retention section: the verdict, the fitted policy, and the held-out
    numbers the choice rests on, from the chosen head's replay report."""
    if not report_path:
        return "No retention head is in this release."
    report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    held = report.get("holdout") or report.get("overall") or {}
    engine = held.get("engine") or {}
    fitted = report.get("fitted") or {}
    lines = ["## Retention", "",
             "`retention` answers which earlier tool results a request will read again; the host releases the rest behind recall pointers.",
             "Its policy was fitted on validation rows by the objective in `scripts/bialy/retention_eval.py` (prompt tokens a release saves "
             "over the request's remaining calls, less the recall round trip a wrong release costs) and scored on held-out workspaces.", "",
             "- Verdict: **%s**%s" % ("viable" if (verdict or {}).get("viable") else "not viable",
                                      "" if not (verdict or {}).get("reasons") else "; " + "; ".join(verdict["reasons"])),
             "- Fitted policy: drop_below %s, confidence_floor %s" % (fitted.get("drop_below"), fitted.get("confidence_floor")),
             "- Held-out: %s rows, %s candidates, AUC %s, objective %s tokens per request, interval %s" % (
                 held.get("rows"), engine.get("candidates"), held.get("auc"), engine.get("objective_per_request"),
                 json.dumps(((held.get("bootstrap") or {}).get("engine")) or None)),
             "- Structural facts recommended to release on their own: %s" % json.dumps(((report.get("recommendation") or {}).get("facts")) or [])]
    return "\n".join(lines)


def build(heads_dir, version, dataset_version, engine_commit, evals, baselines, baseline_label, rerank, out_dir, notes=None, retention=None, verdict=None):
    """The heads, their card, and the reports behind every number on it. `notes` are
    further JSON reports copied into eval/ as they are (the retention selection and the
    pass comparison); `retention` is the chosen retention head's replay report."""
    heads_dir, out_dir = Path(heads_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    settings = config.hub()
    notes = notes or {}
    # The Bialy commit packaging the release; publishing tags it heads-<version> (docs/releases.md).
    factory = factory_source()
    heads = []
    for path in sorted(heads_dir.glob("*.safetensors")):
        shutil.copy2(path, out_dir / path.name)
        heads.append({"file": path.name, "sha256": file_sha256(path), "metadata": head_header(path)})
    card = CARD.format(version=version, dataset_repo=settings["dataset_repo"], dataset_version=dataset_version, code_repo=settings["code_repo"],
                       engine_commit=engine_commit, factory_commit=factory["git_commit"],
                       heads="\n".join("- `%s`: %s, over %s, sha256 `%s`" % (h["file"], h["metadata"].get("label"), h["metadata"].get("backbone"),
                                                                            h["sha256"]) for h in heads),
                       results=results_table(evals, baselines, baseline_label), rerank=rerank_table(rerank),
                       retention=retention_section(retention, verdict))
    (out_dir / "README.md").write_text(card, encoding="utf-8")
    (out_dir / "NOTICE").write_text(NOTICE, encoding="utf-8")
    shutil.copy2(Path(__file__).resolve().parents[2] / "LICENSE", out_dir / "LICENSE")
    # The reports behind the card, so the table can be checked and replayed (`bialy audit heads`).
    (out_dir / "eval").mkdir(exist_ok=True)
    for name, path in evals.items():
        shutil.copy2(path, out_dir / "eval" / ("%s.json" % name))
    for name, path in baselines.items():
        shutil.copy2(path, out_dir / "eval" / ("%s.baseline.json" % name))
    for name, path in rerank.items():
        shutil.copy2(path, out_dir / "eval" / ("rerank-%s.json" % name))
    for name, path in notes.items():
        shutil.copy2(path, out_dir / "eval" / ("%s.json" % name))
    provenance = {"version": version, "dataset_version": dataset_version, "engine_commit": engine_commit, "factory": factory, "heads": heads,
                  "evals": sorted(evals), "baseline": {"label": baseline_label, "sets": sorted(baselines)}, "rerank": sorted(rerank),
                  "notes": sorted(notes), "retention": verdict}
    (out_dir / "PROVENANCE.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    hub.seal(out_dir)
    anchor.record("heads", version, out_dir)
    return out_dir
