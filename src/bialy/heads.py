"""Assemble a heads release from a qualified selector policy: the three head files, the
policy that binds them, a model card, a notice, and checksums.

A release is cut only from a selector directory whose policy is intact, recorded as
qualified (the parity probe and every bar on the development split) and compared once
on the held-out split. Every number on the card comes from a report shipped in `eval/`.
"""

import json
import shutil
import struct
from pathlib import Path

from . import anchor, config, hub
from .campaign import selector
from .campaign.journal import CampaignFault
from .provenance import factory_source

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
base_model: {backbone}
datasets: [{dataset_repo}]
tags: [coding-agents, tool-selection]
---

# Bialy decision heads {version}

Tuned heads built on [Laya](https://github.com/NandhaKishorM/laya) by Convai Innovations,
over its frozen multilingual encoder that Painted Wolf Code
asks as it works: which loadable tool schemas a turn will need (`turn-load`);
how relevant each skill and tool card is to a request (`unit-rank`); and which
code units best answer a request, blended into the text-match order of code
summaries, repository maps, and project search (`code-rank`). The engine
(`bialy`) loads them beside the backbone; a head trained over another
backbone is refused.

Trained on [{dataset_repo}](https://huggingface.co/datasets/{dataset_repo}) {dataset_version}: `turn-load`
and `unit-rank` on its training rows, labeled by what open-weights models did
in coding-agent sessions and scored by open-weights judges; `code-rank` on the
code-rank pairs of its training repositories. Trained, calibrated and replayed
with Painted Wolf Code at commit {engine_commit}; packaged with {code_repo} at
commit {factory_commit} (tag `heads-{version}`).

## Policy

The three heads are one selector policy, `policy.json` (identity `{identity}`):
their digests, the backbone, the option vocabulary the `turn-load` head was
trained with (`preload`), and the decision thresholds, calibrated on the
validation split. A host adopts all of it together.

{heads}

Thresholds: {thresholds}

## Qualification

Measured on the development split, which no head trained on and which selected
no epoch or threshold. The engine and the trainer's own forward agree on every
`turn-load` probability within {tolerance} over {parity_turns} turns (worst gap {parity_gap}).

{qualification}

## Held-out results

Read once, after qualification, on the held-out workspaces. Tool columns are
precision / recall / F1 of the loadable tools a turn used, at the policy's load
threshold; need MRR ranks the tools a `request_tools` need went on to use.

{results}

`code-rank`: the rank of the code unit a request was written for, in the site's
text-match order and blended with the head. No held-out repository has a
candidate dump, so none reached training or epoch selection.

{rerank}

## Check it yourself

Each row comes from a report shipped in `eval/`. `bialy audit heads` (from
{code_repo}) replays these heads through the Painted Wolf Code engine over the
dataset's held-out split, with the catalog and head budget this release
carries, and compares every metric with the shipped report.
"""

COLUMNS = (("tools P/R/F1", lambda r: "%.2f / %.2f / %.2f" % tuple(r["tools"]["_all"][k] for k in ("precision", "recall", "f1"))),
           ("macro R", lambda r: r["tools"]["_all"].get("macro_recall")),
           ("loads/turn", lambda r: r["tools"]["_all"].get("loads_per_turn")),
           ("need MRR", lambda r: r["requests"].get("mrr")))
EVIDENCE = {"development": "development.json", "holdout": "holdout-replay.json", "holdout-comparison": "holdout.json",
            "rerank-development": "rerank-development.json", "qualification": "qualification.json"}


class ReleaseError(ValueError):
    pass


def head_header(path):
    """A safetensors head's metadata (label, backbone, training facts) without loading it."""
    with open(path, "rb") as fh:
        size = struct.unpack("<Q", fh.read(8))[0]
        return json.loads(fh.read(size)).get("__metadata__", {})


def results_table(evals):
    """One row per replayed set."""
    lines = ["| Set | " + " | ".join(name for name, _ in COLUMNS) + " |", "|" + "---|" * (len(COLUMNS) + 1)]
    for name in sorted(evals):
        report = json.loads(Path(evals[name]).read_text(encoding="utf-8"))["overall"]
        cells = []
        for _, get in COLUMNS:
            try:
                value = get(report)
            except (KeyError, TypeError):
                value = None
            cells.append("-" if value is None else str(value))
        lines.append("| %s | %s |" % (name, " | ".join(cells)))
    return "\n".join(lines)


def rerank_table(measures):
    """One row per set of repositories: text-match and blended MRR and hit@1 pooled over
    their pairs, and how many requests the head moved up or down."""
    lines = ["| Repositories | Pairs | MRR text / blended | hit@1 text / blended | improved / regressed |", "|---|---|---|---|---|"]
    for name, m in measures.items():
        if not m.get("pairs"):
            lines.append("| %s | 0 | - | - | - |" % name)
            continue
        lines.append("| %s | %d | %.3f / %.3f | %.3f / %.3f | %d / %d |" % (
            name, m["pairs"], m["lexical"]["mrr"], m["blended"]["mrr"], m["lexical"]["hit_1"], m["blended"]["hit_1"],
            m["improved"], m["regressed"]))
    return "\n".join(lines)


def qualification_table(record):
    lines = ["| Head | Measure | Value | Floor |", "|---|---|---|---|"]
    for head, rules in record["bars"].items():
        lines += ["| `%s` | %s | %s | %s |" % (head, rule["metric"], record["measures"][head][rule["metric"]], rule["at_least"]) for rule in rules]
    return "\n".join(lines)


def qualified(selector_dir):
    """The intact policy, its head files, its qualification and its held-out comparison,
    or the reason the selector directory cannot be released."""
    selector_dir = Path(selector_dir)
    try:
        policy, heads = selector.frozen(selector_dir)
        record = selector.qualification(selector_dir)
    except (CampaignFault, OSError, ValueError) as exc:
        raise ReleaseError("%s holds no intact selector policy: %s" % (selector_dir, exc)) from exc
    if record is None:
        raise ReleaseError("the policy in %s has no recorded qualification; heads are released only after the parity probe "
                           "and every bar pass on the development split" % selector_dir)
    missing = [name for name in EVIDENCE.values() if not (selector_dir / name).is_file()]
    comparison = json.loads((selector_dir / "holdout.json").read_text(encoding="utf-8")) if not missing else {}
    if missing or comparison.get("policy") != policy["identity"]:
        raise ReleaseError("the policy in %s lacks its held-out comparison or evidence: %s" % (selector_dir, ", ".join(missing) or "holdout.json"))
    return policy, heads, record, comparison


def build(selector_dir, version, dataset_version, engine_commit, out_dir):
    """The release of the qualified policy in `selector_dir`: its heads, the policy, the
    catalog its thresholds are written into, the card, and the reports behind the card."""
    selector_dir, out_dir = Path(selector_dir), Path(out_dir)
    policy, heads, record, comparison = qualified(selector_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    settings = config.hub()
    # The Bialy commit packaging the release; publishing tags it heads-<version> (docs/releases.md).
    factory = factory_source()
    shipped = []
    for name, path in sorted(heads.items()):
        shutil.copy2(path, out_dir / path.name)
        shipped.append({"file": path.name, "sha256": policy["heads"][name]["sha256"], "metadata": head_header(path)})
    shutil.copy2(selector_dir / "p0.json", out_dir / "policy.json")
    shutil.copy2(selector_dir / "corpus.json", out_dir / "corpus.json")
    (out_dir / "eval").mkdir(exist_ok=True)
    for name, source in EVIDENCE.items():
        shutil.copy2(selector_dir / source, out_dir / "eval" / (name + ".json"))
    development = json.loads((selector_dir / "rerank-development.json").read_text(encoding="utf-8"))
    card = CARD.format(version=version, backbone=policy["backbone"]["model"], dataset_repo=settings["dataset_repo"],
                       dataset_version=dataset_version, code_repo=settings["code_repo"], engine_commit=engine_commit,
                       factory_commit=factory["git_commit"], identity=policy["identity"],
                       heads="\n".join("- `%s`: %s, over %s, sha256 `%s`" % (h["file"], h["metadata"].get("label"), h["metadata"].get("backbone"),
                                                                            h["sha256"]) for h in shipped),
                       thresholds=", ".join("`%s` %s" % item for item in policy["thresholds"].items()),
                       tolerance=record["parity"]["tolerance"], parity_turns=record["parity"]["turns"], parity_gap=record["parity"]["worst_gap"],
                       qualification=qualification_table(record),
                       results=results_table({name: out_dir / "eval" / (name + ".json") for name in ("development", "holdout")}),
                       rerank=rerank_table({"development": development, "held-out": comparison["code_rank"]}))
    (out_dir / "README.md").write_text(card, encoding="utf-8")
    (out_dir / "NOTICE").write_text(NOTICE, encoding="utf-8")
    shutil.copy2(Path(__file__).resolve().parents[2] / "LICENSE", out_dir / "LICENSE")
    provenance = {"version": version, "dataset_version": dataset_version, "engine_commit": engine_commit, "factory": factory,
                  "policy": policy["identity"], "heads": shipped, "evals": sorted(EVIDENCE)}
    (out_dir / "PROVENANCE.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    hub.seal(out_dir)
    anchor.record("heads", version, out_dir)
    return out_dir
