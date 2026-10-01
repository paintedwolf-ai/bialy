"""Assemble a dataset release: splits, card, provenance, and checksums, locally.

Nothing here uploads. A release directory is what a person publishes when
they choose to.
"""

import json
import shutil
import time
from pathlib import Path

from . import anchor, audit, config, hub, repos
from .provenance import file_sha256

CARD = """---
license: apache-2.0
language: [en, de, fr, es, pt, it, ja, zh, ko]
pretty_name: Painted Wolf Code decision-head dataset
size_categories: [1K<n<10K]
task_categories: [text-classification]
tags: [coding-agents, tool-selection, synthetic]
configs:
  - config_name: default
    data_files:
      - {{split: train, path: train.jsonl}}
      - {{split: validation, path: val.jsonl}}
      - {{split: test, path: holdout.jsonl}}
---

# Painted Wolf Code decision-head dataset {version}

Training rows for the local decision heads of Painted Wolf Code: at each turn
of a coding-agent session, which loadable tool schemas the turn needed, which
instruction units it needed, what kind of work it was, and, from an LLM
judge, how relevant each skill and tool card was to the request.

## How the rows were made

Labels come from coding-agent sessions coordinated by open-weights models,
scored by an open-weights LLM judge. Every session ran in a sandboxed runner,
in one of two kinds of workspace: a public, permissively licensed repository
at a pinned commit, on a request an open-weights model wrote from facts about
that repository, or an empty directory for a greenfield stack, on a request it
wrote to start a project from a drawn idea and scale. Guide
and kind labels are what the session did. Skill, tool, and need scores are
the judge's 0..4 levels, from a model of another family than the one that
drove the session. A turn's tool scores (`labels.tool_scores`) rate every
loadable tool it offered against the request: driving models call tools out
of habit, so the tools a session called (`labels.tools`) overstate what the
request needed. No row comes from a person's private session.

- Session and judge models: {models}
- Repositories: {repos}
- Greenfield stacks: {stacks}
- Held out (repositories and stacks): {holdout}
- Row schema: `pw-decide-row/1`, shipped as `row.schema.json`; the corpus the rows are read with, as `corpus.json`
- Made by: {code_repo} {code_ref}; sessions ran on Painted Wolf Code at commit {engine_commit}
- Second-judge agreement (quadratic-weighted kappa): {kappa}

## Splits

{splits}

## Code-rank pairs

`coderank/` holds the requests the code-rank head trains and is measured on.
`docs-<repo>.jsonl` takes each code unit's leading comment as the request;
`model-<repo>.jsonl` holds requests the generator models wrote for a unit. No
request names its unit, since text matching already finds a named one. Each
pair points at its unit by file, line, and symbol at the pinned commit;
`bialy coderank` harvests the units and candidate sets from the
repositories again.

## Tasks

`tasks.jsonl` is every request the sessions were driven with: the prompt,
follow-ups, the workspace it ran in, the workflow it started, the model that
drove it, and how it ended; a greenfield task also carries the project idea and
scale its request was written from (`meta.seed`). {stopping}A rebuild drives
exactly these tasks through the pinned models and engine.

## Check it yourself

`bialy audit dataset --release .` (from {code_repo}) recomputes every number
above from the rows, validates each row against `row.schema.json`, checks that
no prompt group crosses splits and that held-out workspaces appear only in
the test split, and recomputes the judge agreement from `agreement.jsonl`. With
an endpoint serving one of the pinned judges, `bialy audit dataset --rejudge`
scores a sample of rows again and reports how often the scores are reproduced.

Rows carry request text written by a model and the names of tools, units,
and skills. Requests can quote a repository's identifiers, file paths, and
error output; no repository file is included. Each repository keeps its own
license (listed in PROVENANCE.json, with its text under `licenses/`); the rows
and this card are released under Apache-2.0.
"""


def recipe(stages):
    """What a reader needs to make the rows again, from the provenance each stage wrote:
    the task seed, the engine build and serving the sessions ran on, the judge's corpus,
    and the configuration each stage read. Where and when a stage ran is not part of it."""
    config_of = lambda st: st.get("config") or {}  # noqa: E731
    return {
        "task_generation": [{"seed": st.get("seed"), "counts": st.get("counts"), "serving": st.get("serving"), "config": config_of(st)}
                            for st in stages if st.get("stage") == "tasks"],
        "sessions": [{"engine": st.get("engine") or {}, "serving": st.get("serving"), "decide_env": st.get("decide_env"),
                      "config": config_of(st)} for st in stages if st.get("stage") == "fleet"],
        "judging": [{"units": st.get("units"), "corpus_revision": st.get("corpus_revision"), "corpus_sha256": st.get("corpus_sha256"), "serving": st.get("serving"),
                     "config": config_of(st)} for st in stages if st.get("stage") == "judge"],
        "code_rank_pairs": [{"step": st["stage"].split("-", 1)[1], "counts": st.get("counts"), "per_repo": st.get("per_repo"),
                             "seed": st.get("seed"), "decide_rerank_sha256": st.get("decide_rerank"), "config": config_of(st)}
                            for st in stages if str(st.get("stage", "")).startswith("coderank-")],
    }


def build(factory, split_dir, version, out_dir, schema, corpus, code_ref, agreement=(), stages=(), driven=(), coderank=None,
          stopping=""):
    """A dataset release: the splits, the row schema and corpus they are read with, the
    second-judge sample, the code-rank pairs, a card, provenance, and checksums. Every
    statistic the card and provenance state is counted from the rows, as `bialy audit`
    recounts it."""
    split_dir, out_dir = Path(split_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stats = {}
    splits = []
    for split, name in audit.SPLITS:
        shutil.copy2(split_dir / name, out_dir / name)
        stats[split] = audit.split_stats(audit.load_rows(out_dir / name))
        splits.append("- `%s`: %d rows (%s)" % (name, stats[split]["rows"], ", ".join("%s %d" % kv for kv in stats[split]["hosts"].items())))
    shutil.copy2(schema, out_dir / "row.schema.json")
    shutil.copy2(corpus, out_dir / "corpus.json")
    # Every judge pass's second-judge sample, one record per doubly judged card.
    with open(out_dir / "agreement.jsonl", "w", encoding="utf-8") as fh:
        for path in agreement or ():
            fh.write(Path(path).read_text(encoding="utf-8"))
    kappa, pairs = audit.agreement_kappa(out_dir / "agreement.jsonl")
    for path in sorted(Path(coderank).glob("*.jsonl")) if coderank else []:
        (out_dir / "coderank").mkdir(exist_ok=True)
        shutil.copy2(path, out_dir / "coderank" / path.name)
    # Every task the runs drove, with its outcome: a rebuild drives exactly these.
    with open(out_dir / "tasks.jsonl", "w", encoding="utf-8") as fh:
        for path in driven:
            for line in Path(path).read_text(encoding="utf-8").splitlines():
                if line.strip():
                    fh.write(line + "\n")
    provenance = {
        "version": version, "built": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "models": [{"id": m.id, "hf": m.hf, "revision": m.revision, "license": m.license, "roles": list(m.roles)} for m in factory.models],
        "repositories": [{"name": r.name, "url": r.url, "commit": r.commit, "license": r.spdx, "split": r.split} for r in factory.repos],
        "stacks": [{"name": s.name, "language": s.language, "brief": s.brief, "split": s.split} for s in factory.stacks],
        "stats": stats,
        "judge_agreement": {"weighted_kappa": kappa, "pairs": pairs},
        "tasks": audit.task_stats(out_dir / "tasks.jsonl"),
        "stopping": stopping or None,
        "row_schema_sha256": file_sha256(out_dir / "row.schema.json"),
        "corpus_revision": json.loads((out_dir / "corpus.json").read_text(encoding="utf-8")).get("catalog_revision"),
    }
    provenance["recipe"] = recipe([json.loads(Path(p).read_text(encoding="utf-8")) for p in stages])
    (out_dir / "PROVENANCE.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    engine = next((run["engine"] for run in provenance["recipe"]["sessions"]), {})
    card = CARD.format(version=version, code_repo=config.hub().get("code_repo", ""), code_ref=code_ref,
                       engine_commit=engine.get("lycaon_commit"), models=", ".join("%s (%s)" % (m.hf, m.license) for m in factory.models),
                       repos=", ".join(r.name for r in factory.repos if r.split == "train"),
                       stacks=", ".join(s.name for s in factory.stacks if s.split == "train"),
                       holdout=", ".join(sorted(factory.holdout())),
                       kappa="%s over %d doubly judged cards" % (kappa, pairs), splits="\n".join(splits),
                       stopping=(stopping.rstrip(".") + ". ") if stopping else "")
    (out_dir / "README.md").write_text(card, encoding="utf-8")
    shutil.copy2(Path(__file__).resolve().parents[2] / "LICENSE", out_dir / "LICENSE")
    # Requests quote the repositories' identifiers, paths, and error output.
    for repo in factory.repos:
        for name in repo.license_files:
            target = out_dir / "licenses" / repo.name / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(repos.license_text(factory, repo, name))
    hub.seal(out_dir)
    anchor.record("dataset", version, out_dir, holdout=sorted(factory.holdout()))
    return out_dir
