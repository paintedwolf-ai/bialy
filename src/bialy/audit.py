"""Check a release from its own contents, without the factory, a GPU fleet, or trust.

Everything a release claims is recomputed from the files it ships:

- every row matches the row schema the release carries (`row.schema.json`)
- every statistic on the card and in PROVENANCE.json (rows, hosts, languages,
  prompt groups per split) is recounted from the rows
- the split does not leak: no prompt group on two sides, and every held-out
  repository only in the test split
- the judge agreement is recomputed from the second-judge sample it ships
  (`agreement.jsonl`)

With an OpenAI-compatible endpoint (a single consumer GPU, or a CPU with
patience, serving one of the pinned judges), `rejudge` scores a random sample
of rows again with the judge's exact prompt and reports how often the release's
scores are reproduced.
"""

import collections
import json
import random
from pathlib import Path

from . import anchor, judge
from .hub import check

SPLITS = (("train", "train.jsonl"), ("validation", "val.jsonl"), ("test", "holdout.jsonl"))


class AuditError(ValueError):
    pass


def load_rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def split_stats(rows):
    """What a card states about one split, counted from its rows."""
    return {
        "rows": len(rows),
        "hosts": dict(sorted(collections.Counter(r["host"] for r in rows).items())),
        "languages": dict(sorted(collections.Counter(r["lang"] for r in rows).items())),
        "partial": sum(1 for r in rows if r["partial"]),
        "prompt_groups": len({r["meta"]["prompt_group"] for r in rows}),
        "repositories": dict(sorted(collections.Counter(r["meta"]["repo"] for r in rows).items())),
    }


def task_stats(path):
    """How the driven tasks ended, by outcome."""
    tasks = load_rows(path)
    return {"driven": len(tasks), "outcomes": dict(sorted(collections.Counter(t["outcome"]["status"] for t in tasks).items()))}


def check_tasks(release, rows_by_split):
    """Every row comes from a task the release lists, and the list matches its provenance."""
    tasks = load_rows(Path(release) / "tasks.jsonl")
    ids = {t["id"] for t in tasks}
    for split, rows in rows_by_split.items():
        for row in rows:
            task_id = (row.get("meta") or {}).get("task_id")
            if task_id not in ids:
                raise AuditError("a %s row comes from task %r, which tasks.jsonl does not list" % (split, task_id))
    return task_stats(Path(release) / "tasks.jsonl")


def agreement_kappa(path):
    records = load_rows(path) if Path(path).exists() else []
    return judge.weighted_kappa([(r["first_score"], r["second_score"]) for r in records]), len(records)


def validate_schema(release, rows_by_split):
    from jsonschema import Draft202012Validator

    schema = json.loads((Path(release) / "row.schema.json").read_text(encoding="utf-8"))
    validator = Draft202012Validator(schema)
    for split, rows in rows_by_split.items():
        for n, row in enumerate(rows, 1):
            error = next(iter(validator.iter_errors(row)), None)
            if error is not None:
                raise AuditError("%s row %d does not match row.schema.json: %s" % (split, n, error.message))


def check_task_meta(rows_by_split):
    """Every row names its task, repository, and prompt group, which the leakage check groups by."""
    for split, rows in rows_by_split.items():
        for n, row in enumerate(rows, 1):
            meta = row.get("meta") or {}
            if not (meta.get("task_id") and meta.get("repo") and meta.get("prompt_group")):
                raise AuditError("%s row %d has no task, repository, or prompt group" % (split, n))


def check_leakage(rows_by_split, holdout_repos):
    groups = {split: {r["meta"]["prompt_group"] for r in rows} for split, rows in rows_by_split.items()}
    names = list(groups)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            shared = groups[a] & groups[b]
            if shared:
                raise AuditError("%d prompt groups appear in both %s and %s" % (len(shared), a, b))
    for split, rows in rows_by_split.items():
        repos = {r["meta"]["repo"] for r in rows}
        if split == "test" and repos - holdout_repos:
            raise AuditError("test split has rows from training repositories: %s" % ", ".join(sorted(repos - holdout_repos)))
        if split != "test" and repos & holdout_repos:
            raise AuditError("%s split has rows from held-out repositories: %s" % (split, ", ".join(sorted(repos & holdout_repos))))


def dataset(release, anchored=True):
    """Every check a consumer can run offline; raises AuditError on the first failure.

    The held-out repositories come from the release's anchor, which the release
    cannot rewrite; the provenance must state the same set."""
    release = Path(release)
    check("dataset", release, anchored)
    provenance = json.loads((release / "PROVENANCE.json").read_text(encoding="utf-8"))
    stated = {r["name"] for r in provenance["repositories"] if r["split"] == "holdout"}
    holdout = set(anchor.match("dataset", release)["holdout"]) if anchored else stated
    if stated != holdout:
        raise AuditError("PROVENANCE.json holds out %s; the release's anchor holds out %s" % (sorted(stated), sorted(holdout)))
    rows_by_split = {split: load_rows(release / name) for split, name in SPLITS}
    validate_schema(release, rows_by_split)
    check_task_meta(rows_by_split)
    stats = {split: split_stats(rows) for split, rows in rows_by_split.items()}
    if provenance.get("stats") != stats:
        raise AuditError("PROVENANCE.json statistics differ from the rows: claimed %s, counted %s" % (provenance.get("stats"), stats))
    card = (release / "README.md").read_text(encoding="utf-8")
    for split, name in SPLITS:
        if ("`%s`: %d rows" % (name, stats[split]["rows"])) not in card:
            raise AuditError("the card's row count for %s differs from the rows" % name)
    check_leakage(rows_by_split, holdout)
    tasks = check_tasks(release, rows_by_split)
    if provenance.get("tasks") != tasks:
        raise AuditError("PROVENANCE.json task outcomes differ from tasks.jsonl: claimed %s, counted %s" % (provenance.get("tasks"), tasks))
    kappa, pairs = agreement_kappa(release / "agreement.jsonl")
    claimed = (provenance.get("judge_agreement") or {})
    if claimed.get("weighted_kappa") != kappa or claimed.get("pairs") != pairs:
        raise AuditError("second-judge agreement differs: claimed %s, recomputed kappa %s over %d pairs" % (claimed, kappa, pairs))
    return {"splits": stats, "tasks": tasks, "judge_agreement": {"weighted_kappa": kappa, "pairs": pairs}, "leakage": "none"}


def rejudge(release, base_url, model, sample=50, seed=7, units=("skills",)):
    """Score a random sample of released rows again through an endpoint serving `model`,
    with the judge's own prompt, candidate order, and decoding, and report agreement
    with the release's scores."""
    from .llm import Chat

    release = Path(release)
    corpus = json.loads((release / "corpus.json").read_text(encoding="utf-8"))
    cards = {s["name"]: s["card"] for s in corpus["skills"]}
    rows = [r for _, name in SPLITS for r in load_rows(release / name)
            if not r["partial"] and r["labels"].get("skill_scores") and (r.get("judge") or {}).get("model") == model]
    if not rows:
        raise AuditError("no released rows were judged by %s" % model)
    picked = random.Random(seed).sample(rows, min(sample, len(rows)))
    chat = Chat(base_url, model)
    pairs = []
    for row in picked:
        again = judge.score_cards(chat, row["state"]["user"], cards, judge.order_seed(row, "skills"))
        released = row["labels"]["skill_scores"]
        pairs += [(released[k], again[k]) for k in released if k in again]
    exact = sum(a == b for a, b in pairs) / len(pairs)
    return {"model": model, "rows": len(picked), "pairs": len(pairs), "exact": round(exact, 4), "weighted_kappa": judge.weighted_kappa(pairs)}


METRICS = (("tools", "precision"), ("tools", "recall"), ("tools", "f1"), ("guides", "omission_precision"), ("kind", "accuracy"))


def heads(release, dataset_release, lycaon, engine, split="holdout", tolerance=0.01):
    """Replay the released heads over the dataset's held-out split with the engine and
    compare every headline metric with the report the heads release ships for that set.

    `engine` is a bialy launcher loading the released heads (`bialy serve --device
    cpu --head turn-load=<release>/turn-load.safetensors ...`); `lycaon` is a Painted Wolf
    Code checkout at the heads' `engine_commit`."""
    import subprocess
    import sys
    import tempfile

    release, dataset_release = Path(release), Path(dataset_release)
    check("heads", release)
    shipped = json.loads((release / "eval" / ("%s.json" % split)).read_text(encoding="utf-8"))["overall"]
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "replay.json"
        subprocess.run([sys.executable, str(Path(lycaon) / "scripts/bialy/replay_eval.py"), "--corpus", str(dataset_release / "corpus.json"),
                        "--examples", str(dataset_release / "holdout.jsonl"), "--engine", str(engine),
                        "--holdout-pack", "painted-wolf/browser", "--json", str(out)],
                       check=True, stdout=subprocess.DEVNULL)
        replayed = json.loads(out.read_text(encoding="utf-8"))["overall"]
    diffs = {}
    for family, metric in METRICS:
        a = (shipped.get(family, {}).get("_all") or shipped.get(family, {})).get(metric)
        b = (replayed.get(family, {}).get("_all") or replayed.get(family, {})).get(metric)
        if a is None and b is None:
            continue
        if a is None or b is None or abs(a - b) > tolerance:
            raise AuditError("%s %s: shipped %s, replayed %s" % (family, metric, a, b))
        diffs["%s.%s" % (family, metric)] = {"shipped": a, "replayed": b}
    return {"split": split, "metrics": diffs, "tolerance": tolerance}
