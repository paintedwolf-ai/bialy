"""Context retention: how a run shapes its sessions for the retention head,
searches and selects the head, compares the passes it decided in, and reads
its verdict.

The retention head answers which earlier tool results a request will read
again (Painted Wolf Code, scripts/bialy/README.md "Context retention"). Its
rows need no judge. The replay (`retention_eval.py`) prices every release in
prompt tokens from the rows' own facts, fits the drop threshold and confidence
floor on validation, and scores the fitted policy on the held-out workspaces
with a bootstrap interval; the engine-on pass then shows what the head did
to real sessions. The verdict reads both: a head is viable when its fitted
policy's held-out objective is clear of zero and of the structural facts
alone, and when the sessions it decided in did not pay for it in recalls.
"""
import dataclasses
import itertools
import json
import statistics
from pathlib import Path

# The behaviour measures compared between the engine-off and engine-on passes,
# per request, with the direction a better head moves each.
BEHAVIOUR = (("prompt_tokens", "down"), ("recall_calls", "up"), ("repeated_calls", "up"), ("calls", "flat"))


def shape_archetypes(archetypes, settings):
    """The archetypes under run.retention: every follow-up rate and follow-up cap
    replaced when the run sets them, so a pass's tasks continue with the follow-ups
    that make retention decisions."""
    rate, cap = settings.get("follow_up_rate"), settings.get("follow_ups")
    out = []
    for a in archetypes:
        changes = {}
        if rate is not None:
            changes["follow_up_rate"] = float(rate)
        if cap is not None:
            changes["max_follow_ups"] = int(cap)
        out.append(dataclasses.replace(a, **changes) if changes else a)
    return out


def search_grid(settings):
    """One trainer argument list per point of run.retention.search, in a fixed order,
    each with the label suffix that names it."""
    grid = settings.get("search") or {}
    axes = [("lr", grid.get("lr") or [5e-4]), ("pos_weight", grid.get("pos_weight") or [6.0]), ("truth", grid.get("truth") or ["needed"])]
    points = []
    for values in itertools.product(*(v for _, v in axes)):
        point = dict(zip([k for k, _ in axes], values, strict=True))
        name = "lr%g-pw%g-%s" % (point["lr"], point["pos_weight"], point["truth"])
        points.append((name, ["--lr", str(point["lr"]), "--pos-weight", str(point["pos_weight"]), "--retention-truth", point["truth"]]))
    return points


def select(reports):
    """The search point with the best fitted objective on validation: {name: report}
    -> (name, objective). Ties go to the point whose fitted policy released less."""
    best = None
    for name, report in sorted(reports.items()):
        fitted = (report.get("fitted") or {}).get("tally") or {}
        key = (fitted.get("objective_tokens", float("-inf")), -fitted.get("released", 0))
        if best is None or key > best[1]:
            best = (name, key)
    return (best[0], best[1][0]) if best else (None, None)


def compare(off_rows, on_rows):
    """The engine-on pass against the engine-off pass over the requests both drove,
    matched by task (meta.task_id) and position in the session: per-request means
    of every behaviour measure, their deltas, and how often the head released
    anything. Requests one pass lacks are left out, so a task that failed in one
    pass does not tilt the comparison."""
    def key(r):
        return ((r.get("meta") or {}).get("task_id"), r["host"], r.get("surface"), r.get("request_index"))
    off = {key(r): r for r in off_rows if not r["partial"]}
    on = {key(r): r for r in on_rows if not r["partial"]}
    shared = sorted(set(off) & set(on), key=str)
    out = {"requests": len(shared), "off_only": len(set(off) - set(on)), "on_only": len(set(on) - set(off)), "measures": {}}
    if not shared:
        return out
    for measure, direction in BEHAVIOUR:
        a = [off[k][measure] for k in shared]
        b = [on[k][measure] for k in shared]
        delta = [y - x for x, y in zip(a, b, strict=True)]
        out["measures"][measure] = {"off": round(statistics.fmean(a), 2), "on": round(statistics.fmean(b), 2), "delta": round(statistics.fmean(delta), 2),
                                    "delta_median": statistics.median(delta), "wants": direction}
    asked = [on[k] for k in shared if on[k]["retention"]["asked"]]
    out["asked"] = len(asked)
    out["released_requests"] = sum(1 for r in asked if r["retention"]["dropped"])
    out["released_results"] = sum(r["retention"]["dropped"] for r in asked)
    return out


def viability(replay, comparison, settings):
    """The verdict: {viable, reasons, numbers}, from the held-out replay of the
    selected head and the pass comparison when one ran. Every reason that fails is
    named, so a reader sees the whole picture rather than the first miss."""
    held = (replay or {}).get("holdout") or {}
    boot = ((held.get("bootstrap") or {}).get("engine")) or {}
    structural = held.get("structural") or {}
    best_fact = max((t.get("objective_tokens", 0) for name, t in structural.items() if name != "none"), default=0)
    numbers = {
        "rows": held.get("rows"), "candidates": (held.get("engine") or {}).get("candidates"), "needed_rate": held.get("needed_rate"),
        "auc": held.get("auc"), "objective_per_request": (held.get("engine") or {}).get("objective_per_request"),
        "objective_share": (held.get("engine") or {}).get("objective_share"), "interval": boot,
        "structural_best_objective": best_fact, "released": (held.get("engine") or {}).get("released"),
        "needed_released": (held.get("engine") or {}).get("needed_released"), "abstained": held.get("abstained"),
        "policy": held.get("policy"), "facts": ((replay or {}).get("recommendation") or {}).get("facts"),
    }
    reasons = []
    min_rows = int(settings.get("min_rows") or 20)
    if (numbers["rows"] or 0) < min_rows:
        reasons.append("%s held-out rows replayed; an interval needs %d" % (numbers["rows"], min_rows))
    if not boot or boot.get("low", 0) <= 0:
        reasons.append("held-out objective interval %s is not clear of releasing nothing" % json.dumps(boot or None))
    engine_objective = (held.get("engine") or {}).get("objective_tokens", 0)
    if engine_objective <= best_fact:
        reasons.append("held-out objective %s tokens does not beat the best structural fact's %s" % (engine_objective, best_fact))
    if held.get("abstained"):
        reasons.append("%d held-out rows abstained: the head did not load" % held["abstained"])
    if comparison:
        measures = comparison.get("measures") or {}
        if comparison.get("requests", 0) < min_rows:
            reasons.append("%s requests matched between the passes; the comparison needs %d" % (comparison.get("requests"), min_rows))
        for measure, direction in BEHAVIOUR:
            m = measures.get(measure)
            if not m:
                continue
            if direction == "up" and m["delta"] > 0:
                reasons.append("%s rose by %s per request with the head deciding" % (measure, m["delta"]))
            if direction == "down" and m["delta"] >= 0 and comparison.get("released_results"):
                reasons.append("%s did not fall with the head deciding (%s)" % (measure, m["delta"]))
        numbers["comparison"] = comparison
    return {"viable": not reasons, "reasons": reasons, "numbers": numbers}


def load_rows(path):
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
