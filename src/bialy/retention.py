"""Compaction dataset experiments and held-out retention evaluation."""
import dataclasses
import itertools
import json
import statistics
from pathlib import Path

# Recall use is measured; successful recovery is part of the design.
BEHAVIOUR = (("prompt_tokens", "down"), ("recall_calls", "flat"), ("repeated_calls", "up"), ("calls", "flat"))


def shape_archetypes(archetypes, settings):
    """Apply the requested follow-up rate and cap."""
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
    """Return deterministic trainer arguments for each search point."""
    grid = settings.get("search") or {}
    axes = [("lr", grid.get("lr") or [5e-4]), ("pos_weight", grid.get("pos_weight") or [6.0]), ("truth", grid.get("truth") or ["needed"])]
    points = []
    for values in itertools.product(*(v for _, v in axes)):
        point = dict(zip([k for k, _ in axes], values, strict=True))
        name = "lr%g-pw%g-%s" % (point["lr"], point["pos_weight"], point["truth"])
        points.append((name, ["--lr", str(point["lr"]), "--pos-weight", str(point["pos_weight"]), "--retention-truth", point["truth"]]))
    return points


def select(reports):
    """Select validation objective, preferring fewer releases on ties."""
    best = None
    for name, report in sorted(reports.items()):
        fitted = (report.get("fitted") or {}).get("tally") or {}
        key = (fitted.get("objective_tokens", float("-inf")), -fitted.get("released", 0))
        if best is None or key > best[1]:
            best = (name, key)
    return (best[0], best[1][0]) if best else (None, None)


def compare(off_rows, on_rows):
    """Compare matched complete requests and report unmatched coverage."""
    def key(r):
        return ((r.get("meta") or {}).get("task_id"), r["host"], r.get("surface"), r.get("request_index"))
    off = {key(r): r for r in off_rows if not r["partial"]}
    on = {key(r): r for r in on_rows if not r["partial"]}
    shared = sorted(set(off) & set(on), key=str)
    out = {"requests": len(shared), "off_only": len(set(off) - set(on)), "on_only": len(set(on) - set(off)), "measures": {}}
    out["partial_requests"] = sum(r["partial"] for r in off_rows + on_rows)
    out["verified_usage_requests"] = sum(off[k].get("usage_available") is True and on[k].get("usage_available") is True for k in shared)
    out["verified_outcome_requests"] = sum((off[k].get("meta") or {}).get("outcome_verified") is True and (on[k].get("meta") or {}).get("outcome_verified") is True for k in shared)
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
    """Require held-out benefit, verified labels, and recovery evidence."""
    held = (replay or {}).get("holdout") or {}
    boot = ((held.get("bootstrap") or {}).get("engine")) or {}
    structural = held.get("structural") or {}
    best_fact = structural.get("none", {}).get("objective_tokens", 0)
    numbers = {
        "verified_label_rows": held.get("verified_label_rows", 0),
        "rows": held.get("rows"), "candidates": (held.get("engine") or {}).get("candidates"), "needed_rate": held.get("needed_rate"),
        "auc": held.get("auc"), "objective_per_request": (held.get("engine") or {}).get("objective_per_request"),
        "objective_share": (held.get("engine") or {}).get("objective_share"), "interval": boot,
        "structural_best_objective": best_fact, "released": (held.get("engine") or {}).get("released"),
        "needed_released": (held.get("engine") or {}).get("needed_released"), "abstained": held.get("abstained"),
        "policy": held.get("policy"), "facts": [],
    }
    reasons = []
    min_rows = int(settings.get("min_rows") or 20)
    if numbers["verified_label_rows"] != numbers["rows"]:
        reasons.append("held-out labels lack verified task or judge provenance")
    if (numbers["rows"] or 0) < min_rows:
        reasons.append("%s held-out rows replayed; an interval needs %d" % (numbers["rows"], min_rows))
    if not boot or boot.get("low", 0) <= 0:
        reasons.append("held-out objective interval %s is not clear of releasing nothing" % json.dumps(boot or None))
    engine_objective = (held.get("engine") or {}).get("objective_tokens", 0)
    if engine_objective <= best_fact:
        reasons.append("held-out objective %s tokens does not beat the deterministic baseline's %s" % (engine_objective, best_fact))
    if (held.get("engine") or {}).get("needed_released", 0):
        reasons.append("needed material left the inline context without verified recovery")
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
        if comparison.get("asked", 0) < min_rows:
            reasons.append("matched runs do not demonstrate the retention head affected enough requests")
        if comparison.get("verified_usage_requests", 0) != comparison.get("requests", 0):
            reasons.append("matched requests lack complete provider-reported usage")
        if comparison.get("verified_outcome_requests", 0) != comparison.get("requests", 0):
            reasons.append("matched requests lack independently verified task outcomes")
        if comparison.get("off_only") or comparison.get("on_only") or comparison.get("partial_requests"):
            reasons.append("unmatched or partial requests leave comparison coverage incomplete")
        numbers["comparison"] = comparison
    else:
        reasons.append("provider savings and task continuity need a matched deployed comparison")
    return {"viable": not reasons, "reasons": reasons, "numbers": numbers}


def load_rows(path):
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
