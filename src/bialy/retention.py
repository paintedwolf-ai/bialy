"""Sessions shaped for retention observations, and one pass's requests compared with
and without the engine."""
import dataclasses
import json
import statistics
from pathlib import Path

# How each measure is read with the head deciding: prompt tokens must fall, a rise
# in repeated calls counts against it, and the rest are reported.
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


def load_rows(path):
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
