"""Relabel guide units in archived rows from a corpus whose units declare `needed_with`.

A row's guide labels come from the tools its turn called: a unit was needed when the
turn called one of the tools it attaches to or one it is needed with. Archived rows keep
only the loadable tools a turn called (`labels.tools`), but every floor tool a unit
attaches to is already recorded through that unit's own label, so the turn's call set is
reconstructed exactly: the loadable calls plus the attached tools of every unit labelled
true. Units the corpus does not know keep their archived label.

Usage: relabel_guides.py CORPUS ROWS OUT
"""
import argparse
import json
from pathlib import Path


def called_tools(row, units):
    """The tools a row's turn called, as far as its labels show."""
    called = set(row["labels"].get("tools") or [])
    for uid, needed in (row["labels"].get("guides") or {}).items():
        if needed and uid in units:
            called.update(units[uid].get("attaches") or [])
    return called


def relabel(row, units):
    guides = dict(row["labels"].get("guides") or {})
    if row.get("partial") or not guides:
        return row, 0
    called = called_tools(row, units)
    changed = 0
    for uid in guides:
        unit = units.get(uid)
        if unit is None:
            continue
        attaches = unit.get("attaches") or []
        needed_with = unit.get("needed_with") or []
        if not attaches and not needed_with:
            continue
        label = any(tool in called for tool in attaches) or any(tool in called for tool in needed_with)
        if guides[uid] != label:
            changed += 1
        guides[uid] = label
    row["labels"]["guides"] = guides
    return row, changed


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("corpus")
    ap.add_argument("rows")
    ap.add_argument("out")
    args = ap.parse_args()
    corpus = json.loads(Path(args.corpus).read_text())
    units = {u["id"]: u for u in corpus["units"]}
    counts = {}
    total = changed_rows = 0
    with Path(args.rows).open() as src, Path(args.out).open("x") as dst:
        for line in src:
            if not line.strip():
                continue
            row, changed = relabel(json.loads(line), units)
            total += 1
            changed_rows += bool(changed)
            for uid, label in (row["labels"].get("guides") or {}).items():
                key = (uid, "none" if label is None else ("true" if label else "false"))
                counts[key] = counts.get(key, 0) + 1
            dst.write(json.dumps(row) + "\n")
    print("rows %d, rows with a changed label %d" % (total, changed_rows))
    for uid in sorted({uid for uid, _ in counts}):
        print("  %-28s true=%4d false=%4d none=%4d" % (uid, counts.get((uid, "true"), 0), counts.get((uid, "false"), 0), counts.get((uid, "none"), 0)))


if __name__ == "__main__":
    main()
