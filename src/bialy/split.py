"""Split judged rows into train, validation, and held-out workspaces.

Every row of a held-out workspace (a repository, or a greenfield stack) is held
out. The rest split by prompt group (near-duplicate prompts share one), so
validation measures requests the head never trained on, and a prompt driven
under two models, or its worker legs, never lands on both sides.
"""

import hashlib
import json
from pathlib import Path


def side(group, seed, val_fraction):
    h = int(hashlib.sha256(("%s:%s" % (seed, group)).encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "val" if h < val_fraction else "train"


def split(factory, rows_paths, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    holdout = factory.holdout()
    known = {w.name for w in factory.workspaces()}
    files = {name: open(out_dir / ("%s.jsonl" % name), "w", encoding="utf-8") for name in ("train", "val", "holdout")}
    counts = {name: 0 for name in files}
    groups = {name: set() for name in files}
    try:
        for path in rows_paths:
            for line in open(path, encoding="utf-8"):
                if not line.strip():
                    continue
                row = json.loads(line)
                meta = row.get("meta") or {}
                workspace, group = meta.get("workspace"), meta.get("prompt_group")
                if not workspace or not group:
                    # Grouping by anything else would split near-duplicates apart.
                    raise ValueError("row %s has no task workspace and prompt group; collect it again" % row.get("receipt"))
                if workspace not in known:
                    raise ValueError("row from unknown workspace %r" % workspace)
                name = "holdout" if workspace in holdout else side(group, factory.split["seed"], factory.split["val_fraction"])
                files[name].write(line if line.endswith("\n") else line + "\n")
                counts[name] += 1
                groups[name].add(group)
    finally:
        for fh in files.values():
            fh.close()
    leaked = groups["train"] & (groups["val"] | groups["holdout"])
    if leaked:
        raise ValueError("%d prompt groups on both sides of the split" % len(leaked))
    return {name: {"rows": counts[name], "prompt_groups": len(groups[name])} for name in files}
