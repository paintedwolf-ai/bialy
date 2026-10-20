"""Split judged rows into train, validation, and held-out workspaces.

Every row of a held-out workspace (a repository, or a greenfield stack) is held
out. The rest split by prompt group (near-duplicate prompts share one), so
validation measures requests the head never trained on, and a prompt driven
under two models, or its worker legs, never lands on both sides.
"""

import hashlib
import json
from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory


def side(group, seed, val_fraction):
    h = int(hashlib.sha256(("%s:%s" % (seed, group)).encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "val" if h < val_fraction else "train"


def split(factory, rows_paths, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    holdout = factory.holdout()
    known = {w.name for w in factory.workspaces()}
    names = ("train", "val", "holdout")
    counts = {name: 0 for name in names}
    groups = {name: set() for name in names}
    assignments = {}
    # Validate into temporary files so a bad row cannot publish partial splits
    # or overwrite the previously valid dataset.
    with TemporaryDirectory(prefix=".split-", dir=out_dir) as staging:
        staging = Path(staging)
        with ExitStack() as stack:
            files = {name: stack.enter_context(open(staging / (name + ".jsonl"), "w", encoding="utf-8")) for name in names}
            for path in rows_paths:
                with open(path, encoding="utf-8") as source:
                    for line in source:
                        if not line.strip():
                            continue
                        row = json.loads(line)
                        meta = row.get("meta") or {}
                        workspace, group = meta.get("workspace"), meta.get("prompt_group")
                        if not isinstance(workspace, str) or not workspace or not isinstance(group, str) or not group:
                            raise ValueError("row %s has no task workspace and prompt group; collect it again" % row.get("receipt"))
                        if workspace not in known:
                            raise ValueError("row from unknown workspace %r" % workspace)
                        name = "holdout" if workspace in holdout else side(group, factory.split["seed"], factory.split["val_fraction"])
                        prior = assignments.setdefault(group, name)
                        if prior != name:
                            raise ValueError("prompt group %r appears in both %s and %s splits" % (group, prior, name))
                        files[name].write(line if line.endswith("\n") else line + "\n")
                        counts[name] += 1
                        groups[name].add(group)
        for name in names:
            (staging / (name + ".jsonl")).replace(out_dir / (name + ".jsonl"))
    return {name: {"rows": counts[name], "prompt_groups": len(groups[name])} for name in names}
