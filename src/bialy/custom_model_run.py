"""Pinned collection epochs and qualification boundaries for custom-model datasets."""

import dataclasses
import hashlib
import json
import re
import subprocess
import tempfile
from pathlib import Path

from . import host_bundle, provenance


class DatasetError(ValueError):
    pass


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as fh:
        json.dump(value, fh, indent=2, ensure_ascii=False)
        fh.write("\n")
        temporary = Path(fh.name)
    temporary.replace(path)


def validate_settings(cfg):
    if not cfg.get("enabled"):
        return
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9._-]{0,79}", str(cfg.get("epoch") or "")):
        raise DatasetError("run.custom_model.epoch must name this collection; use a new run and epoch after command rework")
    if not re.fullmatch(r"[0-9a-f]{40}", str(cfg.get("host_commit") or "")):
        raise DatasetError("run.custom_model.host_commit must pin the full Painted Wolf Code commit")
    from .student_host.specification import load
    spec = load()
    students = cfg.get("students") or {}
    if set(students) != set(spec['models']):
        raise DatasetError("run.custom_model.students must declare every pinned student")
    for name, serving in students.items():
        if (set(serving) != {"provider", "kind", "model", "base_url", "context_window"}
                or not all(serving.values()) or serving['kind'] != 'openai-compatible'
                or serving['model'] != spec['models'][name]['directory']
                or serving['context_window'] != spec['context']['max_sequence_tokens']):
            raise DatasetError("run.custom_model.students must match the pinned model and context policy")
    if len({item['provider'] for item in students.values()}) != len(students):
        raise DatasetError("run.custom_model.students must use distinct providers")


def context_window(cfg):
    """The one context window every student is served with."""
    windows = {int(serving["context_window"]) for serving in cfg["students"].values()}
    if len(windows) != 1:
        raise DatasetError("run.custom_model.students must share one context window")
    return windows.pop()


def student_binding(identity, serving):
    return {**identity, 'student_provider': serving['provider'], 'student_kind': serving['kind'],
            'student_model': serving['model'], 'student_context_window': serving['context_window']}


def binding(run):
    """Refuse reuse after changing the collection's source or configuration."""
    cfg = run.setting("custom_model") or {}
    validate_settings(cfg)
    result = subprocess.run(["git", "-C", str(run.checkout), "rev-parse", "HEAD"], capture_output=True, text=True)
    if result.returncode or result.stdout.strip() != cfg["host_commit"]:
        raise DatasetError("Painted Wolf Code checkout does not match custom_model.host_commit")
    status = subprocess.run(["git", "-C", str(run.checkout), "status", "--porcelain", "--untracked-files=normal"], capture_output=True, text=True)
    if status.returncode or status.stdout.strip():
        raise DatasetError("custom-model collection requires a clean pinned Painted Wolf Code checkout")
    if not (run.checkout / "scripts/bialy/custom_model/collect.py").is_file():
        raise DatasetError("pinned host lacks the exact capture and receipt exporter")
    try:
        _, contract_sha256 = host_bundle.contract(run.image("off"), run.path("host-contract"), cfg["host_commit"])
    except host_bundle.HostEvidenceError as exc:
        raise DatasetError("pinned host image cannot state its training contract: " + exc.code) from exc
    identity = {
        "schema": "pw-custom-run/2", "epoch": cfg["epoch"], "contract_sha256": contract_sha256,
        "host_commit": cfg["host_commit"], "settings_sha256": digest(run.settings),
        "students": cfg["students"],
        "factory": provenance.factory_source(),
        "models": [dataclasses.asdict(m) for m in run.factory.models],
        "split": run.factory.split,
        "workspaces": [dataclasses.asdict(w) for w in run.factory.workspaces()],
    }
    identity = json.loads(json.dumps(identity))
    path = run.path("custom-model-binding.json")
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != identity:
            raise DatasetError("custom-model run binding changed; use a new run name and epoch instead of resuming or --from")
    elif run.load_state().get("stages"):
        raise DatasetError("existing run has no custom-model binding; recollect in a new pinned run")
    else:
        write_json(path, identity)
    return identity
