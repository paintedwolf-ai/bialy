"""Pinned collection epochs and qualification boundaries for custom-model datasets."""

import dataclasses
import gzip
import hashlib
import json
import re
import subprocess
import sys
import tempfile
import uuid
from collections import Counter
from pathlib import Path

from . import custom_quality, dpo, provenance


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
    if cfg.get("tool_surface") not in ("legacy", "command-v1"):
        raise DatasetError("run.custom_model.tool_surface must be legacy or command-v1")
    if not re.fullmatch(r"[0-9a-f]{40}", str(cfg.get("host_commit") or "")):
        raise DatasetError("run.custom_model.host_commit must pin the full Painted Wolf Code commit")
    if cfg.get("preload_from_judge"):
        raise DatasetError("custom-model training requires observed offered tools; synthetic judge preloading is unsupported")
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
    """Refuse reuse after changing the collection's source, configuration, or surface."""
    cfg = run.setting("custom_model") or {}
    validate_settings(cfg)
    result = subprocess.run(["git", "-C", str(run.checkout), "rev-parse", "HEAD"], capture_output=True, text=True)
    if result.returncode or result.stdout.strip() != cfg["host_commit"]:
        raise DatasetError("Painted Wolf Code checkout does not match custom_model.host_commit")
    status = subprocess.run(["git", "-C", str(run.checkout), "status", "--porcelain", "--untracked-files=normal"], capture_output=True, text=True)
    if status.returncode or status.stdout.strip():
        raise DatasetError("custom-model collection requires a clean pinned Painted Wolf Code checkout")
    if not (run.checkout / "scripts/bialy/custom_model/collect.py").is_file() and not cfg.get("diagnostic_only"):
        raise DatasetError("pinned host lacks the exact capture and receipt exporter")
    identity = {
        "schema": "pw-custom-run/1", "epoch": cfg["epoch"], "tool_surface": cfg["tool_surface"],
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


def passes_for(run):
    """The fleet passes whose captures this collection consumes: every pass the run
    drives, or the subset custom_model.passes names (engine-on only, say, so the
    student never learns the engine-off habit of loading every tool itself)."""
    driven = ["on"] if run.engine_on_only else (["off", "on"] if run.engine_on else ["off"])
    chosen = (run.setting("custom_model") or {}).get("passes")
    if not chosen:
        return driven
    if not isinstance(chosen, list) or not chosen or any(p not in driven for p in chosen):
        raise DatasetError("run.custom_model.passes must name passes this run drives: %s" % ", ".join(driven))
    return [p for p in driven if p in chosen]


def input_manifest(run, identity):
    """Hash completed stores and their task/judge lineage, including SQLite WAL bytes."""
    state = run.load_state().get("stages") or {}
    files = [run.path("corpus.json"), run.bin_dir / "BUILD.json", run.bin_dir / "lycaon", run.bin_dir / "lycaon-debug"]
    passes = passes_for(run)
    for name in passes:
        suffix = "" if name == "off" else "_on"
        for stage in ("drive", "collect", "judge"):
            if state.get(stage + suffix, {}).get("status") != "done":
                raise DatasetError("custom-model extraction requires completed %s%s" % (stage, suffix))
        base = run.path("fleet-" + name)
        stores = sorted(base.glob("shards/*/store.db"))
        if not stores:
            raise DatasetError("custom-model extraction has no %s shard stores" % name)
        files += [base / "provenance.json", run.path("judged-" + name + ".jsonl")]
        for marker in sorted(base.glob("shards/*/task-aborted.json")):
            files += [marker, marker.parent / "store.db", marker.parent / "manifest.jsonl"]
        for store in stores:
            files += [store, store.parent / "manifest.jsonl"]
            files += sorted(p for p in store.parent.rglob("*") if p.is_file() and (
                p.parent.name == "training-captures" or (p.name == "store.db-wal" and p.stat().st_size > 0) or p.name.endswith(".request.json") or p.name == "model-requests.jsonl"))
    task_files = sorted(run.path("tasks").glob("*.jsonl"))
    if not task_files:
        raise DatasetError("custom-model extraction requires the original task files")
    files += task_files
    quality = (run.setting("custom_model") or {}).get("quality_reports")
    if quality:
        files.append(Path(quality).resolve())
    missing = [str(p.relative_to(run.dir)) if p.is_relative_to(run.dir) else str(p.resolve()) for p in files if not p.is_file()]
    if missing:
        raise DatasetError("custom-model inputs missing: " + ", ".join(missing))
    built = json.loads((run.bin_dir / "BUILD.json").read_text(encoding="utf-8"))
    if built.get("lycaon_commit") != identity["host_commit"]:
        raise DatasetError("runner build does not match the collection host commit")
    engine = provenance.engine(run.factory, run.bin_dir)
    for name in passes:
        recorded = json.loads(run.path("fleet-" + name, "provenance.json").read_text(encoding="utf-8"))
        if recorded.get("engine") != engine:
            raise DatasetError("%s fleet provenance does not match the pinned runner binaries" % name)
    return {"schema": "pw-custom-inputs/1", "binding_sha256": digest(identity),
            "files": {(str(p.relative_to(run.dir)) if p.is_relative_to(run.dir) else str(p.resolve())): provenance.file_sha256(p) for p in sorted(set(files))},
            "passes": passes, "catalog_sha256": provenance.file_sha256(run.path("corpus.json"))}


def qualify_inputs(paths, out, identity, inputs):
    """Only exact, host-validated rows from this epoch may enter training splits."""
    counts = Counter()
    with open(out, "w", encoding="utf-8") as fh:
        for path in paths:
            for line in Path(path).read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                counts["candidates"] += 1
                if not dpo.training_eligible(row):
                    counts["unqualified"] += 1
                    continue
                capture = row.get("request_capture") or {}
                expected = {"epoch": identity["epoch"], "tool_surface": identity["tool_surface"],
                            "host_revision": identity["host_commit"], "catalog_sha256": inputs["catalog_sha256"]}
                if any(capture.get(k) != v for k, v in expected.items()):
                    counts["foreign_epoch"] += 1
                    continue
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                counts["qualified"] += 1
    return dict(counts)


def write_receipt_lineage(source, out, lineage, excluded_stores):
    """Bind task lineage while excluding every candidate from aborted tasks."""
    excluded = 0
    with Path(out).open("w") as output:
        for line in Path(source).read_text().splitlines():
            row = json.loads(line)
            if row["source"]["store_sha256"] in excluded_stores:
                excluded += 1
                continue
            row["meta"] = lineage.get(row["session"], {})
            row["split_group"] = row["meta"].get("prompt_group")
            output.write(json.dumps(row) + "\n")
    return excluded


def collect(run, judge_factory=None):
    cfg = run.setting("custom_model") or {}
    if not cfg.get("enabled"):
        return {"skipped": "run.custom_model is off"}
    identity = binding(run)
    inputs = input_manifest(run, identity)
    base = run.path("custom_model", digest(inputs), uuid.uuid4().hex)
    base.mkdir(parents=True, exist_ok=True)
    write_json(base / "inputs.json", inputs)
    transcripts = base / "transcripts.jsonl.gz"
    # Re-export even on --redo: file existence is not evidence of matching stores.
    dpo.export_transcripts([run.path("fleet-" + name) for name in inputs["passes"]], run.path("tasks"), transcripts)
    if input_manifest(run, identity) != inputs:
        raise DatasetError("collection inputs changed during export; stop writers and collect again")
    sft_raw, dpo_raw = base / "sft.candidates.jsonl", base / "dpo.candidates.jsonl"
    stats = dpo.extract(transcripts, rows_paths=[run.path("judged-" + name + ".jsonl") for name in inputs["passes"]],
                        out_sft=sft_raw, out_dpo=dpo_raw, min_score=int(cfg.get("min_score", 3)), preload_from_judge=False,
                        max_routine_reads_per_session=cfg.get("max_routine_reads_per_session", 6),
                        drop_retired_command_rejections=identity["tool_surface"] == "command-v1")
    sources = {"sft": [sft_raw], "dpo": [dpo_raw]}
    exporter = run.checkout / "scripts/bialy/custom_model/collect.py"
    if exporter.is_file():
        sources = {"dpo": [dpo_raw]}
        for student, serving in cfg["students"].items():
            sources["sft-" + student] = [export_student(run, base / student, identity, serving, inputs, transcripts, stats, judge_factory)]
    workers, seed = int(run.setting("workers", 16)), int(run.setting("seed", 7))
    if cfg.get("repair"):
        out = base / "dpo.repair-proposals.jsonl"
        stats["repair"] = dpo.repair(run.factory, dpo_raw, out, teacher_id=cfg["teacher"], workers=workers, seed=seed)
        sources["dpo"] = [out]
    if cfg.get("shadow"):
        out = base / "dpo.shadow-candidates.jsonl"
        stats["shadow"] = dpo.shadow(run.factory, sft_raw, out, student_id=cfg["student"], workers=workers, seed=seed)
        sources["dpo"].append(out)
    stats.update(epoch=identity["epoch"], tool_surface=identity["tool_surface"], directory=str(base), qualification={})
    for kind, paths in sources.items():
        eligible = base / (kind + ".qualified-review.jsonl")
        stats["qualification"][kind] = qualify_inputs(paths, eligible, identity, inputs)
    if binding(run) != identity or input_manifest(run, identity) != inputs:
        raise DatasetError("collection inputs changed during processing; outputs are unqualified, collect again")
    stats["status"] = "not_ready"
    stats["training_consumable"] = False
    stats["reason"] = "No publishable training splits: diagnostic mode or missing receipt, quality, or split coverage"
    stats['reason'] = 'Diagnostic candidates require campaign episode admission and frozen partitions before training'
    stats["outputs"] = {str(p.relative_to(base)): provenance.file_sha256(p) for p in sorted(base.rglob("*"))
                        if p.is_file() and p.name != "report.json"}
    write_json(base / "report.json", stats)
    write_json(run.path("custom_model", "latest.json"), {"directory": str(base), "status": stats["status"], "inputs_sha256": digest(inputs)})
    return stats


def export_student(run, base, identity, serving, inputs, transcripts, stats, judge_factory):
    base.mkdir(parents=True, exist_ok=True)
    cfg = run.setting("custom_model")
    exporter = run.checkout / "scripts/bialy/custom_model/collect.py"
    binding_path = base / "binding.json"
    write_json(binding_path, student_binding(identity, serving))
    destination = base / "host-receipts"
    cmd = [sys.executable, str(exporter), "--binding", str(binding_path),
           "--catalog-sha256", inputs["catalog_sha256"], "--out", str(destination)]
    for name in inputs["passes"]:
        cmd += ["--fleet", str(run.path("fleet-" + name))]
    if cfg.get("quality_reports"):
        cmd += ["--quality", str(Path(cfg["quality_reports"]).resolve())]
    subprocess.run(cmd, check=True)
    # Historical candidates supply task lineage only, never acceptance.
    lineage = {}
    with gzip.open(transcripts, "rt", encoding="utf-8") as sessions:
        for line in sessions:
            session = json.loads(line)
            lineage[session["session_id"]] = {k: session.get(k) for k in
                ("workspace", "prompt_group", "task_id", "archetype", "model")}
    accepted = base / "sft.host-receipts.jsonl"
    aborted_stores = {inputs["files"][str(marker.parent.relative_to(run.dir) / "store.db")]
                      for name in inputs["passes"] for marker in run.path("fleet-" + name).glob("shards/*/task-aborted.json")}
    stats["excluded_aborted_task_candidates"] = write_receipt_lineage(destination / "sft.receipts.jsonl", accepted, lineage, aborted_stores)
    if cfg.get("quality_judge"):
        quality_path = base / "quality.jsonl"
        stats["quality"] = custom_quality.judge_rows(judge_factory or run.factory, accepted, transcripts, quality_path,
                                                    seed=int(run.setting("seed", 7)))
        reviewed_destination = base / "reviewed-receipts"
        subprocess.run([*cmd[:cmd.index('--out') + 1], str(reviewed_destination),
                        *cmd[cmd.index('--out') + 2:], '--quality', str(quality_path)], check=True)
        accepted = base / 'sft.reviewed.jsonl'
        write_receipt_lineage(reviewed_destination / 'sft.receipts.jsonl', accepted, lineage, aborted_stores)
    return accepted
