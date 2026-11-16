"""Train one head on this machine: the trainer from the pinned checkout in a venv the
run makes, over the accelerator it finds, under a time budget.

Recipes are named as scripts/bialy/README.md names them: B5 for turn-load, E4 for
unit-rank over the judged rows and the skill requests, and code-rank over the site
dumps of the train and validation repositories.
"""
import hashlib
import json
import os
import platform
import shutil
import subprocess
import time
from pathlib import Path

from . import host_contract

BACKBONE = ("convaiinnovations/laya-multilingual", "e4e9ddf21a7b1903b7acffd8814ad4307bf63a67")
TRAIN_HOST = Path(__file__).resolve().parents[2] / "train-host"
TORCH_INDEX = {"rocm": "https://download.pytorch.org/whl/rocm7.14", "cuda": None, "mps": None, "cpu": "https://download.pytorch.org/whl/cpu"}
TURN_ARGS = ["--tool-truth", "consensus", "--tool-weight", "none", "--tool-negatives", "24", "--lr", "5e-4", "--pos-weight", "6",
             "--seed", "11", "--batch-size", "64", "--epochs", "45", "--patience", "8"]
RANK_ARGS = ["--families", "skills,requests", "--skill-scored", "8", "--skill-zeros", "12", "--seed", "11", "--batch-size", "32",
             "--rank-levels", "skills-blended"]
RERANK_ARGS = ["--seed", "7"]
RECIPES = {
    "B5": {"head": "turn-load", "kind": "turn", "families": "tools"},
    "E4": {"head": "unit-rank", "kind": "rank"},
    "code-rank": {"head": "code-rank", "kind": "rerank"},
}


class TrainingError(RuntimeError):
    pass


def accelerator():
    """rocm, cuda, mps, or cpu, from what the host has."""
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        return "mps"
    if shutil.which("nvidia-smi"):
        return "cuda"
    if shutil.which("rocminfo") or Path("/dev/kfd").exists():
        return "rocm"
    return "cpu"


def torch_index(settings):
    return settings.get("torch_index") or TORCH_INDEX[accelerator()]


def venv_python(root, settings):
    """The run's trainer interpreter, made on first use with uv from the pinned requirements."""
    venv = Path(root) / "venvs" / "train"
    python = venv / "bin" / "python"
    if python.exists():
        return python
    requirements = Path(settings.get("requirements") or TRAIN_HOST / "requirements.txt")
    run(["uv", "venv", "-p", str(settings.get("python", "3.12")), str(venv)])
    install = ["uv", "pip", "install", "-r", str(requirements)]
    index = torch_index(settings)
    if index:
        install += ["--extra-index-url", index, "--index-strategy", "unsafe-best-match"]
    run(install, env=dict(os.environ, VIRTUAL_ENV=str(venv)))
    return python


def run(args, cwd=None, env=None, timeout=None, log=None):
    if log:
        with open(log, "a", encoding="utf-8") as fh:
            result = subprocess.run(args, cwd=cwd, env=env, stdout=fh, stderr=subprocess.STDOUT, timeout=timeout)
        if result.returncode != 0:
            raise TrainingError("%s exited %s; see %s" % (Path(args[1] if len(args) > 1 else args[0]).name, result.returncode, log))
        return ""
    result = subprocess.run(args, cwd=cwd, env=env, text=True, capture_output=True, timeout=timeout)
    if result.returncode != 0:
        raise TrainingError("%s failed: %s" % (" ".join(map(str, args[:2])), (result.stderr or result.stdout).strip()[-500:]))
    return result.stdout


def backbone(root):
    """The pinned checkpoint under the run's HF home; returns its snapshot directory."""
    home = Path(root) / "hf"
    snapshot = home / "hub" / ("models--" + BACKBONE[0].replace("/", "--")) / "snapshots" / BACKBONE[1]
    if not snapshot.exists():
        run(["uvx", "--from", "huggingface_hub", "hf", "download", BACKBONE[0], "--revision", BACKBONE[1]], env=dict(os.environ, HF_HOME=str(home)))
        refs = snapshot.parents[1] / "refs"
        refs.mkdir(parents=True, exist_ok=True)
        (refs / "main").write_text(BACKBONE[1], encoding="utf-8")
    return snapshot


def environment(root, settings):
    env = dict(os.environ, HF_HOME=str(Path(root) / "hf"), HF_HUB_OFFLINE="1", PYTHONUNBUFFERED="1",
               LYCAON_DECIDE_MODEL_ID=BACKBONE[0], OMP_NUM_THREADS=str(settings.get("threads", 8)), MKL_NUM_THREADS=str(settings.get("threads", 8)))
    device = settings.get("device", "auto")
    if device and device != "auto":
        env["LYCAON_DECIDE_DEVICE"] = device
    elif accelerator() == "rocm":
        env["LYCAON_DECIDE_DEVICE"] = "cuda"
    return env


def mixed_rows(out, fraction, seed, *rows):
    """The row files concatenated into `out`: every session row, and the generated
    skill-request rows of a seeded `fraction` of the training families."""
    with open(out, "w", encoding="utf-8") as dst:
        for path in rows:
            with open(path, encoding="utf-8") as src:
                for line in src:
                    if not line.strip():
                        continue
                    meta = json.loads(line).get("meta") or {}
                    if meta.get("source") == "skillreq":
                        draw = int(hashlib.sha256(("%s%s" % (seed, meta["family"])).encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
                        if meta.get("split") != "train" or draw >= fraction:
                            continue
                    dst.write(line if line.endswith("\n") else line + "\n")
    return out


def skillreq_partition(judged, partition, out):
    """The judged skill requests whose families the run's ancestry put in `partition`."""
    out = Path(out)
    with open(judged, encoding="utf-8") as src, open(out, "w", encoding="utf-8") as dst:
        for line in src:
            if line.strip() and (json.loads(line).get("meta") or {}).get("split") == partition:
                dst.write(line if line.endswith("\n") else line + "\n")
    return out


def has_rows(path):
    """Whether a rows file exists and holds at least one row."""
    if not path or not Path(path).exists():
        return False
    with open(path, encoding="utf-8") as fh:
        return any(line.strip() for line in fh)


def with_epochs(args, epochs):
    """The trainer arguments with every epoch count replaced, for a short run."""
    if not epochs:
        return args
    out = [str(a) for a in args]
    if "--epochs" in out:
        out[out.index("--epochs") + 1] = str(epochs)
    else:
        out += ["--epochs", str(epochs)]
    return out


def train(recipe, inputs, checkout, root, settings, heads_dir, label, budget_seconds):
    """One recipe into heads_dir/<head>.safetensors. inputs: corpus, train, val,
    skillreq_train, skillreq_eval, and dumps_train and dumps_val (the site dumps of the
    train and validation repositories)."""
    spec = RECIPES[recipe]
    heads_dir = Path(heads_dir)
    heads_dir.mkdir(parents=True, exist_ok=True)
    out = heads_dir / (spec["head"] + ".safetensors")
    if out.exists():
        return {"head": str(out), "reused": True}
    if spec["kind"] == "turn":
        script = "train.py"
        args = ["--corpus", inputs["corpus"], "--train", inputs["train"], "--val", inputs["val"], "--families", spec["families"],
                *TURN_ARGS, "--label", label, "--out", out]
    elif spec["kind"] == "rank":
        if not inputs.get("skillreq_train") or not inputs.get("skillreq_eval"):
            return {"skipped": "E4 needs judged skill requests in the train and validation partitions"}
        script = "train.py"
        mixed = mixed_rows(heads_dir / "e4-train.jsonl", 0.5, 7, inputs["train"], inputs["skillreq_train"])
        args = ["--corpus", inputs["corpus"], "--train", mixed, "--val", inputs["skillreq_eval"], *RANK_ARGS, "--label", label, "--out", out]
    else:
        if not inputs.get("dumps_train") or not inputs.get("dumps_val"):
            return {"skipped": "code-rank needs site dumps of train and validation repositories"}
        script = "rerank/train_rerank.py"
        args = ["--out", out, "--label", label, *RERANK_ARGS, "--metrics", out.with_suffix(".metrics.json")]
        for dump in inputs["dumps_train"]:
            args += ["--dump", dump]
        for dump in inputs["dumps_val"]:
            args += ["--val-dump", dump]
    python = venv_python(root, settings)
    env = environment(root, settings)
    backbone(root)
    log = heads_dir / (recipe + ".log")
    started = time.time()
    try:
        run(host_contract.script(python, script, with_epochs(args, settings.get("epochs"))), cwd=str(checkout), env=env,
            timeout=budget_seconds, log=log)
    except subprocess.TimeoutExpired as exc:
        raise TrainingError("%s exceeded the training budget; see %s" % (recipe, log)) from exc
    if not out.exists():
        raise TrainingError("%s wrote no head; see %s" % (recipe, log))
    return {"head": str(out), "log": str(log), "seconds": round(time.time() - started), "device": env.get("LYCAON_DECIDE_DEVICE", accelerator())}
