"""Train the heads on this machine: the trainer from the pinned checkout in a venv the
run makes, over the accelerator it finds, under a time budget.

Recipes are the released ones, named as scripts/bialy/README.md names them: B5 and
B7G for the turn heads, E4 for unit-rank over the judged rows and the skill
requests, and code-rank over the site dumps.
"""
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

BACKBONE = ("convaiinnovations/laya-multilingual", "e4e9ddf21a7b1903b7acffd8814ad4307bf63a67")
TRAIN_HOST = Path(__file__).resolve().parents[2] / "train-host"
TORCH_INDEX = {"rocm": "https://download.pytorch.org/whl/rocm7.14", "cuda": None, "mps": None, "cpu": "https://download.pytorch.org/whl/cpu"}
TURN_ARGS = ["--tool-truth", "consensus", "--tool-weight", "none", "--tool-negatives", "24", "--lr", "5e-4", "--pos-weight", "6",
             "--seed", "11", "--batch-size", "64", "--epochs", "45", "--patience", "8"]
RANK_ARGS = ["--families", "skills,requests", "--skill-scored", "8", "--skill-zeros", "12", "--seed", "11", "--batch-size", "32",
             "--rank-levels", "skills-blended"]
RECIPES = {
    "B5": {"head": "turn-load", "kind": "turn", "families": "tools"},
    "B7G": {"head": "guide-load", "kind": "turn", "families": "guides"},
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
    requirements = Path(settings.get("requirements") or TRAIN_HOST / "requirements-cuda.txt")
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


def independent_corpus(checkout, corpus, out):
    out = Path(out)
    if not out.exists():
        run([sys.executable, str(TRAIN_HOST / "independent_corpus.py"), str(corpus), str(out)], cwd=str(checkout))
    return out


def mixed_rows(out, fraction, seed, *rows):
    run([sys.executable, str(TRAIN_HOST / "mix.py"), str(out), str(fraction), str(seed)] + [str(r) for r in rows])
    return out


def skillreq_eval(judged, out):
    """The judged skill requests whose families the writer set aside for evaluation."""
    out = Path(out)
    with open(judged, encoding="utf-8") as src, open(out, "w", encoding="utf-8") as dst:
        for line in src:
            if line.strip() and (json.loads(line).get("meta") or {}).get("split") == "eval":
                dst.write(line if line.endswith("\n") else line + "\n")
    return out


def with_epochs(args, epochs):
    """The trainer arguments with every epoch count replaced, for a short run."""
    if not epochs:
        return args
    out = list(args)
    if "--epochs" in out:
        out[out.index("--epochs") + 1] = str(epochs)
    else:
        out += ["--epochs", str(epochs)]
    return out


def train(recipe, inputs, checkout, root, settings, heads_dir, label, budget_seconds):
    """One recipe into heads_dir/<head>.safetensors. inputs: corpus, independent_corpus,
    train, val, skillreq_train, skillreq_eval, dumps (a directory)."""
    spec = RECIPES[recipe]
    heads_dir = Path(heads_dir)
    heads_dir.mkdir(parents=True, exist_ok=True)
    out = heads_dir / (spec["head"] + ".safetensors")
    if out.exists():
        return {"head": str(out), "reused": True}
    python = venv_python(root, settings)
    env = environment(root, settings)
    backbone(root)
    log = heads_dir / (recipe + ".log")
    started = time.time()
    if spec["kind"] == "turn":
        args = [str(python), "scripts/bialy/train.py", "--corpus", str(inputs["independent_corpus"]), "--train", str(inputs["train"]),
                "--val", str(inputs["val"]), "--families", spec["families"], *TURN_ARGS, "--label", label, "--out", str(out)]
    elif spec["kind"] == "rank":
        if not inputs.get("skillreq_train"):
            return {"skipped": "E4 needs judged skill requests; the skillreq stages did not run"}
        mixed = mixed_rows(heads_dir / "e4-train.jsonl", 0.5, 7, inputs["train"], inputs["skillreq_train"])
        args = [str(python), "scripts/bialy/train.py", "--corpus", str(inputs["corpus"]), "--train", str(mixed),
                "--val", str(inputs.get("skillreq_eval") or inputs["val"]), *RANK_ARGS, "--label", label, "--out", str(out)]
    else:
        dumps = sorted(Path(inputs["dumps"]).glob("*.jsonl")) if inputs.get("dumps") else []
        if not dumps:
            return {"skipped": "code-rank needs site dumps; the coderank stage did not run"}
        args = [str(python), "scripts/bialy/rerank/train_rerank.py", "--out", str(out), "--label", label, "--seed", "7"]
        for dump in dumps:
            args += ["--dump", str(dump)]
    args = with_epochs(args, settings.get("epochs"))
    try:
        run(args, cwd=str(checkout), env=env, timeout=budget_seconds, log=log)
    except subprocess.TimeoutExpired as exc:
        raise TrainingError("%s exceeded the training budget; see %s" % (recipe, log)) from exc
    if not out.exists():
        raise TrainingError("%s wrote no head; see %s" % (recipe, log))
    return {"head": str(out), "log": str(log), "seconds": round(time.time() - started), "device": env.get("LYCAON_DECIDE_DEVICE", accelerator())}
