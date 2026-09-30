"""What produced an artifact, recorded beside it so a published dataset can be rebuilt.

Sessions are sampled, so a rebuild reproduces the process, not the bytes: the
same pinned models, repositories, tasks, engine build, and runner image. Each
stage writes provenance.json next to its output; a release carries them all.
"""

import hashlib
import json
import platform
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _source(path):
    """A file the repository tracks: not build output, caches, or editor litter."""
    return path.is_file() and path.name != ".DS_Store" and not any(
        part == "__pycache__" or part.endswith(".egg-info") for part in path.parts)


def tree_sha256(paths):
    """One digest over every source file under `paths`, by relative path and content, so a
    checkout and a copy of it (which has no .git) agree."""
    h = hashlib.sha256()
    for base in paths:
        for path in sorted(p for p in (ROOT / base).rglob("*") if _source(p)):
            h.update(str(path.relative_to(ROOT)).encode() + b"\0")
            h.update(path.read_bytes())
    return h.hexdigest()


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _run(*args):
    try:
        return subprocess.run(list(args), capture_output=True, text=True, timeout=60).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        return ""


def factory_source():
    """The factory's commit, from git in a checkout or from SOURCE_COMMIT in a copy
    deployed without .git, with a digest that checks the copy matches it."""
    head = _run("git", "-C", str(ROOT), "rev-parse", "HEAD")
    dirty = bool(_run("git", "-C", str(ROOT), "status", "--porcelain")) if head else None
    if not head and (ROOT / "SOURCE_COMMIT").exists():
        head = (ROOT / "SOURCE_COMMIT").read_text(encoding="utf-8").strip()
    return {"git_commit": head or None, "dirty": dirty, "tree_sha256": tree_sha256(["src", "config", "runner"])}


def engine(factory, bin_dir=None):
    """The lycaon build the runners carry: BUILD.json written when it was built, plus digests."""
    bin_dir = Path(bin_dir) if bin_dir else factory.root / "bin"
    build = bin_dir / "BUILD.json"
    out = json.loads(build.read_text()) if build.exists() else {}
    out["binaries"] = {name: file_sha256(bin_dir / name) for name in ("lycaon", "lycaon-debug") if (bin_dir / name).exists()}
    return out


def serving(factory):
    version = _run(str(factory.root / "venvs/vllm/bin/python"), "-c", "import vllm; print(vllm.__version__)")
    gpus = _run("nvidia-smi", "--query-gpu=name", "--format=csv,noheader").splitlines()
    return {"vllm": version, "gpus": gpus,
            "models": [{"id": m.id, "hf": m.hf, "revision": m.revision, "license": m.license, "serve": m.serve,
                        "replicas": len(m.replicas), "hosted": {k: v for k, v in (m.hosted or {}).items() if k != "key_env"} or None}
                       for m in factory.models]}


def image(tag):
    return _run("docker", "image", "inspect", "-f", "{{.Id}}", tag) or None


def record(factory, out_path, stage, uses_engine=True, bin_dir=None, **extra):
    """Write provenance.json for one stage's output."""
    doc = {"stage": stage, "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "platform": "%s-%s" % (platform.system(), platform.machine()),
           "factory": factory_source(), "engine": engine(factory, bin_dir) if uses_engine else None, "serving": serving(factory),
           "config": {name: (ROOT / "config" / name).read_text(encoding="utf-8") for name in
                      ("models.yaml", "repos.yaml", "archetypes.yaml", "factory.yaml", "unattended.yaml")}}
    doc.update(extra)
    Path(out_path).write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    return doc
