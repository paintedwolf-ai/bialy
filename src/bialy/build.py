"""Build what a pass runs on: the runner binaries and engine payload from the pinned
Painted Wolf Code checkout, the decision engine for this host, and the pilot an
engine-on pass carries.

Go builds run with the host's Go when it has one, otherwise inside a pinned Go
image with the checkout mounted at its own path, so the same commands and
relative paths hold either way. The runner platform is linux/amd64.
"""
import hashlib
import json
import os
import platform
import shutil
import subprocess
import tarfile
import tempfile
import time
import urllib.request
from pathlib import Path

import yaml

GO_IMAGE = "golang:1.26"
RUNNER_GOOS, RUNNER_GOARCH, RUNNER_TRIPLE = "linux", "amd64", "x86_64-unknown-linux-gnu"
BINARIES = ("lycaon", "lycaon-debug", "decide-rerank")


class BuildError(RuntimeError):
    pass


def sh(args, cwd=None, env=None, capture=True):
    result = subprocess.run(args, cwd=str(cwd) if cwd else None, env=env, text=True, capture_output=capture)
    if result.returncode != 0:
        raise BuildError("%s failed (%s): %s" % (" ".join(map(str, args[:3])), result.returncode, (result.stderr or result.stdout).strip()[-600:]))
    return result.stdout


def host_is_runner_platform():
    return platform.system() == "Linux" and platform.machine() in ("x86_64", "AMD64")


def checkout_commit(checkout):
    try:
        return sh(["git", "-C", str(checkout), "rev-parse", "HEAD"]).strip()
    except BuildError:
        marker = Path(checkout) / "SOURCE_COMMIT"
        return marker.read_text(encoding="utf-8").strip() if marker.exists() else None


def go_run(checkout, cache_root, args, cwd, goos=None, goarch=None, mounts=()):
    """Run a Go command in the checkout: the host's Go, or the pinned image with the
    checkout, caches, and `mounts` at their own absolute paths."""
    checkout, cache_root = Path(checkout).resolve(), Path(cache_root).resolve()
    env = {"CGO_ENABLED": "0", "GOFLAGS": "-trimpath"}
    if goos:
        env["GOOS"], env["GOARCH"] = goos, goarch
    if shutil.which("go"):
        return sh(args, cwd=cwd, env=dict(os.environ, **env, GOMODCACHE=str(cache_root / "gomod"), GOCACHE=str(cache_root / "gobuild")))
    (cache_root / "gomod").mkdir(parents=True, exist_ok=True)
    (cache_root / "gobuild").mkdir(parents=True, exist_ok=True)
    docker = ["docker", "run", "--rm", "-v", "%s:/go/pkg/mod" % (cache_root / "gomod"), "-v", "%s:/root/.cache/go-build" % (cache_root / "gobuild"),
              "-w", str(cwd)]
    for path in {str(checkout), str(cache_root), *(str(Path(m).resolve()) for m in mounts)}:
        Path(path).mkdir(parents=True, exist_ok=True)
        docker += ["-v", "%s:%s" % (path, path)]
    for key, value in env.items():
        docker += ["-e", "%s=%s" % (key, value)]
    return sh(docker + [GO_IMAGE] + list(args))


def binaries(checkout, out_dir, cache_root, goos=RUNNER_GOOS, goarch=RUNNER_GOARCH):
    """lycaon, lycaon-debug, and decide-rerank for one platform, with BUILD.json."""
    checkout, out_dir = Path(checkout).resolve(), Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in BINARIES:
        go_run(checkout, cache_root, ["go", "build", "-o", str(out_dir / name), "./cmd/" + name], cwd=checkout / "lycaon", goos=goos, goarch=goarch,
               mounts=[out_dir])
    build = {"lycaon_commit": checkout_commit(checkout), "goos": goos, "goarch": goarch,
             "built": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "go_image": None if shutil.which("go") else GO_IMAGE}
    (out_dir / "BUILD.json").write_text(json.dumps(build, indent=2) + "\n", encoding="utf-8")
    return build


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def gitengine(checkout, out_dir, platform_key="linux-amd64"):
    """The pinned git toolchain the sidecar runs, fetched and digest-checked from the pin."""
    pin = yaml.safe_load((Path(checkout) / "lycaon/config/gitengine/pin.yaml").read_text(encoding="utf-8"))
    entry = pin["platforms"][platform_key]
    url = entry["url"].format(git_version=pin["git_version"], build=entry["build"])
    out_dir = Path(out_dir)
    if (out_dir / "bin" / "git").exists():
        return {"platform": platform_key, "reused": True}
    with tempfile.TemporaryDirectory() as tmp:
        archive = Path(tmp) / "git.tar.gz"
        urllib.request.urlretrieve(url, archive)
        digest = sha256(archive)
        if digest != entry["sha256"]:
            raise BuildError("gitengine archive digest %s is not the pinned %s" % (digest, entry["sha256"]))
        shutil.rmtree(out_dir, ignore_errors=True)
        out_dir.mkdir(parents=True)
        with tarfile.open(archive) as tar:
            tar.extractall(out_dir, filter="data")
    if not (out_dir / "bin" / "git").exists():
        raise BuildError("gitengine archive did not contain bin/git")
    return {"platform": platform_key, "git_version": pin["git_version"], "sha256": entry["sha256"]}


def opengrep(checkout, engine_dir, cache_root, triple=RUNNER_TRIPLE):
    """The pinned Opengrep release for the runner platform, staged by the checkout's own
    tool. The checkout pins one artifact per platform it releases; a platform without one
    stops the build and says so."""
    cache = Path(cache_root).resolve() / "opengrep"
    cache.mkdir(parents=True, exist_ok=True)
    try:
        artifact = go_run(checkout, cache_root, ["go", "run", "./cmd/opengrep-artifact", "-mode", "fetch", "-cache-root", str(cache), "-target", triple],
                          cwd=Path(checkout) / "lycaon").strip().splitlines()[-1]
    except BuildError as exc:
        if "no prebuilt opengrep release" in str(exc):
            raise BuildError("the checkout pins no Opengrep release for %s (lycaon/config/runtime/scanners/bundled-manifest.yaml); "
                             "select one there, or set run.scanner: none to run without the scanner" % triple) from exc
        raise
    go_run(checkout, cache_root, ["go", "run", "./cmd/opengrep-artifact", "-mode", "stage", "-artifact-directory", artifact,
                                  "-root", str(Path(engine_dir).resolve()), "-target", triple], cwd=Path(checkout) / "lycaon", mounts=[engine_dir])
    return {"artifact": artifact}


def browser(lycaon_bin, engine_dir):
    """chrome-headless-shell for the runner platform, provisioned by the runner's own
    lycaon binary and laid flat under engine/browser/."""
    engine_dir = Path(engine_dir).resolve()
    target = engine_dir / "browser"
    if (target / "chrome-headless-shell").exists():
        return {"reused": True}
    with tempfile.TemporaryDirectory() as tmp:
        cache = Path(tmp) / "cache"
        args = [str(Path(lycaon_bin).resolve()), "browser", "ensure", "--cache-dir", str(cache)]
        if host_is_runner_platform():
            sh(args)
        else:
            sh(["docker", "run", "--rm", "-v", "%s:%s" % (Path(lycaon_bin).resolve().parent, Path(lycaon_bin).resolve().parent),
                "-v", "%s:%s" % (tmp, tmp), GO_IMAGE] + args)
        installs = [p for p in cache.glob("managed/*/*/chrome-headless-shell")]
        if not installs:
            raise BuildError("browser ensure left no chrome-headless-shell under %s" % cache)
        shutil.rmtree(target, ignore_errors=True)
        shutil.copytree(installs[0].parent, target)
    return {"path": str(target / "chrome-headless-shell")}


CANDIDATE_FILES = ("opengrep", "provenance.json", "source-lock.json")


def opengrep_candidate(candidate, engine_dir):
    """A development Opengrep build (the downstream repository's artifact directory for
    linux/amd64), copied as the payload's scanner; development sidecars accept it."""
    candidate = Path(candidate)
    missing = [f for f in CANDIDATE_FILES if not (candidate / f).is_file()]
    if missing:
        raise BuildError("scanner candidate %s lacks %s" % (candidate, ", ".join(missing)))
    provenance = json.loads((candidate / "provenance.json").read_text(encoding="utf-8"))
    if (provenance.get("platform"), provenance.get("architecture")) != ("linux", "x86_64"):
        raise BuildError("scanner candidate %s was built for %s/%s, not the linux/x86_64 runners"
                         % (candidate, provenance.get("platform"), provenance.get("architecture")))
    target = Path(engine_dir) / "opengrep"
    shutil.rmtree(target, ignore_errors=True)
    shutil.copytree(candidate, target)
    os.chmod(target / "opengrep", 0o755)
    return {"candidate": str(candidate), "version": provenance.get("version"), "binary_sha256": provenance.get("binary_sha256")}


def payload(checkout, engine_dir, bin_dir, cache_root, scanner=True, candidate=None):
    """The engine payload the runner image copies: schemas, git, the browser, and the
    scanner: the checkout's pinned Opengrep release, a development candidate, or none
    when the run declines it."""
    checkout, engine_dir = Path(checkout).resolve(), Path(engine_dir).resolve()
    engine_dir.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(engine_dir / "schemas", ignore_errors=True)
    shutil.copytree(checkout / "schemas", engine_dir / "schemas")
    report = {"schemas": len(list((engine_dir / "schemas").glob("*")))}
    report["gitengine"] = gitengine(checkout, engine_dir / "gitengine")
    if candidate:
        report["opengrep"] = opengrep_candidate(candidate, engine_dir)
    elif scanner:
        report["opengrep"] = opengrep(checkout, engine_dir, cache_root)
    else:
        shutil.rmtree(engine_dir / "opengrep", ignore_errors=True)
        report["opengrep"] = None
    report["browser"] = browser(Path(bin_dir) / "lycaon", engine_dir)
    return report


def engine_features():
    if os.environ.get("BIALY_FEATURES"):
        return os.environ["BIALY_FEATURES"]
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        return "metal,mlx"
    return ""


def rust_channel(checkout):
    """The toolchain the checkout pins, from its rust-toolchain.toml."""
    text = (Path(checkout) / "rust-toolchain.toml").read_text(encoding="utf-8")
    for line in text.splitlines():
        if line.strip().startswith("channel"):
            return line.split("=", 1)[1].strip().strip('"')
    raise BuildError("rust-toolchain.toml names no channel")


# The Debian release the container build links against: its glibc (2.31) is older than
# any host or runner image the engine then runs on.
RUST_IMAGE_SUITE = "bullseye"


def rust_image(checkout):
    return "rust:%s-%s" % (rust_channel(checkout), RUST_IMAGE_SUITE)


def cargo_engine(checkout, out_dir, target_dir, features, in_container):
    """cargo build --release of the engine crate: on the host through rustup, which
    installs the pinned toolchain itself, or in the Rust image of that toolchain with the
    checkout mounted at its own path."""
    checkout = Path(checkout).resolve()
    native = checkout / "lycaon/internal/decide/native"
    out_dir, target_dir = Path(out_dir).resolve(), Path(target_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    (target_dir / "registry").mkdir(parents=True, exist_ok=True)
    args = ["cargo", "build", "--release"] + (["--features", features] if features else [])
    image = rust_image(checkout)
    if in_container:
        sh(["docker", "run", "--rm", "-v", "%s:%s" % (checkout, checkout), "-v", "%s:/target" % target_dir,
            "-v", "%s:/usr/local/cargo/registry" % (target_dir / "registry"), "-e", "CARGO_TARGET_DIR=/target", "-w", str(native), image] + args)
    else:
        sh(args, cwd=native, env=dict(os.environ, CARGO_TARGET_DIR=str(target_dir)))
    shutil.copy2(target_dir / "release" / "bialy", out_dir / "bialy")
    os.chmod(out_dir / "bialy", 0o755)
    return {"binary": str(out_dir / "bialy"), "features": features or "cpu", "image": image if in_container else None}


def host_engine(checkout, out_dir, cache_root):
    """The decision engine for this host: evaluation runs through it, and on the runner
    platform the pilot carries it. A host without rustup builds in the toolchain's image,
    so a distribution cargo older than the pin never decides the build."""
    return cargo_engine(checkout, out_dir, Path(cache_root) / "cargo-host", engine_features(), in_container=not shutil.which("rustup"))


def runner_engine(checkout, out_dir, cache_root):
    """A CPU engine for linux/amd64 runners, built in the pinned Rust image."""
    return cargo_engine(checkout, out_dir, Path(cache_root) / "cargo-runner", "", in_container=True)


def launcher(engine_bin, model_dir, heads, out_path, device=None):
    """A one-line launcher replay_eval.py and the audits start the engine with: the same
    serve arguments the host uses, with the checkpoint's own head budget."""
    line = "exec %s serve --model %s --model-id convaiinnovations/laya-multilingual%s %s \"$@\"\n" % (
        engine_bin, model_dir, " --device " + device if device else "",
        " ".join("--head %s=%s" % (name, path) for name, path in sorted(heads.items())))
    out_path = Path(out_path)
    out_path.write_text("#!/bin/sh\n" + line, encoding="utf-8")
    os.chmod(out_path, 0o755)
    return out_path


# The host treats a checkpoint directory as installed only with this marker, which its
# own provisioning writes after the last file; a copied snapshot gets it the same way.
MODEL_COMPLETE_MARKER = ".complete"
MODEL_FILES = ("rl_agent_config.json", "model.safetensors", "encoder/config.json", "tokenizer/tokenizer.json", "tokenizer/tokenizer_config.json")


def model_dir(snapshot, out_dir):
    """The checkpoint as the host and the rerank evaluator expect it: the snapshot's files
    resolved from their blobs, with the completion marker."""
    out_dir = Path(out_dir)
    if (out_dir / MODEL_COMPLETE_MARKER).exists():
        return out_dir
    shutil.rmtree(out_dir, ignore_errors=True)
    shutil.copytree(snapshot, out_dir, symlinks=False)
    missing = [f for f in MODEL_FILES if not (out_dir / f).is_file()]
    if missing:
        raise BuildError("checkpoint %s lacks %s" % (snapshot, ", ".join(missing)))
    (out_dir / MODEL_COMPLETE_MARKER).write_text("", encoding="utf-8")
    return out_dir


def pilot(engine_bin, model, heads_dir, out_dir):
    """What an engine-on runner mounts at /opt/decide: the linux engine, the installed
    checkpoint, and the heads the first pass trained."""
    out_dir = Path(out_dir)
    shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True)
    shutil.copy2(engine_bin, out_dir / "bialy")
    shutil.copytree(model, out_dir / "model")
    if not (out_dir / "model" / MODEL_COMPLETE_MARKER).exists():
        raise BuildError("checkpoint %s is not an installed model directory (see model_dir)" % model)
    (out_dir / "heads").mkdir()
    names = []
    for path in sorted(Path(heads_dir).glob("*.safetensors")):
        shutil.copy2(path, out_dir / "heads" / path.name)
        names.append(path.stem)
    return {"heads": names}
