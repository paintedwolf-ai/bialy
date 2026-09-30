"""Publish checksum-verified releases; dry-run unless `push` is set.

New repositories are private. Each upload creates one tagged commit.
"""

import subprocess
from pathlib import Path

from . import anchor, config
from .provenance import file_sha256

KINDS = {"dataset": ("dataset_repo", "dataset"), "heads": ("model_repo", "model")}


def seal(release_dir):
    """Write SHA256SUMS over every file in the release, by path relative to it."""
    release_dir = Path(release_dir)
    files = sorted(p for p in release_dir.rglob("*") if p.is_file() and p.name != "SHA256SUMS")
    sums = ["%s  %s" % (file_sha256(p), p.relative_to(release_dir).as_posix()) for p in files]
    (release_dir / "SHA256SUMS").write_text("\n".join(sums) + "\n", encoding="utf-8")


def verify(release_dir):
    """Every file SHA256SUMS lists is present and matches; returns the listed names."""
    sums = Path(release_dir) / "SHA256SUMS"
    if not sums.exists():
        raise ValueError("%s has no SHA256SUMS; build it with bialy release or release-heads" % release_dir)
    names = []
    for line in sums.read_text(encoding="utf-8").splitlines():
        digest, name = line.split("  ", 1)
        if file_sha256(Path(release_dir) / name) != digest:
            raise ValueError("%s does not match SHA256SUMS" % name)
        names.append(name)
    return names


REQUIRED = {"dataset": ("README.md", "LICENSE", "PROVENANCE.json", "train.jsonl", "val.jsonl", "holdout.jsonl",
                        "row.schema.json", "corpus.json", "agreement.jsonl", "tasks.jsonl"),
            "heads": ("README.md", "LICENSE", "NOTICE", "PROVENANCE.json")}


def check(kind, release_dir, anchored=True):
    """A release is complete, matches its checksums and the anchor this repository
    recorded for it, and its card has Hub front matter. `anchored=False` checks a
    local build whose anchor is not committed yet."""
    names = verify(release_dir)
    if anchored:
        anchor.match(kind, release_dir)
    missing = [n for n in REQUIRED[kind] if n not in names]
    if missing:
        raise ValueError("%s release lacks %s" % (kind, ", ".join(missing)))
    if kind == "heads" and not any(n.endswith(".safetensors") for n in names):
        raise ValueError("heads release has no .safetensors head")
    if not (Path(release_dir) / "README.md").read_text(encoding="utf-8").startswith("---\nlicense: apache-2.0"):
        raise ValueError("README.md has no Hub front matter")
    return names


def fetch(kind, version, out_dir, repo=None):
    """Download a tagged release from the Hub into out_dir."""
    from huggingface_hub import snapshot_download

    key, repo_type = KINDS[kind]
    repo_id = repo or config.hub()[key]
    return snapshot_download(repo_id=repo_id, repo_type=repo_type, revision=version, local_dir=str(out_dir))


def plan(kind, release_dir, repo=None):
    key, repo_type = KINDS[kind]
    repo_id = repo or config.hub()[key]
    names = check(kind, release_dir) + ["SHA256SUMS"]
    size = sum((Path(release_dir) / n).stat().st_size for n in names)
    return {"repo_id": repo_id, "repo_type": repo_type, "files": names, "bytes": size}


def committed(path):
    """A committed, unchanged anchor lets consumers verify the published release."""
    root = Path(path).parent.parent
    rel = str(Path(path).relative_to(root))
    tracked = subprocess.run(["git", "-C", str(root), "ls-files", "--error-unmatch", rel], capture_output=True)
    clean = subprocess.run(["git", "-C", str(root), "diff", "--quiet", "HEAD", "--", rel], capture_output=True)
    if tracked.returncode or clean.returncode:
        raise ValueError("commit %s before publishing: it is how consumers verify this release" % rel)


def publish(kind, release_dir, version, repo=None, push=False):
    target = plan(kind, release_dir, repo)
    if not push:
        return dict(target, pushed=False)
    committed(anchor.path(kind, version))
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(target["repo_id"], repo_type=target["repo_type"], private=True, exist_ok=True)
    commit = api.upload_folder(folder_path=str(release_dir), repo_id=target["repo_id"], repo_type=target["repo_type"],
                               commit_message="Release %s" % version, allow_patterns=target["files"])
    api.create_tag(target["repo_id"], tag=version, repo_type=target["repo_type"], revision=commit.oid)
    return dict(target, pushed=True, revision=commit.oid, tag=version)
