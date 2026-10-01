"""Finish a pass the way a person would: the release anchors committed on a release
branch, pushed with a pull request, and the releases uploaded to the Hub. Every step
is skipped with its reason where the checkout cannot do it."""
import shutil
import subprocess
from pathlib import Path

from . import anchor, config, hub

ROOT = Path(__file__).resolve().parents[2]


def git(*args, check=True):
    return subprocess.run(["git", "-C", str(ROOT), *args], text=True, capture_output=True, check=check)


def in_checkout():
    return git("rev-parse", "--is-inside-work-tree", check=False).returncode == 0


def anchors_for(version, kinds):
    return [anchor.path(kind, version) for kind in kinds if anchor.path(kind, version).exists()]


def commit_anchors(version, kinds):
    """Commit this version's anchors on release/<version>; returns the branch and commit."""
    if not in_checkout():
        return {"skipped": "not a git checkout; commit releases/ by hand"}
    paths = anchors_for(version, kinds)
    if not paths:
        return {"skipped": "no anchors to commit"}
    branch = "release/" + version
    current = git("rev-parse", "--abbrev-ref", "HEAD").stdout.strip()
    if current != branch:
        git("checkout", "-B", branch)
    git("add", "--", *[str(p.relative_to(ROOT)) for p in paths])
    if git("diff", "--cached", "--quiet", check=False).returncode == 0:
        return {"branch": branch, "commit": git("rev-parse", "HEAD").stdout.strip(), "unchanged": True}
    git("commit", "-s", "-m", "release: record %s anchors" % version)
    return {"branch": branch, "commit": git("rev-parse", "HEAD").stdout.strip()}


def push_and_pull_request(version, body):
    branch = "release/" + version
    if git("remote", "get-url", "origin", check=False).returncode != 0:
        return {"skipped": "no origin remote"}
    pushed = git("push", "-u", "origin", branch, check=False)
    if pushed.returncode != 0:
        return {"skipped": "push failed: " + pushed.stderr.strip()[-200:]}
    if not shutil.which("gh"):
        return {"branch": branch, "pull_request": None, "note": "gh is not installed; open the pull request by hand"}
    result = subprocess.run(["gh", "pr", "create", "--base", "main", "--head", branch, "--title", "Release %s" % version, "--body", body],
                            cwd=str(ROOT), text=True, capture_output=True)
    if result.returncode != 0:
        existing = subprocess.run(["gh", "pr", "view", branch, "--json", "url", "--jq", ".url"], cwd=str(ROOT), text=True, capture_output=True)
        return {"branch": branch, "pull_request": existing.stdout.strip() or None, "note": result.stderr.strip()[-200:]}
    return {"branch": branch, "pull_request": result.stdout.strip()}


def publish(kind, release_dir, version, push):
    return hub.publish(kind, release_dir, version, None, push)


def finish(version, releases, push):
    """Anchors committed and pushed, releases uploaded; a dry run only lists the plan."""
    out = {"anchors": commit_anchors(version, list(releases)) if push else {"skipped": "not pushing"}}
    if push and "commit" in out["anchors"]:
        body = "Release %s.\n\n%s\n\nDataset repo: %s\nModel repo: %s\n" % (
            version, "\n".join("- %s: %s" % (k, v) for k, v in sorted(releases.items())), config.hub()["dataset_repo"], config.hub()["model_repo"])
        out["pull_request"] = push_and_pull_request(version, body)
    for kind, release_dir in sorted(releases.items()):
        out[kind] = publish(kind, release_dir, version, push)
    return out
