"""Finish a pass the way a person would: the release anchors committed on a release
branch and tagged, pushed with a pull request, and the releases uploaded to the Hub.
Every step is skipped with its reason where the checkout cannot do it.

The tag is how a release is traced back to the code that made it: <kind>-<version>
(heads-open1-b5-b7g-e4, dataset-open1-b7g-e4) on the commit that records the release's
anchor, which is also the code that built it. A tag never moves; docs/releases.md lists
every release with its tag and the Painted Wolf Code versions that ship it."""
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
    unchanged = git("diff", "--cached", "--quiet", check=False).returncode == 0
    if not unchanged:
        git("commit", "-s", "-m", "release: record %s anchors" % version)
    commit = git("rev-parse", "HEAD").stdout.strip()
    out = {"branch": branch, "commit": commit, "tags": tag_release(version, [p.name.split("-", 1)[0] for p in paths], commit)}
    if unchanged:
        out["unchanged"] = True
    return out


def tag_name(kind, version):
    return "%s-%s" % (kind, version)


def tag_release(version, kinds, commit):
    """Tag the anchoring commit once per release kind. A tag that already names another
    commit is an error: the release it names was made from that code."""
    tags = []
    for kind in kinds:
        name = tag_name(kind, version)
        existing = git("rev-parse", "--verify", "--quiet", name + "^{commit}", check=False).stdout.strip()
        if existing and existing != commit:
            raise RuntimeError("tag %s already names %s, not %s; a released version is never re-tagged" % (name, existing, commit))
        if not existing:
            git("tag", "-a", name, commit, "-m", "%s release %s" % (kind, version))
        tags.append(name)
    return tags


def push_and_pull_request(version, body, tags=()):
    branch = "release/" + version
    if git("remote", "get-url", "origin", check=False).returncode != 0:
        return {"skipped": "no origin remote"}
    pushed = git("push", "-u", "origin", branch, *tags, check=False)
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
        out["pull_request"] = push_and_pull_request(version, body, out["anchors"].get("tags", ()))
    for kind, release_dir in sorted(releases.items()):
        out[kind] = publish(kind, release_dir, version, push)
    return out
