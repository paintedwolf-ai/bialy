"""Seed repositories: clone at the pinned commit and verify pin and license."""

import hashlib
import subprocess


def git(*args, cwd=None):
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


def license_text(factory, repo, name):
    """A license file as it is at the repository's pinned commit."""
    return subprocess.run(["git", "show", "%s:%s" % (repo.commit, name)], cwd=factory.root / "repos" / repo.name,
                          check=True, capture_output=True).stdout


def fetch(factory):
    """Clone every repository with enough history for history questions, at its pin."""
    base = factory.root / "repos"
    base.mkdir(parents=True, exist_ok=True)
    for repo in factory.repos:
        path = base / repo.name
        if not path.exists():
            git("clone", "-q", "--depth", "400", repo.url, str(path))
        if subprocess.run(["git", "cat-file", "-e", repo.commit], cwd=path).returncode != 0:
            git("fetch", "-q", "--depth", "400", "origin", repo.commit, cwd=path)
        git("checkout", "-q", repo.commit, cwd=path)
    return verify(factory)


def verify(factory):
    """Each checkout sits at its pinned commit, and its license texts are the ones a person
    reviewed against the SPDX label."""
    problems = []
    for repo in factory.repos:
        path = factory.root / "repos" / repo.name
        if not path.exists():
            problems.append("%s: not cloned" % repo.name)
            continue
        head = git("rev-parse", "HEAD", cwd=path)
        if head != repo.commit:
            problems.append("%s: at %s, pinned %s" % (repo.name, head, repo.commit))
        for name, digest in repo.license_files.items():
            try:
                actual = hashlib.sha256(license_text(factory, repo, name)).hexdigest()
            except subprocess.CalledProcessError:
                problems.append("%s: no %s at the pinned commit" % (repo.name, name))
                continue
            if actual != digest:
                problems.append("%s: %s differs from the reviewed text" % (repo.name, name))
    return problems
