import dataclasses
import hashlib
import subprocess

from bialy import repos


def test_a_license_text_that_differs_from_the_reviewed_one_is_a_problem(factory, tmp_path, monkeypatch):
    monkeypatch.undo()  # the real license_text, reading the clone at its pin
    clone = tmp_path / "repos" / "demo"
    clone.mkdir(parents=True)
    (clone / "LICENSE").write_text("MIT License\n")
    for cmd in (["init", "-q"], ["add", "LICENSE"], ["-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "x"]):
        subprocess.run(["git", *cmd], cwd=clone, check=True)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=clone, check=True, capture_output=True, text=True).stdout.strip()
    reviewed = hashlib.sha256(b"MIT License\n").hexdigest()
    repo = dataclasses.replace(factory.repos[0], name="demo", commit=commit, license_files={"LICENSE": reviewed})
    site = dataclasses.replace(factory, root=tmp_path, repos=[repo])
    assert repos.verify(site) == []
    site = dataclasses.replace(site, repos=[dataclasses.replace(repo, license_files={"LICENSE": "0" * 64, "COPYING": reviewed})])
    assert repos.verify(site) == ["demo: LICENSE differs from the reviewed text", "demo: no COPYING at the pinned commit"]
