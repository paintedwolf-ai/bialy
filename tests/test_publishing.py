import json
import subprocess

from bialy import anchor, publishing


def test_anchors_commit_on_a_release_branch_with_sign_off(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", "-b", "main", str(repo)], check=True)
    git = lambda *a: subprocess.run(["git", "-C", str(repo), "-c", "user.email=t@t", "-c", "user.name=T", *a], check=True, capture_output=True, text=True)  # noqa: E731
    (repo / "README.md").write_text("x")
    git("add", "README.md")
    git("commit", "-q", "-m", "init")
    monkeypatch.setattr(publishing, "ROOT", repo)
    monkeypatch.setattr(anchor, "ANCHORS", repo / "releases")
    (repo / "releases").mkdir()
    (repo / "releases/dataset-v9.json").write_text(json.dumps({"schema": anchor.SCHEMA, "kind": "dataset", "version": "v9"}))
    monkeypatch.setenv("GIT_AUTHOR_NAME", "T")
    monkeypatch.setenv("GIT_AUTHOR_EMAIL", "t@t")
    monkeypatch.setenv("GIT_COMMITTER_NAME", "T")
    monkeypatch.setenv("GIT_COMMITTER_EMAIL", "t@t")
    result = publishing.commit_anchors("v9", ["dataset", "heads"])
    assert result["branch"] == "release/v9" and len(result["commit"]) == 40
    log = git("log", "-1", "--format=%B").stdout
    assert "release: record v9 anchors" in log and "Signed-off-by:" in log
    assert publishing.commit_anchors("v9", ["dataset"])["unchanged"]
    assert publishing.push_and_pull_request("v9", "body")["skipped"] == "no origin remote"


def test_outside_a_checkout_publishing_says_so(tmp_path, monkeypatch):
    monkeypatch.setattr(publishing, "ROOT", tmp_path)
    assert "not a git checkout" in publishing.commit_anchors("v1", ["dataset"])["skipped"]
    out = publishing.finish("v1", {}, push=False)
    assert out == {"anchors": {"skipped": "not pushing"}}
