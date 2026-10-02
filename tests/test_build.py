import json
import subprocess

from bialy import build


def test_launcher_uses_the_hosts_serve_arguments_without_a_head_budget(tmp_path):
    out = build.launcher("/e/bialy", "/m/snap", {"turn-load": "/h/turn-load.safetensors", "guide-load": "/h/guide-load.safetensors"}, tmp_path / "l.sh")
    text = out.read_text()
    assert text.startswith("#!/bin/sh\nexec /e/bialy serve --model /m/snap --model-id convaiinnovations/laya-multilingual")
    assert "--head guide-load=/h/guide-load.safetensors --head turn-load=/h/turn-load.safetensors" in text
    assert "--head-max-len" not in text and "--device" not in text
    assert "--device cpu" in build.launcher("/e/bialy", "/m", {}, tmp_path / "c.sh", device="cpu").read_text()


def test_checkout_commit_falls_back_to_the_source_marker(tmp_path):
    assert build.checkout_commit(tmp_path) is None
    (tmp_path / "SOURCE_COMMIT").write_text("abc123\n")
    assert build.checkout_commit(tmp_path) == "abc123"
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q", "--allow-empty", "-m", "x"], check=True)
    assert len(build.checkout_commit(tmp_path)) == 40


def test_engine_features_follow_the_platform(monkeypatch):
    monkeypatch.delenv("BIALY_FEATURES", raising=False)
    monkeypatch.setattr(build.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(build.platform, "machine", lambda: "arm64")
    assert build.engine_features() == "metal,mlx"
    monkeypatch.setattr(build.platform, "system", lambda: "Linux")
    assert build.engine_features() == ""
    monkeypatch.setenv("BIALY_FEATURES", "cuda")
    assert build.engine_features() == "cuda"


def test_gitengine_refuses_a_digest_that_is_not_the_pin(tmp_path, monkeypatch):
    checkout = tmp_path / "co"
    (checkout / "lycaon/config/gitengine").mkdir(parents=True)
    (checkout / "lycaon/config/gitengine/pin.yaml").write_text(
        'git_version: "2.53.0"\nplatforms:\n  linux-amd64:\n    url: "https://example.invalid/{git_version}-{build}.tar.gz"\n    build: "f49d009"\n    sha256: "%s"\n' % ("0" * 64))
    archive = tmp_path / "a.tar.gz"
    import tarfile
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(__file__, arcname="bin/git")
    monkeypatch.setattr(build.urllib.request, "urlretrieve", lambda url, dest: __import__("shutil").copy(archive, dest))
    try:
        build.gitengine(checkout, tmp_path / "engine/gitengine")
    except build.BuildError as exc:
        assert "not the pinned" in str(exc)
    else:
        raise AssertionError("a wrong digest was accepted")
    assert not (tmp_path / "engine/gitengine/bin/git").exists()


def test_model_dir_resolves_the_snapshot_and_marks_it_complete(tmp_path):
    snap = tmp_path / "snap"
    for f in build.MODEL_FILES:
        (snap / f).parent.mkdir(parents=True, exist_ok=True)
        (snap / f).write_text("x")
    out = build.model_dir(snap, tmp_path / "model")
    assert (out / ".complete").exists() and (out / "encoder/config.json").read_text() == "x"
    (snap / "model.safetensors").unlink()
    try:
        build.model_dir(snap, tmp_path / "model2")
    except build.BuildError as exc:
        assert "model.safetensors" in str(exc)
    else:
        raise AssertionError("an incomplete snapshot was accepted")


def test_pilot_carries_the_engine_checkpoint_and_heads(tmp_path):
    (tmp_path / "e").write_bytes(b"engine")
    (tmp_path / "m").mkdir()
    (tmp_path / "m/config.json").write_text("{}")
    (tmp_path / "m/.complete").write_text("")
    (tmp_path / "h").mkdir()
    (tmp_path / "h/turn-load.safetensors").write_bytes(b"h")
    (tmp_path / "h/guide-load.safetensors").write_bytes(b"h")
    (tmp_path / "h/B5.log").write_text("log")
    report = build.pilot(tmp_path / "e", tmp_path / "m", tmp_path / "h", tmp_path / "pilot")
    assert report == {"heads": ["guide-load", "turn-load"]}
    assert (tmp_path / "pilot/bialy").exists() and (tmp_path / "pilot/model/config.json").exists()
    assert sorted(p.name for p in (tmp_path / "pilot/heads").iterdir()) == ["guide-load.safetensors", "turn-load.safetensors"]


def test_binaries_write_a_build_record(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(build, "go_run", lambda checkout, cache, args, cwd, goos=None, goarch=None, mounts=(): calls.append((args[-1], goos, goarch)))
    monkeypatch.setattr(build, "checkout_commit", lambda c: "deadbeef")
    monkeypatch.setattr(build.shutil, "which", lambda name: None)
    record = build.binaries(tmp_path / "co", tmp_path / "bin", tmp_path / "cache")
    assert [c[0] for c in calls] == ["./cmd/lycaon", "./cmd/lycaon-debug", "./cmd/decide-rerank"]
    assert all(c[1:] == ("linux", "amd64") for c in calls)
    assert json.loads((tmp_path / "bin/BUILD.json").read_text())["lycaon_commit"] == "deadbeef" and record["go_image"] == build.GO_IMAGE


def test_rust_channel_comes_from_the_checkouts_toolchain_file(tmp_path):
    (tmp_path / "rust-toolchain.toml").write_text('[toolchain]\nchannel = "1.97.1"\nprofile = "minimal"\n')
    assert build.rust_channel(tmp_path) == "1.97.1" and build.rust_image(tmp_path) == "rust:1.97.1-bullseye"


def test_a_scanner_candidate_is_checked_for_the_runner_platform(tmp_path):
    candidate = tmp_path / "artifact"
    candidate.mkdir()
    (candidate / "opengrep").write_bytes(b"bin")
    (candidate / "source-lock.json").write_text("{}")
    (candidate / "provenance.json").write_text(json.dumps({"platform": "darwin", "architecture": "arm64", "version": "1.30.0+paintedwolf.37"}))
    try:
        build.opengrep_candidate(candidate, tmp_path / "engine")
    except build.BuildError as exc:
        assert "darwin/arm64" in str(exc)
    else:
        raise AssertionError("a macOS candidate was accepted for linux runners")
    (candidate / "provenance.json").write_text(json.dumps({"platform": "linux", "architecture": "x86_64", "version": "1.30.0+paintedwolf.37", "binary_sha256": "ab"}))
    report = build.opengrep_candidate(candidate, tmp_path / "engine")
    assert report["version"] == "1.30.0+paintedwolf.37" and (tmp_path / "engine/opengrep/opengrep").exists()
