import json
import subprocess
from pathlib import Path

import pytest

from bialy import build


def corpus(tmp_path, **state):
    path = tmp_path / "corpus.json"
    path.write_text(json.dumps({"state": state}))
    return path


def test_launcher_serves_at_the_head_budget_of_the_training_corpus(tmp_path):
    heads = {"turn-load": "/h/turn-load.safetensors", "unit-rank": "/h/unit-rank.safetensors"}
    text = build.launcher("/e/bialy", "/m/snap", heads, tmp_path / "l.sh", corpus(tmp_path, head_tokens=512)).read_text()
    assert text.startswith("#!/bin/sh\nexec /e/bialy serve --model /m/snap --model-id convaiinnovations/laya-multilingual --head-max-len 512 ")
    assert "--head turn-load=/h/turn-load.safetensors --head unit-rank=/h/unit-rank.safetensors" in text and "--device" not in text
    other = build.launcher("/e/bialy", "/m", {}, tmp_path / "c.sh", corpus(tmp_path, head_tokens=384), device="cpu").read_text()
    assert "--device cpu --head-max-len 384" in other


def test_no_launcher_is_written_without_the_corpus_head_budget(tmp_path):
    for state in ({}, {"head_tokens": 0}, {"head_tokens": "512"}):
        with pytest.raises(build.BuildError, match="head_tokens"):
            build.launcher("/e/bialy", "/m", {}, tmp_path / "l.sh", corpus(tmp_path, **state))
    assert not (tmp_path / "l.sh").exists()


def test_checkout_commit_falls_back_to_the_source_marker(tmp_path):
    assert build.checkout_commit(tmp_path) is None
    (tmp_path / "SOURCE_COMMIT").write_text("abc123\n")
    assert build.checkout_commit(tmp_path) == "abc123"
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q", "--allow-empty", "-m", "x"], check=True)
    assert len(build.checkout_commit(tmp_path)) == 40


def checkout_with_features(tmp_path, script):
    checkout = tmp_path / "co"
    (checkout / "scripts").mkdir(parents=True)
    (checkout / "scripts/decide-features.sh").write_text("#!/usr/bin/env bash\n" + script)
    (checkout / "rust-toolchain.toml").write_text('[toolchain]\nchannel = "1.90.0"\n')
    return checkout


def test_engine_features_are_the_checkouts_own_selection(tmp_path, monkeypatch):
    monkeypatch.delenv("BIALY_FEATURES", raising=False)
    checkout = checkout_with_features(tmp_path, 'printf "%s\\n" "${BIALY_FEATURES:-mlx}"\n')
    assert build.engine_features(checkout) == "mlx"
    monkeypatch.setenv("BIALY_FEATURES", "vulkan")
    assert build.engine_features(checkout) == "vulkan"


def test_the_host_engine_is_the_checkouts_own_build_under_rustup(tmp_path, monkeypatch):
    checkout = checkout_with_features(tmp_path, "echo mlx\n")
    calls = []
    monkeypatch.setattr(build.shutil, "which", lambda name: "/bin/" + name)
    real = build.sh
    monkeypatch.setattr(build, "sh", lambda args, **kw: real(args, **kw) if "decide-features.sh" in args[1] else calls.append((args, kw)))
    report = build.host_engine(checkout, tmp_path / "out", tmp_path / "cache")
    assert calls == [(["bash", str(checkout.resolve() / "scripts/build-decide.sh"), str((tmp_path / "out").resolve() / "bialy")],
                      {"cwd": checkout.resolve()})]
    assert report == {"binary": str((tmp_path / "out").resolve() / "bialy"), "features": "mlx", "image": None}


def test_without_rustup_the_engine_builds_the_same_features_in_the_toolchain_image(tmp_path, monkeypatch):
    checkout = checkout_with_features(tmp_path, "echo vulkan\n")
    calls = []
    monkeypatch.setattr(build.shutil, "which", lambda name: None)
    monkeypatch.setattr(build.platform, "system", lambda: "Linux")
    monkeypatch.setattr(build.platform, "libc_ver", lambda: ("glibc", "2.39"))
    real = build.sh

    def sh(args, **kw):
        if "decide-features.sh" in args[1]:
            return real(args, **kw)
        calls.append(args)
        target = next(a.split(":")[0] for a in args if a.endswith(":/target"))
        (Path(target) / "release").mkdir(parents=True)
        (Path(target) / "release/bialy").write_bytes(b"engine")
    monkeypatch.setattr(build, "sh", sh)
    report = build.host_engine(checkout, tmp_path / "out", tmp_path / "cache")
    cargo = ["rust:1.90.0-bookworm", "cargo", "build", "--locked", "--release", "--bin", "bialy"]
    assert calls[0][-9:] == [*cargo, "--features", "vulkan"] and report["features"] == "vulkan" and report["image"] == "rust:1.90.0-bookworm"
    build.runner_engine(checkout, tmp_path / "runner", tmp_path / "cache")
    assert calls[1][-7:] == cargo
    monkeypatch.setattr(build.platform, "libc_ver", lambda: ("glibc", "2.35"))
    with pytest.raises(build.BuildError, match="glibc 2.35 is older"):
        build.host_engine(checkout, tmp_path / "out", tmp_path / "cache")
    monkeypatch.setattr(build.platform, "system", lambda: "Darwin")
    with pytest.raises(build.BuildError, match="rustup"):
        build.host_engine(checkout, tmp_path / "out", tmp_path / "cache")


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
    (tmp_path / "h/unit-rank.safetensors").write_bytes(b"h")
    heads = {name: tmp_path / "h" / (name + ".safetensors") for name in ("turn-load", "unit-rank")}
    report = build.pilot(tmp_path / "e", tmp_path / "m", heads, tmp_path / "pilot")
    assert report == {"heads": ["turn-load", "unit-rank"]}
    assert (tmp_path / "pilot/bialy").exists() and (tmp_path / "pilot/model/config.json").exists()
    assert sorted(p.name for p in (tmp_path / "pilot/heads").iterdir()) == ["turn-load.safetensors", "unit-rank.safetensors"]


def test_binaries_write_a_build_record(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(build, "go_run", lambda checkout, cache, args, cwd, goos=None, goarch=None, mounts=(): calls.append((args[-1], goos, goarch)))
    monkeypatch.setattr(build, "checkout_commit", lambda c: "deadbeef")
    monkeypatch.setattr(build.shutil, "which", lambda name: None)
    monkeypatch.setattr(build, "document_core", lambda *args: {"sha256": "core", "image": None})
    go_mod(tmp_path / "co", "go 1.26.10")
    record = build.binaries(tmp_path / "co", tmp_path / "bin", tmp_path / "cache")
    assert [c[0] for c in calls] == ["./cmd/lycaon", "./cmd/lycaon-debug", "./cmd/decide-rerank"]
    assert all(c[1:] == ("linux", "amd64") for c in calls)
    assert json.loads((tmp_path / "bin/BUILD.json").read_text())["lycaon_commit"] == "deadbeef" and record["go_image"] == "golang:1.26.10"


def test_runner_binaries_carry_the_version_and_scanner_identity(tmp_path, monkeypatch):
    go_mod(tmp_path / "co", "go 1.26.10")
    (tmp_path / "co/VERSION").write_text("1.0.2\n")
    flags = []
    monkeypatch.setattr(build, "go_run", lambda checkout, cache, args, cwd, goos=None, goarch=None, mounts=(): flags.append(args[2]))
    monkeypatch.setattr(build, "checkout_commit", lambda c: "deadbeef")
    monkeypatch.setattr(build, "document_core", lambda *args: {"sha256": "core", "image": None})
    record = build.binaries(tmp_path / "co", tmp_path / "bin", tmp_path / "cache", scanner_identity="aWRlbnRpdHk=")
    assert len(set(flags)) == 1
    assert flags[0] == "-ldflags=-X %s=1.0.2 -X %s=aWRlbnRpdHk=" % (build.VERSION_SYMBOL, build.SCANNER_IDENTITY_SYMBOL)
    assert record["version"] == "1.0.2" and record["scanner_identity_sha256"]
    monkeypatch.setattr(build, "go_run", lambda checkout, cache, args, cwd, goos=None, goarch=None, mounts=(): flags.append(args[2]))
    flags.clear()
    assert build.binaries(tmp_path / "co", tmp_path / "bin", tmp_path / "cache")["scanner_identity_sha256"] is None
    assert build.SCANNER_IDENTITY_SYMBOL not in flags[0]


def go_mod(checkout, *lines):
    (checkout / "lycaon").mkdir(parents=True, exist_ok=True)
    (checkout / "lycaon/go.mod").write_text("module github.com/lycaon/lycaon\n\n" + "\n".join(lines) + "\n\nrequire (\n\tgo.uber.org/zap v1.27.0\n)\n")


def test_the_go_image_is_the_toolchain_the_checkouts_module_names(tmp_path):
    go_mod(tmp_path, "go 1.26.10")
    assert build.go_image(tmp_path) == "golang:1.26.10"
    go_mod(tmp_path, "go 1.26.0", "toolchain go1.26.11")
    assert build.go_image(tmp_path) == "golang:1.26.11"
    go_mod(tmp_path)
    with pytest.raises(build.BuildError, match="go.mod"):
        build.go_image(tmp_path)


def test_the_go_container_trusts_the_mounted_checkout(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(build.shutil, "which", lambda name: None)
    monkeypatch.setattr(build, "sh", lambda args, **kw: seen.append(args))
    go_mod(tmp_path / "co", "go 1.26.10")
    build.go_run(tmp_path / "co", tmp_path / "cache", ["go", "build", "./cmd/lycaon"], tmp_path / "co")
    args = seen[0]
    pairs = {args[i + 1] for i, a in enumerate(args) if a == "-e"}
    assert {"GIT_CONFIG_COUNT=1", "GIT_CONFIG_KEY_0=safe.directory",
            "GIT_CONFIG_VALUE_0=%s" % (tmp_path / "co").resolve()} <= pairs
    assert args[args.index("golang:1.26.10") + 1:] == ["go", "build", "./cmd/lycaon"]


def test_rust_channel_comes_from_the_checkouts_toolchain_file(tmp_path):
    (tmp_path / "rust-toolchain.toml").write_text('[toolchain]\nchannel = "1.97.1"\nprofile = "minimal"\n')
    assert build.rust_channel(tmp_path) == "1.97.1" and build.rust_image(tmp_path) == "rust:1.97.1-bookworm"


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


@pytest.mark.parametrize('native', [True, False])
def test_document_helper_uses_pinned_crate_and_target_platform(tmp_path, monkeypatch, native):
    monkeypatch.setattr(build.platform, 'system', lambda: 'Linux' if native else 'Darwin')
    monkeypatch.setattr(build.platform, 'machine', lambda: 'x86_64' if native else 'arm64')
    monkeypatch.setattr(build.shutil, 'which', lambda name: '/bin/rustup')
    monkeypatch.setattr(build, 'rust_image', lambda checkout: 'rust:pinned')
    output = tmp_path / 'bin'
    output.mkdir()
    def execute(args, **kwargs):
        assert '--locked' in args and '--manifest-path' in args
        if native:
            assert args[0] == 'cargo'
        else:
            assert args[:5] == ['docker', 'run', '--rm', '--platform', 'linux/amd64']
            assert 'rust:pinned' in args
        binary = tmp_path / 'cache/document-core-linux-amd64/release/pw-document-core'
        binary.parent.mkdir(parents=True)
        binary.write_bytes(b'compiled helper')
    monkeypatch.setattr(build, 'sh', execute)
    result = build.document_core(tmp_path / 'checkout', output, tmp_path / 'cache', 'linux', 'amd64')
    assert result['sha256'] == build.file_sha256(output / 'pw-document-core')
    assert result['image'] == (None if native else 'rust:pinned')
    assert (output / 'pw-document-core').stat().st_mode & 0o111


def test_a_staged_config_serves_the_vocabulary_it_is_given(tmp_path):
    release = tmp_path / build.RELEASE
    release.parent.mkdir(parents=True)
    release.write_text(json.dumps({"release": "shipped", "heads": {"turn-load": {"sha256": "a"}},
                                   "preload": {"catalog_revision": "old", "options": {"read": "Read a file"}}}))
    build.serve_vocabulary(tmp_path, {"catalog_revision": "new", "options": {"edit": "Edit a file"}})
    written = json.loads(release.read_text())
    assert written["preload"] == {"catalog_revision": "new", "options": {"edit": "Edit a file"}}
    assert written["release"] == "shipped" and written["heads"] == {"turn-load": {"sha256": "a"}}
