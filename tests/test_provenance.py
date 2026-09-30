from bialy import provenance


def test_digest_covers_source_only(monkeypatch, tmp_path):
    (tmp_path / "src" / "pkg").mkdir(parents=True)
    (tmp_path / "src" / "pkg" / "a.py").write_text("x = 1\n")
    monkeypatch.setattr(provenance, "ROOT", tmp_path)
    before = provenance.tree_sha256(["src"])
    (tmp_path / "src" / "pkg.egg-info").mkdir()
    (tmp_path / "src" / "pkg.egg-info" / "PKG-INFO").write_text("built")
    (tmp_path / "src" / "pkg" / "__pycache__").mkdir()
    (tmp_path / "src" / "pkg" / "__pycache__" / "a.pyc").write_bytes(b"\0")
    (tmp_path / "src" / ".DS_Store").write_bytes(b"\0")
    assert provenance.tree_sha256(["src"]) == before
    (tmp_path / "src" / "pkg" / "a.py").write_text("x = 2\n")
    assert provenance.tree_sha256(["src"]) != before


def test_a_deployed_copy_names_its_commit(monkeypatch, tmp_path):
    for base in ("src", "config", "runner"):
        (tmp_path / base).mkdir()
    (tmp_path / "SOURCE_COMMIT").write_text("abc123\n")
    monkeypatch.setattr(provenance, "ROOT", tmp_path)
    source = provenance.factory_source()
    assert source["git_commit"] == "abc123" and source["dirty"] is None
