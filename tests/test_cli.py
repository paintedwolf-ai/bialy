from bialy import cli


def test_check_validates_the_configuration(capsys):
    assert cli.main(["check"]) == 0
    assert "16 repositories (3 held out)" in capsys.readouterr().out


def test_audit_from_the_command_line(factory, split_dir, corpus_file, tmp_path, capsys):
    from test_release import build_release

    out = build_release(factory, split_dir, corpus_file, tmp_path)
    assert cli.main(["verify", "dataset", "--release", str(out)]) == 0
    assert cli.main(["audit", "dataset", "--release", str(out)]) == 0
    assert '"leakage": "none"' in capsys.readouterr().out


def test_publish_lists_without_pushing(factory, split_dir, corpus_file, tmp_path, capsys):
    from test_release import build_release

    out = build_release(factory, split_dir, corpus_file, tmp_path)
    assert cli.main(["publish", "dataset", "--release", str(out), "--version", "v0"]) == 0
    assert '"pushed": false' in capsys.readouterr().out
