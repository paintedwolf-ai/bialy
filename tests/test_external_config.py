import shutil

import pytest
import yaml

from bialy import config


def loaded(tmp_path, monkeypatch, local=None):
    """The repository configuration copied, with this deployment's replicas.local.yaml replaced."""
    root = tmp_path / ("config-%d" % len(list(tmp_path.iterdir())))
    shutil.copytree(config.CONFIG, root, ignore=shutil.ignore_patterns("replicas.local.yaml"))
    monkeypatch.setattr(config, "CONFIG", root)
    if local is not None:
        (root / "replicas.local.yaml").write_text(yaml.safe_dump(local))
    return config.load()


def test_external_pins_load_without_a_tunnel_and_cannot_tunnel(tmp_path, monkeypatch):
    model = loaded(tmp_path, monkeypatch).model("qwen3.6-35b-a3b-q4")
    assert model.wire_model == "qwen3.6-35b-a3b-q4"
    assert model.ports() == [8011]
    with pytest.raises(config.ConfigError, match="replicas.local.yaml"):
        model.tunnels()


def test_the_local_file_supplies_the_tunnel(tmp_path, monkeypatch):
    local = {"external": {"qwen3.6-35b-a3b-q4": {"ssh": "root@gpu-1.example", "key": "/root/.ssh/bialy_tunnel", "remote": "127.0.0.1:8011"}}}
    tunnel = loaded(tmp_path, monkeypatch, local).model("qwen3.6-35b-a3b-q4").tunnels()[0]
    assert tunnel["port"] == 8011 and tunnel["ssh"] == "root@gpu-1.example"
    assert tunnel["artifact"] == "Qwen3.6-35B-A3B-Q4_K_M.gguf"


def test_tunnels_for_factory_served_or_unknown_models_are_refused(tmp_path, monkeypatch):
    tunnel = {"ssh": "root@gpu-1.example", "key": "/k", "remote": "127.0.0.1:1"}
    with pytest.raises(config.ConfigError, match="serves itself"):
        loaded(tmp_path, monkeypatch, {"external": {"qwen3.6-35b-a3b": tunnel}})
    with pytest.raises(config.ConfigError, match="unknown"):
        loaded(tmp_path, monkeypatch, {"external": {"nobody": tunnel}})
    with pytest.raises(config.ConfigError, match="ssh, remote and key"):
        loaded(tmp_path, monkeypatch, {"external": {"qwen3.6-35b-a3b-q4": {"ssh": "root@gpu-1.example"}}})
