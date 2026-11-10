import dataclasses
import json
from unittest.mock import patch

from bialy import fleet, serve


def test_external_primary_is_tunnelled_and_not_launched_or_stopped(factory, tmp_path):
    model = dataclasses.replace(factory.models[0], serve={"port": 8001, "external": {
        "ssh": "fixture", "remote": "127.0.0.1:9000", "key": "/fixture/key", "model": "quantized-fixture", "artifact_sha256": "a" * 64}}, replicas=())
    f = dataclasses.replace(factory, root=tmp_path, models=[model])
    with patch("subprocess.Popen", side_effect=AssertionError("external server must not be launched")):
        assert serve.start(f, model) is None
        assert serve.stop(f, model) is False
    assert model.tunnels()[0]["port"] == 8001
    assert model.ports() == [8001]
    _, _, _, _, wire_models = fleet.sidecar_files(f, {}, 0)
    assert wire_models[model.id] == "quantized-fixture"


def test_remote_container_replica_on_primary_port_with_ssh_port(factory, tmp_path):
    replica = {"port": 8002, "ssh": "root@ssh5.vast.ai", "ssh_port": 41234, "remote": "127.0.0.1:8000", "key": "/fixture/key"}
    extra = {"port": 8005, "ssh": "root@ssh6.vast.ai", "ssh_port": 41890, "remote": "127.0.0.1:8000", "key": "/fixture/key"}
    model = dataclasses.replace(factory.model("gemma-4-26b-a4b-it"), replicas=(replica, extra))
    f = dataclasses.replace(factory, root=tmp_path, models=[model])
    assert model.ports() == [8002, 8005]
    with patch("subprocess.Popen", side_effect=AssertionError("local vllm must not launch when primary port is a replica tunnel")):
        assert serve.start(f, model) is None
        assert serve.stop(f, model) is False
    spawned = []

    class DummyProc:
        pid = 4242

    with patch("subprocess.Popen", side_effect=lambda args, **kw: spawned.append(args) or DummyProc()), \
         patch("bialy.serve._start_time", return_value="100"), \
         patch("time.sleep"):
        serve.start_tunnels(f, model)
        assert serve.running(f, model) is True
    assert len(spawned) == 2
    assert "-p 41234" in spawned[0][2]
    assert "-p 41890" in spawned[1][2]


def test_readiness_requires_the_configured_model(factory):
    class Response:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return json.dumps({"data": [{"id": "wrong-model"}]}).encode()

    with patch("urllib.request.urlopen", return_value=Response()):
        assert not serve.ready(factory, factory.models[0])
