import copy
import dataclasses
import importlib.util
import json
import threading
from pathlib import Path
from types import SimpleNamespace
from urllib.request import Request, urlopen

import pytest
import yaml

from bialy import custom_model_run, faults, fleet, preflight
from bialy import run as runmod


def collection(factory, tmp_path):
    settings = copy.deepcopy(factory.run)
    settings["custom_model"] = dict(settings["custom_model"], enabled=True, epoch="fixture", host_commit="a" * 40)
    settings["generators"] = ["deepseek-v4.1-flash"]
    factory = dataclasses.replace(factory, root=tmp_path, run=settings)
    run = runmod.Run(factory, "t", settings)
    run.factory = runmod.scaled_factory(run)
    return run


def test_custom_collection_cannot_bypass_or_reuse_stale_preflight(factory, tmp_path, monkeypatch):
    run = collection(factory, tmp_path)
    monkeypatch.setattr(fleet, "sh", lambda *args, **kw: SimpleNamespace(stdout="sha256:image"))
    with pytest.raises(faults.RunFault, match="missing_or_stale"):
        preflight.require(run, "on")
    report = {"image_id": "sha256:image", "settings_sha256": custom_model_run.digest(run.settings)}
    run.save_state({"name": "t", "stages": {"preflight_on": {"status": "done", "outputs": report}}})
    preflight.require(run, "on")
    run.settings["task_timeout"] = "30m"
    with pytest.raises(faults.RunFault, match="missing_or_stale"):
        preflight.require(run, "on")


def test_overlay_window_is_read_from_the_actual_image(factory, tmp_path, monkeypatch):
    run = collection(factory, tmp_path)
    seen = []
    def read(*args):
        seen.append(args)
        return SimpleNamespace(stdout=yaml.safe_dump({"window_source": "explicit", "model_context_window": 49152, "live_budget_pct": 75}))
    monkeypatch.setattr(fleet, "sh", read)
    preflight.verify_window(run, "on")
    assert seen[0][:4] == ("docker", "run", "--rm", "--entrypoint")
    run.settings["custom_model"]["students"]["e4b"] = dict(run.settings["custom_model"]["students"]["e4b"], context_window=32768)
    with pytest.raises(faults.RunFault, match="window_mismatch"):
        preflight.verify_window(run, "on")


def test_preflight_blocks_paid_driving_and_is_required_for_custom_collection(factory, tmp_path, monkeypatch):
    run = collection(factory, tmp_path)
    assert runmod.STAGES.index("preflight") < runmod.STAGES.index("drive")
    assert runmod.STAGES.index("preflight_on") < runmod.STAGES.index("drive_on")
    run.settings["preflight"] = dict(run.settings["preflight"], enabled=False)
    with pytest.raises(faults.RunFault, match="required_for_collection"):
        preflight.qualify(run, "on")
    called = []
    monkeypatch.setattr(fleet, "sh", lambda *args, **kw: SimpleNamespace(stdout="image"))
    monkeypatch.setattr(fleet, "ensure_network", lambda *args: called.append("network"))
    with pytest.raises(faults.RunFault, match="missing_or_stale"):
        runmod.drive_pass(run, "on", None)
    assert called == []


@pytest.mark.parametrize("phase", ["mock", "fixture"])
def test_preflight_uses_a_separate_shard_and_never_pays_for_driving(factory, tmp_path, monkeypatch, phase):
    run = collection(factory, tmp_path)
    monkeypatch.setattr(preflight, "verify_receipts", lambda *_: 4)
    monkeypatch.setattr(fleet, "container_args", lambda f, name, shard, mounts, image: ("owned", tmp_path / "work", ["docker", "run", image]))
    monkeypatch.setattr(fleet, "release_work", lambda *_: None)
    monkeypatch.setattr(fleet, "sh", lambda *args, **kw: SimpleNamespace(stdout="", returncode=0))
    def execute(args, **kw):
        shard = run.path("preflight-on", "shards", phase)
        providers = yaml.safe_load((shard / "providers.local.yaml").read_text())["providers"]
        policy = yaml.safe_load((shard / "model-policy.yaml").read_text())
        env = (shard / "sidecar.env").read_text()
        assert kw["timeout"] == 300
        if phase == "fixture":
            assert policy["coordinator"]["provider_id"] == policy["lite"]["provider_id"] == "preflight-fixture"
            proxy = next(p for p in providers if p["id"] == "preflight-fixture")
            assert proxy["models"][0]["reasoning_effort"] == "none"
            assert "BIALY_PREFLIGHT_UPSTREAM=" in env and "LYCAON_LLM_MOCK" not in env
            assert "BIALY_HOSTED_PROVIDERS=preflight-fixture:FIREWORKS_API_KEY" in env
        else:
            assert "LYCAON_LLM_MOCK=1" in env and "BIALY_HOSTED_PROVIDERS=''" in env
        (shard / "preflight.json").write_text(json.dumps({"status": "passed", "phase": phase}))
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(preflight.subprocess, "run", execute)
    assert preflight.run_phase(run, "on", phase, None)["status"] == "passed"
    assert not run.path("fleet-on").exists()


def runner_module():
    path = Path(__file__).parents[1] / "runner/preflight.py"
    spec = importlib.util.spec_from_file_location("runner_preflight", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_local_fixture_forwards_only_one_typed_compaction_request(monkeypatch):
    module = runner_module()
    forwarded = []
    class Reply:
        def __enter__(self):
            return self
        def __exit__(self, *_args):
            pass
        def read(self):
            return b'{"choices":[{"message":{"content":"summary"},"finish_reason":"stop"}]}'
    def upstream(request, **kw):
        forwarded.append(json.loads(request.data))
        assert request.get_header("Authorization") == "Bearer fixture-key"
        return Reply()
    monkeypatch.setattr(module, "urlopen", upstream)
    monkeypatch.setenv("BIALY_PREFLIGHT_UPSTREAM", "https://provider.invalid/v1/chat/completions")
    server = module.ThreadingHTTPServer(("127.0.0.1", 0), module.Fixture)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    def call(body):
        with urlopen(Request("http://127.0.0.1:%d/v1/chat/completions" % server.server_port, data=json.dumps(body).encode(),
                             headers={"Authorization": "Bearer fixture-key"}), timeout=5) as response:
            return json.load(response)
    try:
        assert call({"model": "teacher", "messages": []})["choices"][0]["finish_reason"] == "stop"
        assert forwarded == []
        body = {"model": "teacher", "messages": [], "response_format": {"type": "json_schema", "json_schema": {"name": "compaction_summary"}}}
        call(body)
        assert forwarded == [body]
        from urllib.error import HTTPError
        with pytest.raises(HTTPError) as error:
            call(body)
        assert error.value.code == 400 and len(forwarded) == 1
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_runner_dispatches_the_bundled_scanner_and_requires_engine_readiness(tmp_path, monkeypatch):
    module = runner_module()
    calls = []
    def api(path, body=None):
        calls.append((path, body))
        if path == "/v1/preflight":
            return {"probes": [{"id": "decision_engine", "status": "ok"}, {"id": "scanner_engine", "status": "ok"}]}
        if path == "/v1/projects":
            return {"id": "project"}
        return {}
    class Store:
        def __enter__(self):
            return self
        def __exit__(self, *_args):
            pass
        def execute(self, sql):
            assert "scanner_id='lycaon-sast'" in sql
            return [("complete",)]
    monkeypatch.setattr(module, "api", api)
    monkeypatch.setattr(module.sqlite3, "connect", lambda *args, **kw: Store())
    monkeypatch.setenv("LYCAON_DECIDE_DEVICE", "cpu")
    module.before(str(tmp_path))
    assert calls[-1] == ("/v1/projects/project/scans", {"scanner_ids": ["lycaon-sast"]})
    monkeypatch.setenv("LYCAON_DECIDE_DEVICE", "cuda")
    with pytest.raises(module.ProbeFailure, match="engine_unavailable"):
        module.before(str(tmp_path))


def test_fixture_reduction_thresholds_preserve_the_declared_window(tmp_path):
    module = runner_module()
    source = tmp_path / "source"
    path = source / "config/packs/painted-wolf/platform/host/compaction.yaml"
    path.parent.mkdir(parents=True)
    path.write_text("window_source: explicit\nmodel_context_window: 24576\nlive_budget_pct: 75\n"
                    "budget_trigger_pct: 60\ntarget_tokens_pct: 35\nkeep_recent_messages: 16\nprune_protect_tail_messages: 8\n")
    module.configure(source, tmp_path / "probe")
    cfg = yaml.safe_load((tmp_path / "probe/config/packs/painted-wolf/platform/host/compaction.yaml").read_text())
    assert cfg["model_context_window"] == 24576 and cfg["window_source"] == "explicit" and cfg["live_budget_pct"] == 75
    assert cfg["budget_trigger_pct"] == 99 and cfg["keep_recent_messages"] == 2
    assert yaml.safe_load(path.read_text())["keep_recent_messages"] == 16


@pytest.mark.parametrize("counts, expected", [({"accepted_candidates": 4}, 4),
    ({"excluded:request control needs a model-specific renderer": 4}, None),
    ({"accepted_candidates": 1, "excluded:invalid receipt": 1}, None)])
def test_preflight_requires_receipt_admission_not_just_kept_payloads(factory, tmp_path, monkeypatch, counts, expected):
    run = collection(factory, tmp_path)
    shard = run.path("preflight-on/shards/fixture")
    (shard / "training-captures").mkdir(parents=True)
    (shard / "training-captures/one.json").write_text(json.dumps({"schema": "pw-training-turn/1", "student_request": {"tools": []}}))
    def export(args, **kw):
        assert args[1] == str(run.checkout / "scripts/bialy/custom_model/collect.py")
        source = Path(args[args.index("--fleet") + 1])
        assert (source / "shards" / "fixture").resolve() == shard.resolve()
        assert [p.name for p in (source / "shards").iterdir()] == ["fixture"]
        binding = json.loads(Path(args[args.index("--binding") + 1]).read_text())
        assert binding["student_model"] in {"gemma-4-E4B-it", "gemma-4-12B-it"} and binding["epoch"] == "fixture"
        output = Path(args[args.index("--out") + 1])
        output.mkdir()
        (output / "manifest.json").write_text(json.dumps({"counts": counts}))
        return SimpleNamespace(returncode=0, stdout="", stderr="")
    monkeypatch.setattr(preflight.subprocess, "run", export)
    if expected:
        assert preflight.verify_receipts(run, shard) == {"e4b": expected, "12b": expected}
    else:
        with pytest.raises(faults.RunFault, match="receipt_not_accepted"):
            preflight.verify_receipts(run, shard)
