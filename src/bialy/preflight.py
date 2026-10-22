"""Qualify each runner image before admitting paid driving."""

import copy
import json
import shlex
import subprocess

import yaml

from . import custom_model_run, faults, fleet


def verify_window(run, pass_name):
    custom = run.setting("custom_model") or {}
    if not custom.get("enabled"):
        return
    result = fleet.sh("docker", "run", "--rm", "--entrypoint", "cat", run.image(pass_name),
                      "/opt/bialy/engine/" + str(fleet.COMPACTION))
    config = yaml.safe_load(result.stdout)
    window = int(custom["student_serving"]["context_window"])
    if config["window_source"] != "explicit" or int(config["model_context_window"]) != window * int(config["live_budget_pct"]) // 100:
        raise faults.RunFault("preflight_student_window_mismatch", "image")


def run_phase(run, pass_name, phase, pilot):
    directory = run.path("preflight-" + pass_name)
    shard = directory / "shards" / phase
    if shard.exists():
        raise faults.RunFault("preflight_evidence_exists_use_new_run", phase)
    shard.mkdir(parents=True)
    custom = run.setting("custom_model") or {}
    environment = fleet.pilot_env(pilot)
    environment["BIALY_PREFLIGHT"] = phase
    if phase == "mock":
        environment["LYCAON_LLM_MOCK"] = "1"
    if custom.get("enabled"):
        serving = custom["student_serving"]
        environment.update(PW_TRAINING_CAPTURE_DIR="/shard/training-captures", PW_TRAINING_EPOCH=custom["epoch"],
                           PW_TRAINING_STUDENT_PROVIDER=serving["provider"], PW_TRAINING_STUDENT_MODEL=serving["model"])
    providers, policy, _, chosen, models = fleet.sidecar_files(run.factory, environment, 0)
    if phase == "fixture":
        teacher = run.factory.generators()[0]
        if not teacher.hosted:
            raise faults.RunFault("preflight_requires_hosted_summary_provider", phase)
        value = yaml.safe_load(providers)
        proxy = copy.deepcopy(next(p for p in value["providers"] if p["id"] == chosen[teacher.id]))
        environment["BIALY_PREFLIGHT_UPSTREAM"] = proxy["base_url"].rstrip("/") + "/chat/completions"
        proxy.update(id="preflight-fixture", base_url="http://127.0.0.1:8851/v1")
        value["providers"].append(proxy)
        providers = yaml.safe_dump(value)
        value = yaml.safe_load(policy)
        value["coordinator"] = value["lite"] = {"provider_id": "preflight-fixture", "model": models[teacher.id]}
        value["agent_pool"]["models"] = [value["coordinator"]]
        policy = yaml.safe_dump(value)
        environment["BIALY_HOSTED_PROVIDERS"] = "preflight-fixture:" + teacher.hosted["key_env"]
    else:
        # A mock turn never needs provider credentials.
        environment["BIALY_HOSTED_PROVIDERS"] = ""
    env = "".join("%s=%s\n" % (key, shlex.quote(str(value))) for key, value in sorted(environment.items()))
    for name, body in (("providers.local.yaml", providers), ("model-policy.yaml", policy), ("sidecar.env", env)):
        (shard / name).write_text(body)
    (shard / "workspace.json").write_text(json.dumps({"workspace": "preflight", "kind": "stack", "prompt_timeout": "3m"}))
    task = {"id": "preflight", "prompt": "Repeat a fixture observation for a transcript transport check."}
    if phase == "fixture":
        task["follow_ups"] = [{"prompt": "Repeat the fixture observation again."} for _ in range(3)]
    (shard / "tasks.jsonl").write_text(json.dumps(task) + "\n")
    mounts = [(str(pilot), fleet.PILOT_MOUNT)] if pilot else []
    name, work, args = fleet.container_args(run.factory, directory.name, shard, mounts, run.image(pass_name))
    fleet.Ledger(directory).append(event="start", container=name, shard=phase)
    try:
        # Host stderr may contain identifying paths; keep it only in this private run.
        with (shard / "docker.log").open("w") as log:
            result = subprocess.run(args, stdout=log, stderr=log, timeout=int(run.setting("preflight")["timeout_seconds"]))
        report = json.loads((shard / "preflight.json").read_text()) if (shard / "preflight.json").exists() else {}
        if result.returncode or report.get("status") != "passed":
            raise faults.RunFault(report.get("code", "preflight_runner_failed"), phase)
        return report
    except subprocess.TimeoutExpired as exc:
        raise faults.RunFault("preflight_timeout", phase) from exc
    finally:
        fleet.sh("docker", "stop", "--time", "10", name, check=False)
        fleet.release_work(work)


def qualify(run, pass_name, pilot=None):
    if not run.setting("preflight")["enabled"]:
        if run.setting("custom_model").get("enabled"):
            raise faults.RunFault("preflight_required_for_collection", pass_name)
        return {"skipped": "preflight disabled"}
    verify_window(run, pass_name)
    phases = [run_phase(run, pass_name, "mock", pilot)]
    if run.setting("custom_model").get("enabled"):
        phases.append(run_phase(run, pass_name, "fixture", pilot))
    report = {"schema": "pw-runner-preflight/1", "image": run.image(pass_name), "phases": phases,
              "image_id": fleet.sh("docker", "image", "inspect", "--format", "{{.Id}}", run.image(pass_name)).stdout.strip(),
              "settings_sha256": custom_model_run.digest(run.settings)}
    custom_model_run.write_json(run.path("preflight-" + pass_name, "report.json"), report)
    return report


def require(run, pass_name):
    if not run.setting("custom_model").get("enabled"):
        return
    stage = "preflight_on" if pass_name == "on" else "preflight"
    record = run.load_state()["stages"].get(stage) or {}
    report = record.get("outputs") or {}
    image_id = fleet.sh("docker", "image", "inspect", "--format", "{{.Id}}", run.image(pass_name)).stdout.strip()
    if (record.get("status") != "done" or not image_id or report.get("image_id") != image_id or
            report.get("settings_sha256") != custom_model_run.digest(run.settings)):
        raise faults.RunFault("preflight_missing_or_stale", pass_name)
