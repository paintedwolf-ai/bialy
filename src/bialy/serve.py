"""The factory's vLLM servers: one per configured model, bound to the private bridge.

A server is started in its own session with its PID and start time recorded,
and stopped only by that PID after checking the start time still matches, so
a reused PID is never signalled.
"""

import json
import os
import signal
import subprocess
import time
import urllib.request


def _start_time(pid):
    try:
        with open("/proc/%d/stat" % pid, encoding="utf-8") as fh:
            return fh.read().rsplit(")", 1)[1].split()[19]
    except OSError:
        return None


def _record(factory, model):
    return factory.root / "run" / ("vllm-%s.json" % model.id)


def snapshot(factory, model):
    return factory.root / "hf" / "hub" / ("models--" + model.hf.replace("/", "--")) / "snapshots" / model.revision


def start(factory, model):
    record = _record(factory, model)
    if running(factory, model):
        return json.loads(record.read_text())["pid"]
    record.parent.mkdir(parents=True, exist_ok=True)
    # Only what vLLM needs: the weights are local, so no token or other operator
    # credential reaches the server process.
    env = dict({k: os.environ[k] for k in ("HOME", "LANG", "LD_LIBRARY_PATH", "TMPDIR", "USER") if k in os.environ}, HF_HOME=str(factory.root / "hf"), HF_HUB_OFFLINE="1", CUDA_VISIBLE_DEVICES=str(model.serve["gpu"]),
               CUDA_HOME="/usr/local/cuda", PATH="%s:/usr/local/cuda/bin:%s" % (factory.root / "venvs/vllm/bin", os.environ["PATH"]))
    args = [str(factory.root / "venvs/vllm/bin/vllm"), "serve", str(snapshot(factory, model)), "--served-model-name", model.id,
            "--host", factory.network["gateway"], "--port", str(model.port), "--max-model-len", "131072",
            "--gpu-memory-utilization", "0.92", "--enable-prefix-caching", "--enable-auto-tool-choice", *model.serve["args"]]
    log = open(factory.root / "logs" / ("vllm-%s.log" % model.id), "w")
    proc = subprocess.Popen(args, env=env, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
    time.sleep(1)
    record.write_text(json.dumps({"pid": proc.pid, "start": _start_time(proc.pid), "args": args}))
    return proc.pid


def _tunnel_record(factory, model, replica):
    return factory.root / "run" / ("tunnel-%s-%d.json" % (model.id, int(replica["port"])))


def start_tunnels(factory, model):
    """One supervised SSH tunnel per replica, from the bridge gateway to the replica's
    loopback server; the loop reconnects if the connection drops."""
    for replica in model.replicas:
        record = _tunnel_record(factory, model, replica)
        if record.exists() and _start_time(json.loads(record.read_text())["pid"]) == json.loads(record.read_text())["start"]:
            continue
        forward = "%s:%d:%s" % (factory.network["gateway"], int(replica["port"]), replica["remote"])
        loop = ("while true; do ssh -N -i %s -o BatchMode=yes -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 "
                "-o StrictHostKeyChecking=accept-new -o GatewayPorts=yes -L %s %s; sleep 5; done") % (replica["key"], forward, replica["ssh"])
        log = open(factory.root / "logs" / ("tunnel-%s-%d.log" % (model.id, int(replica["port"]))), "w")
        proc = subprocess.Popen(["bash", "-c", loop], stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
        time.sleep(1)
        record.write_text(json.dumps({"pid": proc.pid, "start": _start_time(proc.pid), "forward": forward, "host": replica["ssh"]}))


def stop_tunnels(factory, model):
    for replica in model.replicas:
        record = _tunnel_record(factory, model, replica)
        if not record.exists():
            continue
        entry = json.loads(record.read_text())
        if _start_time(entry["pid"]) == entry["start"]:
            os.killpg(entry["pid"], signal.SIGTERM)
        record.unlink()


def running(factory, model):
    record = _record(factory, model)
    if not record.exists():
        return False
    entry = json.loads(record.read_text())
    return _start_time(entry["pid"]) == entry["start"]


def ready(factory, model):
    """Every server of the model answers, replicas included."""
    for port in model.ports():
        try:
            with urllib.request.urlopen(factory.base_url(model, port) + "/models", timeout=5) as resp:
                if resp.status != 200:
                    return False
        except OSError:
            return False
    return True


def stop(factory, model):
    record = _record(factory, model)
    if not running(factory, model):
        record.unlink(missing_ok=True)
        return False
    entry = json.loads(record.read_text())
    os.killpg(entry["pid"], signal.SIGTERM)
    record.unlink()
    return True
