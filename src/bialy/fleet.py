"""Runner containers on the GPU host: build, isolate, shard, run, and reap.

Each shard is one container: a fresh checkout of one repository, one sidecar,
and a handful of tasks driven one after another. Containers sit on a private
bridge whose only routes are the factory's model servers on the bridge
gateway and the public web on ports 80 and 443; private ranges and the cloud
metadata service are dropped. Every container this factory starts is written
to the run's ledger first, and cleanup touches only ledger entries.
"""

import json
import os
import random
import shutil
import subprocess
import time
from pathlib import Path

from . import config

RUNNER = Path(__file__).resolve().parents[2] / "runner"
CONFIG = Path(__file__).resolve().parents[2] / "config"
EGRESS_CHAIN = "BIALY-EGRESS"
INPUT_CHAIN = "BIALY-INPUT"
PRIVATE = ("10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16", "169.254.0.0/16", "100.64.0.0/10")


def sh(*args, check=True, **kw):
    return subprocess.run(list(args), check=check, text=True, capture_output=True, **kw)


DECISION_SECTIONS = ("turn", "request", "lookup", "tool_event")
# The sidecar embeds its configuration; LYCAON_CONFIG_ROOT (set in the image to the
# engine's overlay/) replaces the whole embedded tree with the config/ under it, so the
# overlay carries the complete tree of the commit the binaries were built from.
DECISIONS = Path("overlay/config/packs/painted-wolf/platform/host/decisions.yaml")


def raise_decision_deadlines(config_dir, engine_dir, deadline_ms):
    """Stage `config_dir`, the lycaon/config tree of the commit the binaries were built
    from, with a longer deadline on every turn decision.

    A runner's CPU answers a turn in seconds, and past the deadline the engine abstains.
    The deadline decides only whether an answer arrives, never what it is, so an
    engine-on pass raises it."""
    overlay = Path(engine_dir) / "overlay" / "config"
    shutil.rmtree(overlay, ignore_errors=True)
    shutil.copytree(config_dir, overlay, symlinks=False)
    path = Path(engine_dir) / DECISIONS
    lines = path.read_text(encoding="utf-8").splitlines()
    section = None
    for i, line in enumerate(lines):
        if line and not line.startswith((" ", "#")) and line.endswith(":"):
            section = line[:-1]
        elif section in DECISION_SECTIONS and line.startswith("  deadline_ms:"):
            lines[i] = "  deadline_ms: %d" % deadline_ms
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# What the sidecar needs from its engine root to offer the tools it advertises. A run
# without them still produces rows, from sessions whose git, browser, and scanner tools
# fail, so the image refuses to build instead. The scanner alone can be declined
# explicitly, for a platform the checkout pins no Opengrep release for; the sidecar
# then reports the scanner unavailable, which its scan tools surface as such.
ENGINE_PAYLOAD = ("schemas", "gitengine/bin/git", "browser/chrome-headless-shell")
SCANNER_PAYLOAD = ("opengrep/opengrep", "opengrep/provenance.json", "opengrep/source-lock.json")


PILOT_MOUNT = "/opt/decide"


def pilot_env(pilot):
    """The sidecar's decision engine for a pass: off, or the engine and heads in `pilot`
    (bialy, model/, heads/*.safetensors), mounted read-only at PILOT_MOUNT."""
    if not pilot:
        return {"LYCAON_DECIDE_DISABLED": "1"}
    heads = sorted(p.stem for p in (Path(pilot) / "heads").glob("*.safetensors"))
    if not heads:
        raise ValueError("pilot %s carries no heads" % pilot)
    return {"LYCAON_DECIDE_BINARY": PILOT_MOUNT + "/bialy", "LYCAON_DECIDE_MODEL_DIR": PILOT_MOUNT + "/model",
            "LYCAON_DECIDE_HEADS": ",".join("%s=%s/heads/%s.safetensors" % (h, PILOT_MOUNT, h) for h in heads)}


def check_engine(engine_dir, scanner=True):
    wanted = ENGINE_PAYLOAD + (SCANNER_PAYLOAD if scanner else ())
    missing = [rel for rel in wanted if not (Path(engine_dir) / rel).exists()]
    if missing:
        raise ValueError("engine payload %s lacks %s; see README (engine/)" % (engine_dir, ", ".join(missing)))


def build_image(factory, lycaon_bin, engine_dir, tag=None, deadline_ms=None, config_dir=None, scanner=True):
    """Assemble the build context from runner/, the pinned lycaon binaries, and the
    engine payload, then build the runner image."""
    check_engine(engine_dir, scanner)
    ctx = factory.root / "build" / "runner"
    if ctx.exists():
        shutil.rmtree(ctx)
    shutil.copytree(RUNNER, ctx)
    (ctx / "bin").mkdir()
    for name in ("lycaon", "lycaon-debug"):
        shutil.copy2(Path(lycaon_bin) / name, ctx / "bin" / name)
    shutil.copytree(engine_dir, ctx / "engine", symlinks=False)
    if deadline_ms:
        raise_decision_deadlines(config_dir, ctx / "engine", deadline_ms)
    shutil.copy2(CONFIG / "unattended.yaml", ctx / "unattended.yaml")
    subprocess.run(["docker", "build", "-q", "-t", tag or factory.fleet["image"], str(ctx)], check=True)


# Every runner's sidecar watches its checkout; the kernel's default of 128 inotify
# instances per user runs out long before a full fleet has started.
HOST_SYSCTL = {"fs.inotify.max_user_instances": 8192, "fs.inotify.max_user_watches": 4194304,
               "fs.inotify.max_queued_events": 65536}


def ensure_network(factory):
    """The private bridge, its egress rules, and the host limits a full fleet needs; safe
    to run again."""
    for key, value in HOST_SYSCTL.items():
        sh("sysctl", "-w", "%s=%d" % (key, value))
    net = factory.network
    inspect = sh("docker", "network", "inspect", "-f", "{{json .Options}}\t{{json .Containers}}\t{{json .IPAM.Config}}",
                 net["name"], check=False)
    exists = inspect.returncode == 0
    if exists:
        options, containers, ipam = (json.loads(part) or {} for part in inspect.stdout.strip().split("\t"))
        subnets = [c.get("Subnet") for c in ipam] if isinstance(ipam, list) else []
        isolated = options.get("com.docker.network.bridge.enable_icc") == "false"
        if not isolated and containers:
            raise RuntimeError("network %s allows inter-container traffic and has containers attached; "
                               "reap the run, then set up the network again" % net["name"])
        if subnets != [net["subnet"]] and containers:
            print("network %s is %s, not %s; it is recreated once no runner is attached" % (net["name"], subnets, net["subnet"]))
        elif not isolated or subnets != [net["subnet"]]:
            sh("docker", "network", "rm", net["name"])
            exists = False
    if not exists:
        # No inter-container traffic: a runner reaches the model servers on the
        # gateway and the public web, never another runner.
        sh("docker", "network", "create", "--subnet", net["subnet"], "--gateway", net["gateway"],
           "--opt", "com.docker.network.bridge.enable_icc=false",
           "--opt", "com.docker.network.bridge.name=br-" + net["name"], net["name"])
    bridge = "br-" + net["name"]
    ports = ",".join(str(p) for m in factory.models for p in m.ports())
    for chain, parent in ((EGRESS_CHAIN, "DOCKER-USER"), (INPUT_CHAIN, "INPUT")):
        sh("iptables", "-N", chain, check=False)
        sh("iptables", "-F", chain)
        if sh("iptables", "-C", parent, "-i", bridge, "-j", chain, check=False).returncode != 0:
            sh("iptables", "-I", parent, "1", "-i", bridge, "-j", chain)
    rules = [["-m", "conntrack", "--ctstate", "RELATED,ESTABLISHED", "-j", "RETURN"]]
    rules += [["-d", cidr, "-j", "DROP"] for cidr in PRIVATE]
    rules += [["-p", proto, "--dport", "53", "-j", "RETURN"] for proto in ("udp", "tcp")]
    rules += [["-p", "tcp", "-m", "multiport", "--dports", "80,443", "-j", "RETURN"], ["-j", "DROP"]]
    for rule in rules:
        sh("iptables", "-A", EGRESS_CHAIN, *rule)
    # The host itself answers only the model servers on the bridge.
    for rule in (["-m", "conntrack", "--ctstate", "RELATED,ESTABLISHED", "-j", "ACCEPT"],
                 ["-p", "tcp", "-m", "multiport", "--dports", ports, "-j", "ACCEPT"],
                 ["-p", "udp", "--dport", "53", "-j", "ACCEPT"], ["-j", "DROP"]):
        sh("iptables", "-A", INPUT_CHAIN, *rule)


def endpoint_provider(factory, model, port):
    """The provider instance id a shard's sidecar uses for one server of a model."""
    base = factory.provider_id(model)
    return base if port == model.port else "%s-%d" % (base, port)


def sidecar_files(factory, decide_env, lane):
    """providers.local.yaml, model-policy.yaml, and sidecar.env for a shard. `lane` picks
    which server of each model the shard's sessions use, so shards spread over replicas."""
    providers = ["providers:"]
    chosen, wire_model = {}, {}
    for m in factory.generators():
        if m.hosted:
            # A hosted generator: the sidecar reads the key from the container's environment.
            chosen[m.id] = factory.provider_id(m)
            wire_model[m.id] = m.hosted["model"]
            providers += ["  - id: %s" % chosen[m.id], "    kind: %s" % m.hosted["provider"],
                          "    base_url: %s" % m.hosted["base_url"], "    api_key_env: %s" % m.hosted["key_env"],
                          "    models: [{id: %s}]" % m.hosted["model"]]
            continue
        ports = m.ports()
        port = ports[lane % len(ports)]
        chosen[m.id] = endpoint_provider(factory, m, port)
        wire_model[m.id] = m.id
        providers += ["  - id: %s" % chosen[m.id], "    kind: openai-compatible",
                      "    base_url: %s" % factory.base_url(m, port), "    reasoning_wire: reasoning_content",
                      "    models: [{id: %s}]" % m.id]
    gens = factory.generators()
    first = gens[0]
    policy = ["coordinator: {provider_id: %s, model: %s}" % (chosen[first.id], wire_model[first.id]),
              "lite: {provider_id: %s, model: %s}" % (chosen[first.id], wire_model[first.id]),
              "agent_pool:", "  selection: random", "  models:"]
    policy += ["    - {provider_id: %s, model: %s}" % (chosen[m.id], wire_model[m.id]) for m in gens]
    # The sidecar takes provider keys only from its own credential store, so the shard
    # names which provider gets which environment variable's key; run-shard.sh stores it
    # through the API after the sidecar starts. The names travel, never the keys.
    hosted = ",".join("%s:%s" % (chosen[m.id], m.hosted["key_env"]) for m in gens if m.hosted)
    lines = dict(decide_env, BIALY_HOSTED_PROVIDERS=hosted) if hosted else dict(decide_env)
    env = "\n".join("%s=%s" % kv for kv in sorted(lines.items()))
    return "\n".join(providers) + "\n", "\n".join(policy) + "\n", env + "\n", chosen, wire_model


def plan_shards(factory, tasks_dir, run_dir, decide_env, repos=None, skip=()):
    """Split every repository's tasks into shards under run_dir/shards. Shards named in
    `skip` keep their files, and still advance the replica lanes."""
    per = int(factory.fleet["tasks_per_shard"])
    count = planned = 0
    for repo in factory.repos:
        if repos and repo.name not in repos:
            continue
        path = Path(tasks_dir) / (repo.name + ".jsonl")
        if not path.exists():
            continue
        lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        # Task files are grouped by archetype; mixed shards take similar time. Shards are
        # cut per prompt budget, so a shard of quick tasks never waits on a long budget.
        random.Random("shards:" + repo.name).shuffle(lines)
        budget = lambda line: (json.loads(line).get("meta") or {}).get("prompt_timeout") or factory.fleet["prompt_timeout"]  # noqa: E731
        lines.sort(key=lambda line: -config.minutes(budget(line)))
        # A shard of long-budget tasks could otherwise run for most of a day; its size is
        # what fits shard_minutes at the budget, from tasks_per_shard down to one task.
        groups, i = [], 0
        while i < len(lines):
            size = max(1, min(per, int(factory.fleet["shard_minutes"]) // config.minutes(budget(lines[i]))))
            group = [lines[i]]
            while len(group) < size and i + len(group) < len(lines) and budget(lines[i + len(group)]) == budget(lines[i]):
                group.append(lines[i + len(group)])
            groups.append(group)
            i += len(group)
        for index, group in enumerate(groups):
            shard = run_dir / "shards" / ("%s-%03d" % (repo.name, index))
            if shard.name in skip or (shard / "tasks.jsonl").exists():
                count += 1
                continue
            shard.mkdir(parents=True, exist_ok=True)
            providers, policy, env, chosen, wire_model = sidecar_files(factory, decide_env, count)
            shard_tasks = []
            for line in group:
                task = json.loads(line)
                # The driver names the model as the sidecar's provider knows it; the task
                # file keeps the factory id, which collect restores on the rows.
                task["provider_id"] = chosen[task["model"]]
                task["model"] = wire_model[task["model"]]
                shard_tasks.append(json.dumps(task, ensure_ascii=False))
            (shard / "tasks.jsonl").write_text("\n".join(shard_tasks) + "\n", encoding="utf-8")
            (shard / "providers.local.yaml").write_text(providers, encoding="utf-8")
            (shard / "model-policy.yaml").write_text(policy, encoding="utf-8")
            (shard / "sidecar.env").write_text(env, encoding="utf-8")
            timeout = max((budget(line) for line in group), key=config.minutes)
            (shard / "repo.json").write_text(json.dumps({"repo": repo.name, "commit": repo.commit, "prompt_timeout": timeout}),
                                             encoding="utf-8")
            count += 1
            planned += 1
    return planned


def replan(factory, tasks_dir, run_dir, decide_env, fresh=False):
    """Plan again every shard without rows, so a change in serving (a replica added or
    moved) or in the shard files reaches it; shards that produced rows or are running
    keep their files. A shard that ran and failed is planned again like one that never
    started. `fresh` discards finished shards too, for a pass planned over from the start;
    running containers are always kept and adopted."""
    run_dir = Path(run_dir)
    running = set(running_containers(run_dir))
    removed = 0
    if fresh:
        for stale in ("rows.jsonl", "tasks-driven.jsonl"):
            (run_dir / stale).unlink(missing_ok=True)
    for shard in (run_dir / "shards").iterdir():
        if shard.name in running or (not fresh and (shard / "rows.jsonl").exists()):
            continue
        shutil.rmtree(shard)
        removed += 1
    started = {s.name for s in (run_dir / "shards").iterdir()}
    return removed, plan_shards(factory, tasks_dir, run_dir, decide_env, skip=started)


class Ledger:
    """runs/<run>/ledger.jsonl: every container this run started, before it starts."""

    def __init__(self, run_dir):
        self.path = run_dir / "ledger.jsonl"

    def append(self, **entry):
        entry["at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry) + "\n")

    def containers(self):
        if not self.path.exists():
            return []
        started = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            entry = json.loads(line)
            if entry.get("event") == "start":
                started.append(entry["container"])
        return started


def reap(factory, run_dir):
    """Remove this run's containers that still exist, running or not: ledger entries only,
    by exact name, then release their work directories and cache views. `run` adopts
    running containers instead; reaping is for abandoning a run."""
    removed = []
    for name in Ledger(run_dir).containers():
        if sh("docker", "container", "inspect", name, check=False).returncode == 0:
            sh("docker", "rm", "-f", name, check=False)
            removed.append(name)
    work = factory.root / "work" / Path(run_dir).name
    for shard in sorted(work.iterdir()) if work.exists() else []:
        release_work(shard)
    return removed


def mount_cache(factory, work):
    """The shared package cache as this shard sees it: the warmed cache read-only below,
    the shard's own writes above. Nothing a runner installs reaches another runner."""
    merged = work / "cache"
    for sub in ("cache-upper", "cache-work", "cache"):
        (work / sub).mkdir()
    sh("mount", "-t", "overlay", "overlay", "-o", "lowerdir=%s,upperdir=%s,workdir=%s" % (
        factory.root / "cache", work / "cache-upper", work / "cache-work"), str(merged))
    return merged


def release_work(work):
    """Unmount a shard's cache view, then remove its work directory."""
    if (work / "cache").is_mount():
        sh("umount", str(work / "cache"), check=False)
    shutil.rmtree(work, ignore_errors=True)


def warm_cache(factory, image=None):
    """Fill the shared package cache once per repository at its pinned commit, before any
    session runs; runners then read it through their own overlay (mount_cache)."""
    (factory.root / "cache").mkdir(parents=True, exist_ok=True)
    (factory.root / "warm").mkdir(parents=True, exist_ok=True)
    procs = {}
    for repo in factory.repos:
        name = "bialy-warm-%s" % repo.name
        script = ('git clone -q --no-hardlinks /repos/{r} /work/{r} && git -C /work/{r} checkout -q {c} && '
                  '/opt/bialy/setup-repo.sh /work/{r}').format(r=repo.name, c=repo.commit)
        args = ["docker", "run", "--rm", "--name", name, "--network", factory.network["name"], "--read-only",
                "--cap-drop", "ALL", "--cap-add", "CHOWN", "--cap-add", "DAC_OVERRIDE", "--cap-add", "FOWNER",
                "--security-opt", "no-new-privileges", "--tmpfs", "/work:rw,exec,size=16g", "--tmpfs", "/shard:rw,size=1g",
                "--tmpfs", "/tmp:rw,exec,size=4g", "--tmpfs", "/root:rw,exec,size=4g",
                "-v", "%s:/repos:ro" % (factory.root / "repos"), "-v", "%s:/cache" % (factory.root / "cache"),
                "--entrypoint", "bash", image or factory.fleet["image"], "-c", script]
        # Each repository's log stays under <root>/warm/ for a failed warm to be read.
        procs[repo.name] = subprocess.Popen(args, stdout=open(factory.root / "warm" / (repo.name + ".log"), "w"), stderr=subprocess.STDOUT)
    return {name: proc.wait() for name, proc in procs.items()}


def shard_minutes(factory, shard):
    repo = json.loads((shard / "repo.json").read_text(encoding="utf-8"))
    return config.minutes(repo.get("prompt_timeout") or factory.fleet["prompt_timeout"])


def container_args(factory, run_name, shard, extra_mounts, image=None):
    fleet = factory.fleet
    repo = json.loads((shard / "repo.json").read_text(encoding="utf-8"))
    work = factory.root / "work" / run_name / shard.name
    if work.exists():
        release_work(work)
    (work / "work").mkdir(parents=True)
    cache = mount_cache(factory, work)
    name = "bialy-%s-%s" % (run_name, shard.name)
    args = ["docker", "run", "--rm", "--name", name, "--label", "bialy.run=" + run_name,
            "--network", factory.network["name"], "--cpus", str(fleet["cpus"]), "--memory", fleet["memory"],
            "--pids-limit", str(fleet["pids"]), "--cap-drop", "ALL",
            "--cap-add", "CHOWN", "--cap-add", "DAC_OVERRIDE", "--cap-add", "FOWNER", "--cap-add", "FSETID",
            "--cap-add", "KILL", "--cap-add", "SETUID", "--cap-add", "SETGID",
            "--security-opt", "no-new-privileges", "--read-only",
            "--tmpfs", "/tmp:rw,exec,size=8g", "--tmpfs", "/root:rw,exec,size=4g", "--tmpfs", "/cfg:rw,size=4g",
            "-v", "%s:/repos:ro" % (factory.root / "repos"), "-v", "%s:/shard" % shard, "-v", "%s:/work" % (work / "work"),
            "-v", "%s:/cache" % cache,
            "-e", "REPO=" + repo["repo"], "-e", "COMMIT=" + repo["commit"], "-e", "PROMPT_TIMEOUT=" + (repo.get("prompt_timeout") or fleet["prompt_timeout"])]
    for host_path, container_path in extra_mounts:
        args += ["-v", "%s:%s:ro" % (host_path, container_path)]
    # Hosted generators: the key travels as an environment variable, never in a shard file.
    for key_env in sorted({m.hosted["key_env"] for m in factory.generators() if m.hosted}):
        if os.environ.get(key_env):
            args += ["-e", key_env]
    return name, work, args + [image or fleet["image"]]


def running_containers(run_dir):
    """This run's ledger containers that are still running, by shard name."""
    out = {}
    starts = {}
    for line in (run_dir / "ledger.jsonl").read_text(encoding="utf-8").splitlines() if (run_dir / "ledger.jsonl").exists() else []:
        entry = json.loads(line)
        if entry.get("event") == "start":
            starts[entry["container"]] = entry["shard"]
    for name, shard in starts.items():
        state = sh("docker", "container", "inspect", "-f", "{{.State.Running}}", name, check=False)
        if state.returncode == 0 and state.stdout.strip() == "true":
            out[shard] = name
    return out


# A shard's tasks each get their prompt budget, and the runner spends time on setup,
# exports, and the turns after a timed-out prompt, so a shard is overdue only well past
# the sum of its budgets.
OVERDUE_FACTOR, OVERDUE_GRACE_MINUTES = 1.5, 30


def shard_deadline(factory, shard, started):
    """When a shard started at `started` (epoch seconds) counts as stuck."""
    tasks = sum(1 for line in (shard / "tasks.jsonl").read_text(encoding="utf-8").splitlines() if line.strip())
    return started + 60 * (OVERDUE_FACTOR * tasks * shard_minutes(factory, shard) + OVERDUE_GRACE_MINUTES)


def run(factory, run_dir, runners=None, extra_mounts=(), image=None):
    """Run every shard without rows, at most `runners` at a time, until all have finished.

    A restarted run adopts its containers that are still running and waits for them; a
    ledger container that has stopped without rows is removed and its shard runs again.
    A container still running past its shard's deadline is killed and counted failed."""
    run_dir = Path(run_dir)
    run_name = run_dir.name
    ledger = Ledger(run_dir)
    adopted = running_containers(run_dir)
    for name in Ledger(run_dir).containers():
        if name not in adopted.values() and sh("docker", "container", "inspect", name, check=False).returncode == 0:
            sh("docker", "rm", "-f", name, check=False)
            ledger.append(event="reaped", container=name)
    (factory.root / "cache").mkdir(parents=True, exist_ok=True)
    active = {}
    for shard_name, name in adopted.items():
        shard = run_dir / "shards" / shard_name
        ledger.append(event="adopted", container=name, shard=shard_name)
        active[name] = (subprocess.Popen(["docker", "wait", name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL),
                        shard, factory.root / "work" / run_name / shard_name, shard_deadline(factory, shard, time.time()))
    # Longest budgets first: the shards that run longest start while the fleet is full,
    # instead of trailing a nearly idle fleet at the end of the pass.
    pending = sorted((s for s in (run_dir / "shards").iterdir() if not (s / "rows.jsonl").exists() and s.name not in adopted),
                     key=lambda s: (-shard_minutes(factory, s), s.name))
    for shard in pending:
        for stale in ("manifest.jsonl", "runner.log", "sidecar.log", "setup.log", "rows.part"):
            (shard / stale).unlink(missing_ok=True)
    limit = int(runners or factory.fleet["runners"])
    done = failed = 0
    while pending or active:
        while pending and len(active) < limit:
            shard = pending.pop(0)
            name, work, args = container_args(factory, run_name, shard, extra_mounts, image)
            ledger.append(event="start", container=name, shard=shard.name)
            active[name] = (subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=open(shard / "docker.log", "w")), shard, work,
                            shard_deadline(factory, shard, time.time()))
        time.sleep(5)
        for name, (proc, shard, work, deadline) in list(active.items()):
            code = proc.poll()
            if code is None and time.time() > deadline:
                sh("docker", "kill", name, check=False)
                ledger.append(event="overdue", container=name, shard=shard.name)
                (shard / "rows.jsonl").unlink(missing_ok=True)
                code = proc.wait()
            if code is None:
                continue
            del active[name]
            ledger.append(event="exit", container=name, shard=shard.name, code=code)
            release_work(work)
            if (shard / "rows.jsonl").exists():
                done += 1
            else:
                failed += 1
            print("%s exit=%s done=%d failed=%d active=%d pending=%d" % (shard.name, code, done, failed, len(active), len(pending)), flush=True)
    return done, failed


def salvage(run_dir, into):
    """Keep what a run's live containers already drove before the run is abandoned.

    For every running container, export the rows of the sessions its manifest lists from
    the live store, and copy them with the manifest into `into`/shards/<shard>, which
    `collect` reads like any run. Returns the task ids that settled anywhere in the run."""
    run_dir, into = Path(run_dir), Path(into)
    saved = 0
    for shard_name, name in running_containers(run_dir).items():
        shard = run_dir / "shards" / shard_name
        manifest = shard / "manifest.jsonl"
        if not manifest.exists() or not manifest.read_text(encoding="utf-8").strip():
            continue
        roots = [json.loads(line)["root_session"] for line in manifest.read_text(encoding="utf-8").splitlines()]
        (shard / "salvage-roots.txt").write_text("\n".join(roots) + "\n", encoding="utf-8")
        result = sh("docker", "exec", name, "/opt/bialy/bin/lycaon-debug", "decide", "export", "--db", "/cfg/store.db",
                    "--roots", "/shard/salvage-roots.txt", "--out", "/shard/salvage-rows.jsonl", check=False)
        if result.returncode != 0:
            print("%s: export failed: %s" % (shard_name, result.stderr.strip()[:200]))
            continue
        dest = into / "shards" / shard_name
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copy2(manifest, dest / "manifest.jsonl")
        shutil.copy2(shard / "salvage-rows.jsonl", dest / "rows.jsonl")
        saved += 1
    settled = set()
    for manifest in (run_dir / "shards").glob("*/manifest.jsonl"):
        for line in manifest.read_text(encoding="utf-8").splitlines():
            entry = json.loads(line)
            if entry["status"] == "settled":
                settled.add(entry["task_id"])
    return saved, settled


def collect(run_dir, pass_name, tasks_dir):
    """Every shard's rows, each carrying its task's metadata, and tasks-driven.jsonl: the
    full specification of every task the run drove, with its outcome, so a release can
    say exactly which tasks it contains however the run ended."""
    run_dir = Path(run_dir)
    specs = {}
    for path in Path(tasks_dir).glob("*.jsonl"):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                task = json.loads(line)
                specs[task["id"]] = task
    out = run_dir / "rows.jsonl"
    stats = {"rows": 0, "shards": 0, "tasks": 0, "settled": 0}
    driven = []
    with open(out, "w", encoding="utf-8") as fh:
        for shard in sorted((run_dir / "shards").iterdir()):
            rows = shard / "rows.jsonl"
            manifest = shard / "manifest.jsonl"
            if not rows.exists() or not manifest.exists():
                continue
            stats["shards"] += 1
            meta, model_of = {}, {}
            for line in manifest.read_text(encoding="utf-8").splitlines():
                entry = json.loads(line)
                if entry["task_id"] not in specs:
                    raise ValueError("%s: task %s is not in %s" % (shard.name, entry["task_id"], tasks_dir))
                stats["tasks"] += 1
                stats["settled"] += entry["status"] == "settled"
                # The task file, not the driver's echo, is what a row's grouping comes from.
                meta[entry["root_session"]] = dict(specs[entry["task_id"]]["meta"], task_status=entry["status"], run=run_dir.name, pass_name=pass_name)
                model_of[entry["root_session"]] = specs[entry["task_id"]]["model"]
                driven.append(dict(specs[entry["task_id"]], outcome={"status": entry["status"], "run": run_dir.name, "shard": shard.name,
                                                                    "pass_name": pass_name, "root_session": entry["root_session"]}))
            for line in rows.read_text(encoding="utf-8").splitlines():
                row = json.loads(line)
                if row["root_session"] not in meta:
                    raise ValueError("%s: rows from session %s, which the manifest does not list" % (shard.name, row["root_session"]))
                row["meta"] = meta[row["root_session"]]
                row["model"] = model_of[row["root_session"]]
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                stats["rows"] += 1
    with open(run_dir / "tasks-driven.jsonl", "w", encoding="utf-8") as fh:
        for task in driven:
            fh.write(json.dumps(task, ensure_ascii=False) + "\n")
    return out, stats

