"""One command on one machine: `bialy run` drives a pass from repositories to a
head release, resumable, with nothing uploaded unless asked.

Every stage writes its outputs under <root>/runs/<name>/ and its status to
run.json there. A rerun skips finished stages; `--from` reruns from a stage,
`--until` stops after one. Settings come from factory.yaml's `run` section and
the flags that override it for one run.

The pass has two halves. The first drives every task with the decision engine
off and trains pilot heads on what the judges labelled. The second drives the
same tasks again with those heads deciding, so the release also holds rows from
turns the engine shaped, then trains the final heads over both.
"""
import concurrent.futures
import dataclasses
import json
import os
import platform
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path

from . import (
    anchor,
    build,
    coderank,
    config,
    custom_model_run,
    fleet,
    heads,
    judge,
    preflight,
    provenance,
    publishing,
    release,
    repos,
    retention,
    skillreq,
    split,
    tasks,
    training,
)

STAGES = ("check", "repos", "build", "image", "corpus", "tasks", "skillreq", "warm", "preflight", "plan", "drive", "collect", "judge",
          "judge_skillreq", "split_pilot", "train_pilot", "engine", "pilot", "preflight_on", "plan_on", "drive_on", "collect_on", "compare", "judge_on",
          "split", "custom_model", "coderank", "release", "train", "evaluate", "release_heads", "report")
ENGINE_ON_STAGES = ("split_pilot", "train_pilot", "pilot", "preflight_on", "plan_on", "drive_on", "collect_on", "compare", "judge_on")


class RunError(RuntimeError):
    pass


@dataclasses.dataclass
class Run:
    factory: object
    name: str
    settings: dict
    push: bool = False
    rebuild_image: bool = False
    # until names the last stage this invocation runs; check requires only what those stages use.
    until: str = None

    def runs(self, *stages):
        last = STAGES.index(self.until) if self.until else len(STAGES) - 1
        return any(STAGES.index(s) <= last for s in stages)

    @property
    def dir(self):
        return self.factory.root / "runs" / self.name

    @property
    def state_path(self):
        return self.dir / "run.json"

    def load_state(self):
        if self.state_path.exists():
            return json.loads(self.state_path.read_text(encoding="utf-8"))
        return {"name": self.name, "stages": {}}

    def save_state(self, state):
        self.dir.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")

    def outputs(self, stage):
        return (self.load_state()["stages"].get(stage) or {}).get("outputs") or {}

    def path(self, *parts):
        return self.dir.joinpath(*parts)

    def setting(self, key, default=None):
        return self.settings.get(key, default)

    @property
    def checkout(self):
        return Path(self.setting("lycaon_checkout"))

    @property
    def cache_root(self):
        return self.factory.root / "build-cache"

    @property
    def bin_dir(self):
        return self.path("build", "bin")

    @property
    def host_bin_dir(self):
        return self.bin_dir if build.host_is_runner_platform() else self.path("build", "host-bin")

    @property
    def engine_dir(self):
        return self.path("build", "engine")

    @property
    def scanner(self):
        return run_scanner(self.setting("scanner"), self.setting("scanner_candidate"))

    @property
    def scanner_candidate(self):
        return self.setting("scanner_candidate") if self.setting("scanner") == "candidate" else None

    @property
    def engine_on(self):
        return bool(self.setting("engine_on", {}).get("enabled"))

    @property
    def collection_window(self):
        """The student's declared context window when this run collects custom-model data."""
        custom = self.setting("custom_model") or {}
        return int(custom["student_serving"]["context_window"]) if custom.get("enabled") else None

    @property
    def engine_on_only(self):
        """The engine-on pass drives with released heads, so no engine-off pass or pilot training runs."""
        return self.engine_on and bool(self.setting("engine_on", {}).get("heads"))

    def image(self, pass_name):
        return "%s:%s-%s" % (self.factory.fleet["image"], self.name, pass_name)


def run_scanner(value, candidate=None):
    """Whether the payload carries a scanner: the pinned release, a candidate build, or
    none. Returns False for none and True otherwise."""
    if value not in ("pinned", "none", "candidate"):
        raise RunError("run.scanner must be pinned, candidate, or none, not %r" % value)
    if value == "candidate" and not candidate:
        raise RunError("run.scanner: candidate needs run.scanner_candidate, the artifact directory of a linux/amd64 build")
    return value != "none"


def stamp():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def run_generators(run):
    named = run.setting("generators") or []
    if named:
        generators = [m for m in run.factory.models if m.id in named and ("generator" in m.roles or "teacher" in m.roles)]
    else:
        generators = run.factory.generators()
    return [m for m in generators if m.hosted or m.serve]


def writer_model(run):
    """The model that writes skill requests and code-rank pairs: the configured one, or the
    first writer that is hosted or served here."""
    named = run.setting("skillreq", {}).get("writer")
    for m in run.factory.models:
        if (m.hosted or m.serve) and ((named and m.id == named) or (not named and "writer" in m.roles)):
            return m
    raise RunError("no writer model is usable; give one the writer role in models.yaml")


def roster(run):
    """The models with the roles this run gives them: run.generators and run.judges keep
    those roles on the models they name and take them off the rest; a model keeps its
    other roles either way."""
    generators = {m.id for m in run_generators(run)}
    judges = set(run.setting("judges") or [])
    out = []
    for m in run.factory.models:
        base_roles = list(m.roles)
        if m.id in generators and "generator" not in base_roles:
            base_roles.append("generator")
        if m.id in judges and "judge" not in base_roles and (m.serve or "generator" in base_roles or "teacher" in base_roles):
            base_roles.append("judge")
        roles = tuple(r for r in base_roles if not (r == "generator" and m.id not in generators) and not (r == "judge" and judges and m.id not in judges))
        if roles:
            out.append(dataclasses.replace(m, roles=roles))
    return out


def scaled_factory(run):
    """The factory with generation limited to the run's cap, roster, and workspaces."""
    f = run.factory
    cap, share = run.setting("task_cap"), run.setting("task_share")
    if share is not None and not 0 < float(share) <= 1:
        raise RunError("run.task_share is a fraction of every archetype's tasks, above 0 and at most 1")
    models = roster(run)
    # A share keeps the configured mix at a smaller volume; a cap flattens it to at most n per archetype.
    archetypes = [dataclasses.replace(a, per_workspace=max(1, round(a.per_workspace * float(share)))) if share else a for a in f.archetypes]
    archetypes = [dataclasses.replace(a, per_workspace=min(a.per_workspace, int(cap))) if cap else a for a in archetypes]
    chosen = set(run.setting("archetypes") or [])
    unknown = chosen - {a.id for a in f.archetypes}
    if unknown:
        raise RunError("run.archetypes names unknown archetypes: %s" % ", ".join(sorted(unknown)))
    archetypes = [a for a in archetypes if not chosen or a.id in chosen]
    archetypes = retention.shape_archetypes(archetypes, run.setting("retention", {}))
    names = set(run.setting("workspaces") or [])
    return dataclasses.replace(f, models=models, archetypes=archetypes, repos=[r for r in f.repos if not names or r.name in names],
                               stacks=[s for s in f.stacks if not names or s.name in names], run=run.settings)


# ---- check --------------------------------------------------------------------

CHECKOUT_FILES = ("lycaon/cmd/lycaon", "lycaon/cmd/lycaon-debug", "lycaon/cmd/decide-rerank", "lycaon/cmd/opengrep-artifact",
                  "lycaon/internal/decide/native/Cargo.toml", "lycaon/config/gitengine/pin.yaml", "rust-toolchain.toml", "schemas",
                  "scripts/bialy/train.py", "scripts/bialy/row.schema.json", "scripts/bialy/replay_eval.py", "scripts/bialy/rerank/train_rerank.py",
                  "scripts/bialy/retention-row.schema.json", "scripts/bialy/retention_eval.py")


def stage_check(run):
    f = run.factory
    faults = []
    generators = run_generators(run)
    cfg = run.setting("custom_model") or {}
    if cfg.get("enabled"):
        try:
            custom_model_run.validate_settings(cfg)
        except custom_model_run.DatasetError as exc:
            faults.append(str(exc))
    if not generators:
        faults.append("no generator model is usable: name hosted generators in run.generators or serve one locally")
    used = list(generators)
    scaled = scaled_factory(run)
    generator_ids = [model.id for model in generators]
    inactive = set(run.setting("generators") or []) - set(generator_ids)
    if inactive:
        faults.append("run.generators names unavailable generators: %s" % ", ".join(sorted(inactive)))
    routing = run.setting("archetype_generators") or {}
    unknown_archetypes = set(routing) - {a.id for a in scaled.archetypes}
    if unknown_archetypes:
        faults.append("run.archetype_generators names unknown archetypes: %s" % ", ".join(sorted(unknown_archetypes)))
    if generators:
        for archetype in scaled.archetypes:
            try:
                if archetype.id in routing and not isinstance(routing[archetype.id], (dict, list)):
                    raise config.ConfigError("generator routing for %s must be a list or weight mapping" % archetype.id)
                tasks.pick_generator(scaled, archetype.id, generator_ids, random.Random(0))
            except config.ConfigError as exc:
                faults.append(str(exc))
                break
    unknown = set(run.setting("workspaces") or []) - {w.name for w in f.workspaces()}
    if unknown:
        faults.append("run.workspaces names %s, which no repository or stack is" % ", ".join(sorted(unknown)))
    for name in run.setting("judges") or []:
        if not any(m.id == name and ("judge" in m.roles or m.serve) for m in f.models):
            faults.append("run.judges names %s, which is not a judge in models.yaml" % name)
    for model in generators:
        try:
            used.append(scaled.judge_for(model.id))
        except config.ConfigError as exc:
            faults.append(str(exc))
    try:
        used.append(writer_model(run))
    except RunError as exc:
        faults.append(str(exc))
    for role, action in (("student", "shadow"), ("teacher", "repair")):
        if cfg.get("enabled") and cfg.get(action):
            try:
                model = f.model(cfg[role])
                if not model.hosted and not model.serve:
                    faults.append("custom-model %s has no configured endpoint" % role)
                used.append(model)
            except (config.ConfigError, KeyError) as exc:
                faults.append(str(exc))
    for model in {m.id: m for m in used}.values():
        if model.hosted and not os.environ.get(model.hosted["key_env"]):
            faults.append("model %s needs %s in the environment" % (model.id, model.hosted["key_env"]))
    for rel in CHECKOUT_FILES:
        if not (run.checkout / rel).exists():
            faults.append("run.lycaon_checkout lacks %s" % rel)
    if shutil.which("docker") is None:
        faults.append("docker is not on PATH")
    if run.runs("warm") and hasattr(os, "geteuid") and os.geteuid() != 0:
        faults.append("the fleet stages mount cache overlays and set iptables rules: run as root (sudo)")
    if shutil.which("uv") is None and run.runs("train_pilot", "train"):
        faults.append("uv is not on PATH; the trainer's environment is made with it")
    try:
        scanner = run.scanner
    except RunError as exc:
        scanner = None
        faults.append(str(exc))
    recipes = run.setting("train", {}).get("recipes") or []
    for recipe in recipes:
        if recipe not in training.RECIPES:
            faults.append("unknown training recipe %s; known: %s" % (recipe, ", ".join(sorted(training.RECIPES))))
    ceiling = float(run.setting("spend_ceiling_usd") or 0)
    priced = [m.id for m in used if (m.hosted or {}).get("price_per_million")]
    if ceiling > 0 and not priced:
        faults.append("run.spend_ceiling_usd needs hosted.price_per_million on at least one model it would count")
    if faults:
        raise RunError("; ".join(faults))
    return {"generators": [m.id for m in generators], "judges": sorted({scaled.judge_for(m.id).id for m in generators}),
            "writer": writer_model(run).id, "repositories": len(scaled.repos), "stacks": len(scaled.stacks),
            "runners": int(run.setting("runners") or f.fleet["runners"]), "engine_on": run.engine_on, "recipes": recipes,
            "scanner": run.setting("scanner") if scanner else "none",
            "accelerator": training.accelerator(), "go": "host" if shutil.which("go") else build.GO_IMAGE,
            "cargo": "rustup" if shutil.which("rustup") else build.rust_image(run.checkout), "priced_models": priced,
            "spend_note": "usage is checked between stages; spend_ceiling_usd is not a hard provider spending cap"}


# ---- sources -------------------------------------------------------------------

def stage_repos(run):
    problems = repos.fetch(run.factory) or repos.verify(run.factory)
    if problems:
        raise RunError("; ".join(problems))
    return {"repositories": [r.name for r in run.factory.repos]}


def stage_build(run):
    """The runner binaries and engine payload from the pinned checkout, plus binaries for
    this host when it is not the runner platform."""
    # A pinned scanner's identity is linked into the runner binaries; a development
    # candidate is passed to the sidecar at run time instead.
    identity = build.scanner_identity(run.checkout, run.cache_root) if run.scanner and not run.scanner_candidate else None
    out = {"runner": build.binaries(run.checkout, run.bin_dir, run.cache_root, scanner_identity=identity)}
    if not build.host_is_runner_platform():
        goos = platform.system().lower()
        goarch = {"x86_64": "amd64", "amd64": "amd64", "arm64": "arm64", "aarch64": "arm64"}[platform.machine()]
        out["host"] = build.binaries(run.checkout, run.host_bin_dir, run.cache_root, goos, goarch)
    out["payload"] = build.payload(run.checkout, run.engine_dir, run.bin_dir, run.cache_root, scanner=run.scanner, candidate=run.scanner_candidate)
    return out


def stage_image(run):
    tag = run.image("off")
    fleet.ensure_network(run.factory)
    present = subprocess.run(["docker", "image", "inspect", tag], capture_output=True).returncode == 0
    if present and not run.rebuild_image:
        return {"image": tag, "built": False}
    fleet.build_image(run.factory, run.bin_dir, run.engine_dir, tag, config_dir=run.checkout / "lycaon/config",
                      scanner=run.scanner, context_window=run.collection_window)
    return {"image": tag, "built": True, "scanner": run.scanner, "context_window": run.collection_window}


def stage_corpus(run):
    """corpus.json from the runners' own build, so option texts match the rows."""
    out = run.path("corpus.json")
    debug = run.host_bin_dir / "lycaon-debug"
    result = subprocess.run([str(debug), "decide", "corpus", "--out", str(out)], capture_output=True, text=True)
    if result.returncode != 0:
        raise RunError("lycaon-debug decide corpus failed: %s" % result.stderr.strip()[-300:])
    revision = json.loads(out.read_text(encoding="utf-8")).get("catalog_revision")
    return {"corpus": str(out), "catalog_revision": revision}


def stage_tasks(run):
    out = run.path("tasks")
    scaled = scaled_factory(run)
    counts = tasks.generate(scaled, run.factory.root / "repos", out, seed=int(run.setting("seed", 7)), workers=int(run.setting("workers", 16)))
    provenance.record(run.factory, out / "provenance.json", "tasks", uses_engine=False, seed=int(run.setting("seed", 7)), counts=counts)
    return {"tasks": str(out), "counts": counts}


def corpus(run):
    return json.loads(run.path("corpus.json").read_text(encoding="utf-8"))


def recipes(run):
    return run.setting("train", {}).get("recipes") or []


def judging_off(run):
    """Whether this run passes rows through unjudged: judging is a paid stage only the
    tool, guide, and skill recipes need."""
    return not run.setting("judge", {}).get("enabled", True)


def stage_skillreq(run):
    """Requests for every skill, written by the writer model; the judges label them later."""
    if "E4" not in recipes(run):
        return {"skipped": "E4 is not in run.train.recipes; skill requests are only for the unit-rank head"}
    settings = run.setting("skillreq", {})
    out = run.path("skillreq", "requests.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    writer = writer_model(run)
    families = {k: int(v) for k, v in settings["families"].items()}
    cap = run.setting("task_cap")
    if cap:
        families = {k: min(v, 1) for k, v in families.items()}
    none_families = min(int(settings["none_families"]), 2) if cap else int(settings["none_families"])
    report = skillreq.write(scaled_factory(run), corpus(run), out, writer.id, families, none_families, int(settings["per_family"]),
                            seed=int(run.setting("seed", 7)), workers=int(run.setting("workers", 16)),
                            workspaces=tuple(run.setting("workspaces") or ()))
    provenance.record(run.factory, str(out) + ".provenance.json", "skillreq", uses_engine=False, writer=writer.id, families=families,
                      none_families=none_families, per_family=int(settings["per_family"]), seed=int(run.setting("seed", 7)),
                      corpus_revision=corpus(run)["catalog_revision"])
    if not report["rows"]:
        raise RunError("the writer produced no skill requests")
    return {"requests": str(out), "report": report}


# ---- passes --------------------------------------------------------------------

def stage_warm(run):
    scaled = scaled_factory(run)
    failed = {name: code for name, code in fleet.warm_cache(scaled, run.image("off")).items() if code}
    if failed:
        raise RunError("cache warm failed for %s; see %s" % (", ".join(sorted(failed)), run.factory.root / "warm"))
    return {"warmed": len(scaled.repos) + sum(1 for s in scaled.stacks if s.warm)}


def plan_pass(run, pass_name, pilot):
    run_dir = run.path("fleet-" + pass_name)
    run_dir.mkdir(parents=True, exist_ok=True)
    decide_env = fleet.pilot_env(pilot)
    custom = run.setting("custom_model") or {}
    if custom.get("enabled"):
        serving = custom["student_serving"]
        decide_env.update(PW_TRAINING_CAPTURE_DIR="/shard/training-captures", PW_TRAINING_EPOCH=custom["epoch"],
                          PW_TRAINING_STUDENTS=json.dumps([{"provider": serving["provider"], "model": serving["model"]}]))
    if (run_dir / "shards").exists():
        # A plan stage runs again only when its pass is being redone, so earlier shards
        # of the pass are stale; containers still running are kept and adopted.
        _, planned = fleet.replan(scaled_factory(run), run.path("tasks"), run_dir, decide_env, fresh=True)
    else:
        planned = fleet.plan_shards(scaled_factory(run), run.path("tasks"), run_dir, decide_env, run.setting("workspaces") or None)
    tasks_record = run.path("tasks", "provenance.json")
    provenance.record(run.factory, run_dir / "provenance.json", "fleet", bin_dir=run.bin_dir, tasks=str(run.path("tasks")),
                      pilot=str(pilot) if pilot else None, decide_env=decide_env, image=provenance.image(run.image(pass_name)),
                      pass_name="engine-" + pass_name, scanner=run.setting("scanner"), tasks_provenance=json.loads(tasks_record.read_text()) if tasks_record.exists() else None)
    if not planned and not any((run_dir / "shards").glob("*/tasks.jsonl")):
        raise RunError("no shards to drive: the task stage wrote nothing for the chosen repositories")
    return {"planned": planned}


def drive_pass(run, pass_name, pilot):
    preflight.require(run, pass_name)
    fleet.ensure_network(run.factory)
    mounts = [(str(pilot), fleet.PILOT_MOUNT)] if pilot else []
    done, failed = fleet.run(scaled_factory(run), run.path("fleet-" + pass_name), int(run.setting("runners") or run.factory.fleet["runners"]),
                             mounts, run.image(pass_name))
    if not done:
        raise RunError("no shard produced rows (%d failed)" % failed)
    return {"shards_done": done, "shards_failed": failed}


def collect_pass(run, pass_name):
    out, stats = fleet.collect(run.path("fleet-" + pass_name), "engine-" + pass_name, run.path("tasks"))
    if not stats["rows"]:
        raise RunError("collect found no rows")
    if pass_name == "on":
        stats["engine_states"] = engine_states(out)
        if set(stats["engine_states"]) <= {"abstained/engine unavailable"}:
            raise RunError("every engine-on row says the engine was unavailable; the pilot did not load in the runners")
    return {"rows": str(out), "retention": str(out.parent / "retention.jsonl"), "stats": stats}


def engine_states(rows):
    """How the sidecar's engine answered each row, as the rows record it."""
    counts = {}
    for line in Path(rows).read_text(encoding="utf-8").splitlines():
        if line.strip():
            engine = json.loads(line).get("engine") or {}
            key = engine.get("state", "?") + ("/" + engine["reason"] if engine.get("reason") else "")
            counts[key] = counts.get(key, 0) + 1
    return counts


def model_dir(run):
    return build.model_dir(training.backbone(run.factory.root), run.path("build", "model"))


JUDGE_SUFFIXES = ("", ".partial.jsonl", ".manifest.json", ".agreement.jsonl", ".agreement.json", ".provenance.json")


def judge_rows(run, rows_in, out, units=judge.UNITS):
    c = corpus(run)
    scaled = scaled_factory(run)
    manifest = Path(str(out) + ".manifest.json")
    if manifest.exists():
        # A judging that resumes continues under the same signature; one left by an
        # earlier run of the pass, over other rows, is cleared rather than mixed in.
        started = json.loads(manifest.read_text(encoding="utf-8")).get("signature")
        if started != judge.signature(scaled, c, str(rows_in), units, scaled.judge["second_judge_fraction"]):
            for suffix in JUDGE_SUFFIXES:
                Path(str(out) + suffix).unlink(missing_ok=True)
    report = judge.run(scaled, c, str(rows_in), str(out), workers=int(run.setting("workers", 16)), units=units)
    provenance.record(run.factory, str(out) + ".provenance.json", "judge", rows=str(rows_in), units=list(units),
                      corpus_revision=c["catalog_revision"], corpus_sha256=provenance.file_sha256(run.path("corpus.json")))
    Path(str(out) + ".agreement.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if not report["rows"]:
        raise RunError("no rows were judged")
    return {"judged": str(out), "report": report}


def engine_off_or_skip(run):
    if run.engine_on_only:
        return {"skipped": "run.engine_on.heads drives the engine-on pass only"}
    return None


def stage_preflight(run):
    return engine_off_or_skip(run) or preflight.qualify(dataclasses.replace(run, factory=scaled_factory(run)), "off")


def stage_preflight_on(run):
    return engine_on_or_skip(run) or preflight.qualify(dataclasses.replace(run, factory=scaled_factory(run)), "on", run.path("build", "pilot"))


def stage_plan(run):
    return engine_off_or_skip(run) or plan_pass(run, "off", None)


def stage_drive(run):
    return engine_off_or_skip(run) or drive_pass(run, "off", None)


def stage_collect(run):
    return engine_off_or_skip(run) or collect_pass(run, "off")


def stage_judge(run):
    return engine_off_or_skip(run) or judge_or_pass(run, run.path("fleet-off", "rows.jsonl"), run.path("judged-off.jsonl"))


def judge_or_pass(run, rows_in, out):
    """Judge the rows, or with run.judge.enabled off copy them through unjudged, so the
    split and release stages read the same file either way."""
    if judging_off(run):
        shutil.copy2(rows_in, out)
        return {"skipped": "run.judge.enabled is off; rows pass through unjudged", "judged": str(out)}
    return judge_rows(run, rows_in, out)


def stage_judge_skillreq(run):
    if "E4" not in recipes(run) or not run.path("skillreq", "requests.jsonl").exists():
        return {"skipped": "no skill requests were written"}
    if judging_off(run):
        return {"skipped": "run.judge.enabled is off; unjudged skill requests train nothing"}
    return judge_rows(run, run.path("skillreq", "requests.jsonl"), run.path("skillreq", "judged.jsonl"), units=("skills",))


def engine_on_or_skip(run):
    if not run.engine_on:
        return {"skipped": "run.engine_on is off"}
    return None


def split_rows(run, out_dir, judged, retention_rows):
    """The judged rows split into out_dir, and the retention rows split by the same
    groups into out_dir/retention when any pass wrote them."""
    out = {"split": str(out_dir), "counts": split.split(run.factory, [str(p) for p in judged], out_dir), "passes": len(judged)}
    if retention_rows:
        out["retention"] = split.split(run.factory, [str(p) for p in retention_rows], out_dir / "retention")
    return out


def stage_split_pilot(run):
    return engine_on_or_skip(run) or engine_off_or_skip(run) or split_rows(run, run.path("split-pilot"), [run.path("judged-off.jsonl")],
                                                [p for p in (run.path("fleet-off", "retention.jsonl"),) if training.has_rows(p)])


def stage_train_pilot(run):
    return engine_on_or_skip(run) or engine_off_or_skip(run) or train_heads(run, run.path("split-pilot"), run.path("pilot-heads"), "pilot")


def pilot_heads_dir(run):
    """The heads the engine-on runners decide with: released heads when configured, else this run's pilot."""
    heads = run.setting("engine_on", {}).get("heads")
    if not heads:
        return run.path("pilot-heads")
    path = Path(heads)
    if not path.is_dir() or not any(path.glob("*.safetensors")):
        raise RunError("run.engine_on.heads must be a directory of released head .safetensors files: %s" % heads)
    return path


def stage_engine(run):
    """The decision engine for this host, and for the runners when they differ."""
    out = {"host": build.host_engine(run.checkout, run.path("build", "host-engine"), run.cache_root)}
    if run.engine_on and not build.host_is_runner_platform():
        out["runner"] = build.runner_engine(run.checkout, run.path("build", "runner-engine"), run.cache_root)
    return out


def stage_pilot(run):
    if not run.engine_on:
        return {"skipped": "run.engine_on is off"}
    # The pilot retention head is chosen here, once the host engine exists to replay
    # the search on the pilot split; the engine-on pass then drives with it.
    pilot_search = sorted(run.path("pilot-heads", "search").glob("retention-*.safetensors"))
    retained = retention_select(run, pilot_search, run.path("split-pilot"), run.path("pilot-heads"), run.path("eval", "retention-pilot")) if pilot_search else {}
    engine_bin = run.path("build", "runner-engine", "bialy") if not build.host_is_runner_platform() else run.path("build", "host-engine", "bialy")
    report = build.pilot(engine_bin, model_dir(run), pilot_heads_dir(run), run.path("build", "pilot"))
    if retained:
        report["retention"] = {k: retained.get(k) for k in ("chosen", "fitted", "facts")}
    deadline = int(run.setting("engine_on", {}).get("deadline_ms") or 0)
    tag = run.image("on")
    present = subprocess.run(["docker", "image", "inspect", tag], capture_output=True).returncode == 0
    if not present or run.rebuild_image:
        fleet.build_image(run.factory, run.bin_dir, run.engine_dir, tag, deadline, run.checkout / "lycaon/config", scanner=run.scanner,
                          context_window=run.collection_window)
    return dict(report, pilot=str(run.path("build", "pilot")), image=tag)


def stage_plan_on(run):
    return engine_on_or_skip(run) or plan_pass(run, "on", run.path("build", "pilot"))


def stage_drive_on(run):
    return engine_on_or_skip(run) or drive_pass(run, "on", run.path("build", "pilot"))


def stage_collect_on(run):
    return engine_on_or_skip(run) or collect_pass(run, "on")


def stage_judge_on(run):
    return engine_on_or_skip(run) or judge_or_pass(run, run.path("fleet-on", "rows.jsonl"), run.path("judged-on.jsonl"))


def judged_files(run):
    return [p for p in (run.path("judged-off.jsonl"), run.path("judged-on.jsonl")) if p.exists()]


def retention_files(run):
    """Observational retention rows collected by each pass."""
    return [p for p in (run.path("fleet-off", "retention.jsonl"), run.path("fleet-on", "retention.jsonl")) if training.has_rows(p)]


def stage_compare(run):
    """The engine-on pass against the engine-off pass over the requests both drove:
    what the retention head's releases did to prompt tokens, recalls, and repeated
    calls (retention.compare)."""
    if not run.engine_on:
        return {"skipped": "run.engine_on is off"}
    off = retention.load_rows(run.path("fleet-off", "requests.jsonl"))
    on = retention.load_rows(run.path("fleet-on", "requests.jsonl"))
    if not off or not on:
        return {"skipped": "a pass wrote no request rows; the host that drove it predates them"}
    comparison = retention.compare(off, on)
    run.path("compare.json").write_text(json.dumps(comparison, indent=2) + "\n", encoding="utf-8")
    return {"comparison": comparison, "report": str(run.path("compare.json"))}


def stage_split(run):
    # The same seed and groups, so a request's retention row lands beside its turn rows.
    return split_rows(run, run.path("split"), judged_files(run), retention_files(run))


def stage_custom_model(run):
    return custom_model_run.collect(run, judge_factory=scaled_factory(run))


# ---- code-rank pairs -----------------------------------------------------------

def stage_coderank(run):
    """Units, request pairs, and site dumps for the code-rank head, through this host's
    decide-rerank build and the writer model."""
    if "code-rank" not in (run.setting("train", {}).get("recipes") or []):
        return {"skipped": "code-rank is not in run.train.recipes"}
    scaled = scaled_factory(run)
    if not scaled.repos:
        return {"skipped": "code-rank pairs come from repositories' code, and this run names none"}
    binary = str(run.host_bin_dir / "decide-rerank")
    base = run.path("coderank")
    per_repo = int(run.setting("coderank", {}).get("per_repo") or 40)
    if run.setting("task_cap"):
        per_repo = min(per_repo, 4)
    units = coderank.harvest(scaled, binary, base / "units")
    provenance.record(run.factory, base / "units.provenance.json", "coderank-harvest", bin_dir=run.host_bin_dir, counts=units,
                      decide_rerank=provenance.file_sha256(binary))
    pairs = coderank.pairs(scaled, base / "units", base / "pairs", per_repo, seed=int(run.setting("seed", 7)), writer=writer_model(run))
    provenance.record(run.factory, base / "pairs.provenance.json", "coderank-pairs", bin_dir=run.host_bin_dir, counts=pairs, per_repo=per_repo,
                      seed=int(run.setting("seed", 7)), decide_rerank=provenance.file_sha256(binary))
    coderank.dumps(scaled, binary, base / "units", base / "pairs", base / "dumps")
    dumps = sorted(p.name for p in (base / "dumps").glob("*.jsonl"))
    if not dumps:
        raise RunError("decide-rerank wrote no dumps")
    return {"units": units, "pairs": pairs, "dumps": len(dumps)}


# ---- releases ------------------------------------------------------------------

def stage_release(run):
    version = run.setting("dataset_version") or run.name
    out = run.path("dist", "dataset-" + version)
    stages = [p for p in (run.path("tasks", "provenance.json"), run.path("skillreq", "requests.jsonl.provenance.json"),
                          run.path("fleet-off", "provenance.json"), run.path("fleet-on", "provenance.json"),
                          run.path("judged-off.jsonl.provenance.json"), run.path("judged-on.jsonl.provenance.json"),
                          run.path("coderank", "units.provenance.json"), run.path("coderank", "pairs.provenance.json")) if p.exists()]
    driven = [p for p in (run.path("fleet-off", "tasks-driven.jsonl"), run.path("fleet-on", "tasks-driven.jsonl")) if p.exists()]
    if not driven:
        raise RunError("no tasks-driven.jsonl; the release names every task the run drove")
    agreement = [str(p) + ".agreement.jsonl" for p in judged_files(run) if Path(str(p) + ".agreement.jsonl").exists()]
    pairs = run.path("coderank", "pairs")
    retention = run.path("split", "retention")
    release.build(run.factory, run.path("split"), version, out, run.checkout / "scripts/bialy/row.schema.json", run.path("corpus.json"),
                  run.setting("code_ref", "HEAD"), agreement=agreement, stages=[str(p) for p in stages], driven=[str(p) for p in driven],
                  coderank=str(pairs) if pairs.exists() else None, stopping=run.setting("stopping", ""),
                  retention=retention if retention.exists() else None, retention_schema=run.checkout / "scripts/bialy/retention-row.schema.json")
    return {"release": str(out), "version": version, "anchor": str(anchor.path("dataset", version))}


def train_heads(run, split_dir, heads_dir, label):
    """Every recipe in run.train.recipes over `split_dir`, into `heads_dir`, within the budget."""
    settings = dict(run.setting("train", {}), retention=run.setting("retention", {}))
    recipes = settings.get("recipes") or []
    if not recipes:
        return {"skipped": "run.train.recipes is empty"}
    judged_skillreq = run.path("skillreq", "judged.jsonl")
    inputs = {"corpus": run.path("corpus.json"),
              "independent_corpus": training.independent_corpus(run.checkout, run.path("corpus.json"), run.path("independent-corpus.json")),
              "train": split_dir / "train.jsonl", "val": split_dir / "val.jsonl",
              "skillreq_train": judged_skillreq if judged_skillreq.exists() else None,
              "skillreq_eval": training.skillreq_eval(judged_skillreq, run.path("skillreq", "eval.jsonl")) if judged_skillreq.exists() else None,
              "dumps": run.path("coderank", "dumps") if run.path("coderank", "dumps").exists() else None,
              "retention_train": split_dir / "retention" / "train.jsonl", "retention_val": split_dir / "retention" / "val.jsonl"}
    budget = float(settings.get("max_hours", 24)) * 3600
    started = time.time()
    trained = {}
    for recipe in recipes:
        remaining = budget - (time.time() - started)
        if remaining <= 0:
            raise RunError("training budget of %s hours spent before %s" % (settings.get("max_hours", 24), recipe))
        try:
            trained[recipe] = training.train(recipe, inputs, run.checkout, run.factory.root, settings, heads_dir,
                                             "%s-%s-%s" % (run.name, label, recipe), remaining)
        except training.TrainingError as exc:
            raise RunError(str(exc)) from exc
    return {"heads": trained, "heads_dir": str(heads_dir)}


def stage_train(run):
    return train_heads(run, run.path("split"), run.path("heads"), "final")


def stage_evaluate(run):
    heads_dir = run.path("heads")
    search = sorted((heads_dir / "search").glob("retention-*.safetensors")) if heads_dir.exists() else []
    # The retention head is chosen here, from the search, before the launcher that
    # carries every final head is written.
    retained = retention_select(run, search, run.path("split"), heads_dir, run.path("eval", "retention")) if search else \
        {"viability": {"viable": False, "reasons": ["no retention head was trained"], "numbers": {}}}
    trained = {p.stem: str(p) for p in heads_dir.glob("*.safetensors")} if heads_dir.exists() else {}
    if not trained:
        return {"skipped": "no heads were trained", "retention": retained}
    launcher = build.launcher(run.path("build", "host-engine", "bialy"), model_dir(run), trained, run.path("build", "launcher.sh"))
    limit = int(run.setting("evaluate", {}).get("rows") or 0)
    evals = {}
    # The turn replay asks the turn-load and unit-rank heads; without both it has nothing to score.
    turn_heads = {"turn-load", "unit-rank"} <= set(trained)
    for name in ("val", "holdout") if turn_heads else ():
        out = run.path("eval", "%s.json" % name)
        out.parent.mkdir(parents=True, exist_ok=True)
        rows, total = sample_rows(run.path("split", "%s.jsonl" % name), run.path("eval", "%s.sample.jsonl" % name), limit,
                                  "%s:evaluate:%s" % (run.setting("seed", 7), name))
        args = [sys.executable, "scripts/bialy/replay_eval.py", "--corpus", str(run.path("corpus.json")), "--examples", str(rows),
                "--engine", str(launcher), "--tool-truth", "consensus", "--json", str(out)]
        result = subprocess.run(args, cwd=str(run.checkout), capture_output=True, text=True, env=training.environment(run.factory.root, run.setting("train", {})))
        if result.returncode != 0:
            raise RunError("replay on %s failed: %s" % (name, result.stderr.strip()[-400:]))
        evals[name] = {"report": str(out), "rows": min(total, limit) if limit else total, "of": total}
    if not turn_heads:
        evals["skipped"] = "turn-load and unit-rank were not both trained; the turn replay has nothing to score"
    rerank = rerank_evals(run, trained) if "code-rank" in trained else {}
    return {"evals": evals, "launcher": str(launcher), "rerank": rerank, "retention": retained}


def retention_select(run, search, split_dir, heads_dir, out_dir):
    """Every searched retention head replayed on split_dir's validation rows with its
    policy fitted there, the best fitted objective promoted to heads_dir/retention.safetensors,
    that head scored on the held-out rows with an interval, and the verdict read from
    the held-out score and the pass comparison. Everything the choice rests on is
    written under out_dir."""
    settings = run.setting("retention", {})
    val, holdout = split_dir / "retention" / "val.jsonl", split_dir / "retention" / "holdout.jsonl"
    if not training.has_rows(val):
        return {"viability": {"viable": False, "reasons": ["no validation retention rows to fit on"], "numbers": {}}}
    out_dir.mkdir(parents=True, exist_ok=True)
    env = training.environment(run.factory.root, run.setting("train", {}))
    reports = {}
    for head in search:
        name = head.stem[len("retention-"):]
        launcher = build.launcher(run.path("build", "host-engine", "bialy"), model_dir(run), {"retention": str(head)}, out_dir / ("%s.launcher.sh" % name))
        out = out_dir / ("%s.json" % name)
        args = [sys.executable, "scripts/bialy/retention_eval.py", "--corpus", str(run.path("corpus.json")), "--examples", str(val), "--fit",
                "--bootstrap", str(int(settings.get("bootstrap") or 500)), "--seed", str(run.setting("seed", 7)), "--json", str(out),
                "--dump", str(out_dir / ("%s.candidates.jsonl" % name)), "--engine", str(launcher)]
        if training.has_rows(holdout):
            args += ["--holdout", str(holdout)]
        result = subprocess.run(args, cwd=str(run.checkout), capture_output=True, text=True, env=env)
        if result.returncode != 0:
            raise RunError("retention replay of %s failed: %s" % (name, result.stderr.strip()[-400:]))
        reports[name] = json.loads(out.read_text(encoding="utf-8"))
    chosen, objective = retention.select(reports)
    report = reports[chosen]
    shutil.copy2(heads_dir / "search" / ("retention-%s.safetensors" % chosen), heads_dir / "retention.safetensors")
    selection = {"chosen": chosen, "validation_objective_tokens": objective, "fitted": {k: report["fitted"][k] for k in ("drop_below", "confidence_floor")},
                 "recommendation": report.get("recommendation"),
                 "search": {name: {"validation_objective_tokens": ((r.get("fitted") or {}).get("tally") or {}).get("objective_tokens"),
                                   "auc": (r.get("overall") or {}).get("auc"),
                                   "holdout_objective_per_request": ((r.get("holdout") or {}).get("engine") or {}).get("objective_per_request"),
                                   "holdout_interval": ((r.get("holdout") or {}).get("bootstrap") or {}).get("engine")}
                            for name, r in sorted(reports.items())}}
    (out_dir / "selection.json").write_text(json.dumps(selection, indent=2) + "\n", encoding="utf-8")
    (out_dir / "decisions-fragment.yaml").write_text((report.get("recommendation") or {}).get("fragment", "") + "\n", encoding="utf-8")
    comparison = (run.outputs("compare") or {}).get("comparison")
    return {"chosen": chosen, "report": str(out_dir / ("%s.json" % chosen)), "selection": str(out_dir / "selection.json"),
            "fragment": str(out_dir / "decisions-fragment.yaml"), "fitted": selection["fitted"], "facts": (report.get("recommendation") or {}).get("facts"),
            "viability": retention.viability(report, comparison, settings)}


def sample_rows(rows, out, limit, seed):
    """The rows a replay measures: all of them, or `limit` drawn with `seed`, kept in file
    order. Returns the file to replay and how many rows the split holds."""
    lines = [line for line in Path(rows).read_text(encoding="utf-8").splitlines() if line.strip()]
    if not limit or len(lines) <= limit:
        return rows, len(lines)
    keep = sorted(random.Random(seed).sample(range(len(lines)), limit))
    Path(out).write_text("".join(lines[i] + "\n" for i in keep), encoding="utf-8")
    return out, len(lines)


RERANK_WORKERS = 4


def rerank_evals(run, trained):
    """The code-rank head measured on the held-out repositories: every site's text-match
    order against the blend, on the human-written doc pairs, through the host engine."""
    env = dict(os.environ, LYCAON_DECIDE_BINARY=str(run.path("build", "host-engine", "bialy")), LYCAON_DECIDE_MODEL_DIR=str(model_dir(run)),
               LYCAON_DECIDE_HEADS="code-rank=" + trained["code-rank"], HF_HUB_OFFLINE="1")
    binary = str(run.host_bin_dir / "decide-rerank")
    jobs = []
    for repo in scaled_factory(run).repos:
        units = run.path("coderank", "units", repo.name + ".jsonl")
        pairs = run.path("coderank", "pairs", "docs-%s.jsonl" % repo.name)
        if repo.split != "holdout" or not units.exists() or not pairs.exists():
            continue
        jobs += [(site, repo, units, pairs) for site in coderank.SITES]

    def evaluate(job):
        site, repo, units, pairs = job
        out = run.path("eval", "rerank-%s-%s.json" % (site, repo.name))
        result = subprocess.run([binary, "eval", "--site", site, "--repo", str(run.factory.root / "repos" / repo.name), "--name", repo.name,
                                 "--units", str(units), "--pairs", str(pairs), "--json", str(out)], env=env, capture_output=True, text=True)
        if result.returncode != 0:
            raise RunError("decide-rerank eval %s on %s failed: %s" % (site, repo.name, (result.stderr or result.stdout).strip()[-300:]))
        return "%s-%s" % (site, repo.name), str(out)

    # Each evaluation runs its own engine on a few CPU threads, so the sites of the
    # held-out repositories run side by side rather than one after another.
    with concurrent.futures.ThreadPoolExecutor(max_workers=RERANK_WORKERS) as pool:
        return dict(pool.map(evaluate, jobs))


def stage_release_heads(run):
    heads_dir = run.path("heads")
    if not heads_dir.exists() or not any(heads_dir.glob("*.safetensors")):
        return {"skipped": "no heads were trained"}
    version = run.setting("heads_version") or run.name
    dataset_version = run.setting("dataset_version") or run.name
    evaluate = run.outputs("evaluate")
    evals = {name: e["report"] for name, e in (evaluate.get("evals") or {}).items() if isinstance(e, dict) and e.get("report")}
    rerank = evaluate.get("rerank") or {}
    retained = evaluate.get("retention") or {}
    notes = {name: retained[key] for name, key in (("retention", "report"), ("retention-selection", "selection")) if retained.get(key)}
    if run.path("compare.json").exists():
        notes["retention-comparison"] = str(run.path("compare.json"))
    commit = (run.outputs("build").get("runner") or {}).get("lycaon_commit") or build.checkout_commit(run.checkout) or "unknown"
    out = run.path("dist", "heads-" + version)
    heads.build(heads_dir, version, dataset_version, commit, evals, {}, "base checkpoint", rerank, out, notes=notes,
                retention=retained.get("report"), verdict=retained.get("viability"))
    return {"release": str(out), "version": version, "anchor": str(anchor.path("heads", version))}


# ---- accounting and report -----------------------------------------------------

def usage_summary(run):
    totals = {}
    for path in (run.path("usage.jsonl"), run.path("requests.jsonl"), run.path("fleet-off", "requests.jsonl"), run.path("fleet-on", "requests.jsonl")):
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                entry = json.loads(line)
            except ValueError:
                continue
            usage = entry.get("usage") or {}
            prompt_toks = int(usage.get("prompt_tokens") if "prompt_tokens" in usage else (entry.get("input_tokens") or 0))
            comp_toks = int(usage.get("completion_tokens") if "completion_tokens" in usage else (entry.get("output_tokens") or 0))
            t = totals.setdefault(entry.get("model", "?"), {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0})
            t["calls"] += 1
            t["prompt_tokens"] += prompt_toks
            t["completion_tokens"] += comp_toks
    return totals


def estimated_spend(run, totals):
    """Dollars, when models declare hosted.price_per_million {input, output}."""
    spend = 0.0
    priced = False
    for model in run.factory.models:
        price = (model.hosted or {}).get("price_per_million")
        key = (model.hosted or {}).get("model", model.id)
        t = totals.get(key)
        if price and t:
            priced = True
            spend += t["prompt_tokens"] / 1e6 * float(price.get("input", 0)) + t["completion_tokens"] / 1e6 * float(price.get("output", 0))
    return round(spend, 2) if priced else None


def check_spend(run):
    ceiling = float(run.setting("spend_ceiling_usd") or 0)
    if ceiling <= 0:
        return
    spend = estimated_spend(run, usage_summary(run))
    if spend is not None and spend > ceiling:
        raise RunError("estimated spend $%.2f exceeds run.spend_ceiling_usd %.2f" % (spend, ceiling))


def stage_report(run, publish=True):
    state = run.load_state()
    totals = usage_summary(run)
    lines = ["# Run %s" % run.name, "", "| Stage | Status | Seconds | Outputs |", "|---|---|---|---|"]
    for name in STAGES:
        s = state["stages"].get(name)
        if not s:
            continue
        outputs = s.get("outputs") or {}
        summary = "; ".join("%s=%s" % (k, v if not isinstance(v, (dict, list)) else json.dumps(v)[:80]) for k, v in outputs.items())
        if s.get("error"):
            summary = "error: " + s["error"]
        lines.append("| %s | %s | %s | %s |" % (name, s.get("status"), s.get("seconds", ""), summary.replace("|", "\\|")))
    lines += ["", "## Provider usage", ""]
    if totals:
        lines += ["| Model | Calls | Prompt tokens | Completion tokens |", "|---|---|---|---|"]
        for model, t in sorted(totals.items()):
            lines.append("| %s | %d | %d | %d |" % (model, t["calls"], t["prompt_tokens"], t["completion_tokens"]))
        spend = estimated_spend(run, totals)
        if spend is not None:
            lines.append("\nEstimated spend: $%.2f" % spend)
        lines.append("\nIncludes collected runner sidecar usage; live calls can exceed the ceiling before the next stage check.")
    else:
        lines.append("No provider calls were recorded.")
    lines += retention_report(run)
    dataset = run.outputs("release")
    head_release = run.outputs("release_heads")
    releases = {kind: rel["release"] for kind, rel in (("dataset", dataset), ("heads", head_release)) if rel.get("release")}
    lines += ["", "## Publishing", ""]
    published = {}
    if releases and run.push and publish:
        version = dataset.get("version") or head_release.get("version")
        published = publishing.finish(version, releases, push=True)
        for kind, result in sorted(published.items()):
            lines.append("- %s: %s" % (kind, json.dumps(result)))
    for kind, rel in (("dataset", dataset), ("heads", head_release)):
        if rel.get("release") and kind not in published:
            lines.append("- `bialy publish %s --release %s --version %s --push`" % (kind, rel["release"], rel["version"]))
    if not releases:
        lines.append("Nothing to publish.")
    run.path("REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"report": str(run.path("REPORT.md")), "published": published}


def retention_report(run):
    """The report's retention section: the verdict and every reason, the fitted policy
    with its held-out interval, each structural fact priced, the pass comparison, and
    the search table, so the choice can be read without opening the JSON."""
    retained = run.outputs("evaluate").get("retention") or {}
    verdict = retained.get("viability")
    if verdict is None:
        return []
    lines = ["", "## Retention head", "", "**%s.**" % ("Viable" if verdict.get("viable") else "Not viable")]
    lines += ["- " + reason for reason in verdict.get("reasons") or []]
    numbers = verdict.get("numbers") or {}
    if retained.get("chosen"):
        lines += ["", "Chosen `%s`; fitted policy %s; structural facts to release on their own: %s." % (
            retained["chosen"], json.dumps(retained.get("fitted")), json.dumps(retained.get("facts")))]
    if numbers:
        lines += ["", "| Held-out measure | Value |", "|---|---|"]
        lines += ["| %s | %s |" % (k, json.dumps(v)) for k, v in sorted(numbers.items()) if k != "comparison"]
    report = json.loads(Path(retained["report"]).read_text(encoding="utf-8")) if retained.get("report") and Path(retained["report"]).exists() else {}
    structural = ((report.get("holdout") or report.get("overall") or {}).get("structural")) or {}
    if structural:
        lines += ["", "| Structural facts | Released | Needed among them | Objective per request |", "|---|---|---|---|"]
        lines += ["| %s | %s | %s | %s |" % (name, t.get("released"), t.get("needed_released"), t.get("objective_per_request")) for name, t in structural.items()]
    comparison = numbers.get("comparison") or {}
    if comparison.get("measures"):
        lines += ["", "Engine-off against engine-on over %d matched requests (%d with a release):" % (comparison["requests"], comparison.get("released_requests", 0)),
                  "", "| Per request | Off | On | Delta | Wants |", "|---|---|---|---|---|"]
        lines += ["| %s | %s | %s | %s | %s |" % (m, v["off"], v["on"], v["delta"], v["wants"]) for m, v in comparison["measures"].items()]
    selection = json.loads(Path(retained["selection"]).read_text(encoding="utf-8")) if retained.get("selection") and Path(retained["selection"]).exists() else {}
    if selection.get("search"):
        lines += ["", "| Search point | Validation objective | AUC | Held-out objective per request | Held-out interval |", "|---|---|---|---|---|"]
        lines += ["| %s | %s | %s | %s | %s |" % (name, s.get("validation_objective_tokens"), s.get("auc"), s.get("holdout_objective_per_request"), json.dumps(s.get("holdout_interval")))
                  for name, s in selection["search"].items()]
    counts = run.outputs("split").get("retention") or {}
    if counts:
        lines.append("\nRetention rows by split: %s" % json.dumps({k: v.get("rows") for k, v in counts.items()}))
    return lines


STAGE_FUNCS = {name: globals()["stage_" + name] for name in STAGES}


def plan(run, start=None, until=None, redo=()):
    """The stages this invocation will run, in order: those not done, those from `start`
    on, and those named in `redo`."""
    state = run.load_state()
    names = list(STAGES)
    if until:
        names = names[:names.index(until) + 1]
    out = []
    for name in names:
        done = state["stages"].get(name, {}).get("status") == "done"
        if (start and STAGES.index(name) >= STAGES.index(start)) or name in redo:
            done = False
        out.append((name, "skip" if done else "run"))
    return out


def execute(run, start=None, until=None, funcs=None, redo=()):
    funcs = funcs or STAGE_FUNCS
    if run.setting("custom_model", {}).get("enabled"):
        try:
            custom_model_run.binding(run)
        except custom_model_run.DatasetError as exc:
            raise RunError(str(exc)) from exc
    os.environ["BIALY_USAGE_JSONL"] = str(run.path("usage.jsonl"))
    state = run.load_state()
    for name in (STAGES[STAGES.index(start):] if start else ()) + tuple(redo):
        state["stages"].pop(name, None)
    run.save_state(state)
    for name, action in plan(run, start, until, redo):
        if action == "skip":
            continue
        started = time.time()
        print("%s: %s" % (run.name, name), flush=True)
        try:
            check_spend(run)
            outputs = funcs[name](run)
            state = run.load_state()
            status = "not_ready" if outputs and outputs.get("status") == "not_ready" else "done"
            state["stages"][name] = {"status": status, "at": stamp(), "seconds": round(time.time() - started), "outputs": outputs}
            run.save_state(state)
            if status == "not_ready":
                stage_report(run, publish=False)
                return run.load_state()
            if outputs and outputs.get("skipped"):
                print("%s: %s skipped: %s" % (run.name, name, outputs["skipped"]), flush=True)
        except Exception as exc:  # any stage failure stops the run where it stands
            state = run.load_state()
            state["stages"][name] = {"status": "failed", "at": stamp(), "seconds": round(time.time() - started), "error": str(exc)}
            run.save_state(state)
            if name != "report":
                try:
                    stage_report(run, publish=False)
                except Exception:
                    pass
            raise RunError("stage %s failed: %s" % (name, exc)) from exc
    # Every invocation leaves a current report, even one that stops early; only the
    # report stage itself publishes.
    if until and until != "report":
        stage_report(run, publish=False)
    return run.load_state()


def settings_for(factory, overrides):
    settings = json.loads(json.dumps(factory.run))
    for key, value in overrides.items():
        if value is None:
            continue
        if "." in key:
            section, sub = key.split(".", 1)
            settings[section][sub] = value
        else:
            settings[key] = value
    return settings
