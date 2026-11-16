"""One command on one machine: `bialy run` drives a pass from repositories to a
head release, resumable, with nothing uploaded unless asked.

Every stage writes its outputs under <root>/runs/<name>/ and its status to
run.json there. A rerun skips finished stages; `--from` reruns from a stage,
`--until` stops after one. Settings come from factory.yaml's `run` section and
the flags that override it for one run.

The pass has two halves. The first drives every task with the decision engine
off, and the selector fits a pilot policy on what the judges labelled. The
second drives the same tasks again with that policy deciding, so the release
also holds rows from turns the engine shaped. The selector then fits the final
policy over the rows of both: the rank families train on every row, and the
turn-load head on the rows whose turn the engine did not answer, since a tool
the engine preloaded and the session then called is not an independent label.
A heads release is cut only from a policy the selector qualified.
"""
import dataclasses
import json
import os
import platform
import random
import shutil
import subprocess
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
    host_contract,
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
from .campaign import ancestry, selector
from .campaign.journal import CampaignFault

STAGES = ("check", "repos", "build", "image", "corpus", "tasks", "ancestry", "skillreq", "warm", "preflight", "plan", "drive", "collect", "judge",
          "judge_skillreq", "coderank", "engine", "split_pilot", "train_pilot", "pilot", "preflight_on", "plan_on", "drive_on", "collect_on",
          "judge_on", "split", "release", "train", "evaluate", "release_heads", "report")
ENGINE_ON_STAGES = ("split_pilot", "train_pilot", "pilot", "preflight_on", "plan_on", "drive_on", "collect_on", "judge_on")


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
        return custom_model_run.context_window(custom) if custom.get("enabled") else None

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
    """The models with the roles this run gives them: run.generators and run.judges give
    those roles to the models they name and take them off the rest; a model keeps its
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
    scenarios = [s for s in f.scenarios if (not chosen or s.archetype in chosen) and (not names or s.workspace in names)]
    variants = max(1, round(f.scenario_variants * float(share))) if share else f.scenario_variants
    variants = min(variants, int(cap)) if cap else variants
    return dataclasses.replace(f, models=models, archetypes=archetypes, repos=[r for r in f.repos if not names or r.name in names],
                               stacks=[s for s in f.stacks if not names or s.name in names], scenarios=scenarios,
                               scenario_variants=variants, run=run.settings)


# ---- check --------------------------------------------------------------------

CHECKOUT_FILES = ("lycaon/cmd/lycaon", "lycaon/cmd/lycaon-debug", "lycaon/cmd/decide-rerank", "lycaon/cmd/opengrep-artifact",
                  "lycaon/internal/decide/native/Cargo.toml", "lycaon/config/gitengine/pin.yaml", "rust-toolchain.toml", "schemas",
                  "scripts/decide-features.sh", "scripts/build-decide.sh", "scripts/bialy/row.schema.json",
                  "scripts/bialy/retention-row.schema.json", "lycaon/config/" + build.RELEASE)


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
    reviewer_slots = run.setting("judges") or [model.id for model in scaled.models if "judge" in model.roles]
    for slot in range(max(1, len(reviewer_slots))):
        try:
            used.append(scaled.reviewer(slot))
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
    try:
        host_contract.verify(run.checkout)
    except host_contract.HostContractError as exc:
        faults.append("run.lycaon_checkout does not declare what the runner asks of it: %s" % "; ".join(exc.missing))
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
    policy_recipes = set(selector.load()["heads"].values())
    if policy_recipes & set(recipes) and not policy_recipes <= set(recipes):
        faults.append("run.train.recipes names some of %s; a selector policy is whole or absent, so name all of them or none"
                      % ", ".join(sorted(policy_recipes)))
    ceiling = float(run.setting("spend_ceiling_usd") or 0)
    priced = [m.id for m in used if (m.hosted or {}).get("price_per_million")]
    if ceiling > 0 and not priced:
        faults.append("run.spend_ceiling_usd needs hosted.price_per_million on at least one model it would count")
    if faults:
        raise RunError("; ".join(faults))
    return {"generators": [m.id for m in generators], "judges": reviewer_slots,
            "writer": writer_model(run).id, "repositories": len(scaled.repos), "stacks": len(scaled.stacks),
            "runners": int(run.setting("runners") or f.fleet["runners"]), "engine_on": run.engine_on, "recipes": recipes,
            "scanner": run.setting("scanner") if scanner else "none",
            "accelerator": training.accelerator(), "go": "host" if shutil.which("go") else build.go_image(run.checkout),
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
    """corpus.json from the pinned checkout's lycaon-debug, so option texts match the rows."""
    out = run.path("corpus.json")
    debug = run.host_bin_dir / "lycaon-debug"
    result = subprocess.run([str(debug), *host_contract.arguments("lycaon-debug", "decide corpus", out=out)], capture_output=True, text=True)
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
    tool and skill recipes need."""
    return not run.setting("judge", {}).get("enabled", True)


def skillreq_plan(run):
    """The skill-request families this run writes, or None when no recipe needs them."""
    if "E4" not in recipes(run):
        return None
    settings = run.setting("skillreq", {})
    families = {k: int(v) for k, v in settings["families"].items()}
    cap = run.setting("task_cap")
    if cap:
        families = {k: min(v, 1) for k, v in families.items()}
    none_families = min(int(settings["none_families"]), 2) if cap else int(settings["none_families"])
    return families, none_families


def stage_ancestry(run):
    """Every task and skill-request family the run will generate, joined by shared
    ancestry and allocated to its partitions before anything is driven or written."""
    planned = skillreq_plan(run)
    families = skillreq.family_nodes(scaled_factory(run), corpus(run), *planned, int(run.setting("seed", 7)),
                                     workspaces=tuple(run.setting("workspaces") or ())) if planned else []
    assignment = split.assign(run.factory, split.read_tasks(run.path("tasks")), families)
    assignment.save(run.path("ancestry.json"))
    return {"ancestry": str(run.path("ancestry.json")), "sha256": assignment.sha256, "components": assignment.counts()}


def ancestry_of(run):
    return ancestry.load(run.path("ancestry.json"))


def stage_skillreq(run):
    """Requests for every skill, written by the writer model; the judges label them later."""
    planned = skillreq_plan(run)
    if planned is None:
        return {"skipped": "E4 is not in run.train.recipes; skill requests are only for the unit-rank head"}
    families, none_families = planned
    settings = run.setting("skillreq", {})
    out = run.path("skillreq", "requests.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    writer = writer_model(run)
    report = skillreq.write(scaled_factory(run), corpus(run), out, writer.id, families, none_families, int(settings["per_family"]),
                            seed=int(run.setting("seed", 7)), workers=int(run.setting("workers", 16)),
                            workspaces=tuple(run.setting("workspaces") or ()), assignment=ancestry_of(run))
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
        decide_env.update(PW_TRAINING_CAPTURE_DIR="/shard/training-captures", PW_TRAINING_EPOCH=custom["epoch"],
                          PW_TRAINING_STUDENTS=json.dumps([{"provider": serving["provider"], "model": serving["model"]}
                                                         for serving in custom["students"].values()]))
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


JUDGE_SUFFIXES = ("", ".partial.jsonl", ".manifest.json", ".qualification.json", ".agreement.jsonl", ".agreement.json", ".provenance.json")


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
    return engine_on_or_skip(run) or preflight.qualify(dataclasses.replace(run, factory=scaled_factory(run)), "on", pilot_mount(run))


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
    """The judged rows split into out_dir by the run's ancestry, and the retention rows
    split the same way into out_dir/retention when any pass wrote them."""
    assignment = ancestry_of(run)
    out = {"split": str(out_dir), "ancestry": assignment.sha256, "counts": split.split(assignment, [str(p) for p in judged], out_dir),
           "passes": len(judged)}
    if retention_rows:
        out["retention"] = split.split(assignment, [str(p) for p in retention_rows], out_dir / "retention")
    return out


def stage_split_pilot(run):
    return engine_on_or_skip(run) or engine_off_or_skip(run) or split_rows(run, run.path("split-pilot"), [run.path("judged-off.jsonl")],
                                                [p for p in (run.path("fleet-off", "retention.jsonl"),) if training.has_rows(p)])


def stage_train_pilot(run):
    return engine_on_or_skip(run) or engine_off_or_skip(run) or fit_heads(run, run.path("split-pilot"), run.path("pilot-policy"), "pilot")


def released_heads(run):
    """The released head files run.engine_on.heads names. They must be the heads the
    checkout's decision release pins, because the runners serve that release's option
    vocabulary and thresholds from the checkout's own config."""
    directory = Path(run.setting("engine_on", {}).get("heads"))
    release = json.loads((run.checkout / "lycaon/config" / build.RELEASE).read_text(encoding="utf-8"))
    found = {name: directory / (name + ".safetensors") for name in release["heads"]}
    wrong = sorted(name for name, path in found.items() if not path.is_file() or provenance.file_sha256(path) != release["heads"][name]["sha256"])
    if wrong:
        raise RunError("run.engine_on.heads must hold the heads the checkout's decision release %s pins; %s differ or are missing in %s"
                       % (release.get("release"), ", ".join(wrong), directory))
    return found


def stage_engine(run):
    """The decision engine for this host, and the CPU engine a pilot carries into the runners."""
    out = {"host": build.host_engine(run.checkout, run.path("build", "host-engine"), run.cache_root)}
    if run.engine_on or trains_policy(run):
        out["runner"] = build.runner_engine(run.checkout, run.path("build", "runner-engine"), run.cache_root)
    return out


def stage_pilot(run):
    """The mount the engine-on runners decide from: the pilot policy with its thresholds
    and option vocabulary, or released heads under the checkout's own release."""
    if not run.engine_on:
        return {"skipped": "run.engine_on is off"}
    policy = run.outputs("train_pilot").get("selector")
    if run.engine_on_only:
        mount = run.path("build", "pilot")
        report = build.pilot(run.path("build", "runner-engine", "bialy"), model_dir(run), released_heads(run), mount)
    elif policy:
        mount = Path(policy["pilot"])
        report = {"heads": sorted(policy["heads"]), "policy": policy["policy"]["identity"]}
    else:
        raise RunError("the engine-on pass has no policy to serve: name the selector's recipes in run.train.recipes, "
                       "give released heads, or turn run.engine_on off")
    deadline = int(run.setting("engine_on", {}).get("deadline_ms") or 0)
    tag = run.image("on")
    present = subprocess.run(["docker", "image", "inspect", tag], capture_output=True).returncode == 0
    if not present or run.rebuild_image:
        fleet.build_image(run.factory, run.bin_dir, run.engine_dir, tag, deadline, run.checkout / "lycaon/config", scanner=run.scanner,
                          context_window=run.collection_window)
    return dict(report, pilot=str(mount), image=tag)


def pilot_mount(run):
    return Path(run.outputs("pilot")["pilot"])


def stage_plan_on(run):
    return engine_on_or_skip(run) or plan_pass(run, "on", pilot_mount(run))


def stage_drive_on(run):
    return engine_on_or_skip(run) or drive_pass(run, "on", pilot_mount(run))


def stage_collect_on(run):
    return engine_on_or_skip(run) or collect_pass(run, "on")


def stage_judge_on(run):
    return engine_on_or_skip(run) or judge_or_pass(run, run.path("fleet-on", "rows.jsonl"), run.path("judged-on.jsonl"))


def judged_files(run):
    return [p for p in (run.path("judged-off.jsonl"), run.path("judged-on.jsonl")) if p.exists()]


def retention_files(run):
    """Observational retention rows collected by each pass."""
    return [p for p in (run.path("fleet-off", "retention.jsonl"), run.path("fleet-on", "retention.jsonl")) if training.has_rows(p)]


def stage_split(run):
    return split_rows(run, run.path("split"), judged_files(run), retention_files(run))


# ---- code-rank pairs -----------------------------------------------------------

def stage_coderank(run):
    """Units and request pairs for every repository, and site dumps for the repositories
    the ancestry puts in train and validation, through this host's decide-rerank build
    and the writer model."""
    if "code-rank" not in recipes(run):
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
    dumps = coderank.dumps(scaled, binary, base / "units", base / "pairs", base / "dumps", ancestry_of(run))
    return {"units": units, "pairs": pairs, "dumps": {partition: len(paths) for partition, paths in dumps.items()}}


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
                  run.setting("code_ref", "HEAD"), run.path("ancestry.json"), agreement=agreement, stages=[str(p) for p in stages], driven=[str(p) for p in driven],
                  coderank=str(pairs) if pairs.exists() else None, stopping=run.setting("stopping", ""),
                  retention=retention if retention.exists() else None, retention_schema=run.checkout / "scripts/bialy/retention-row.schema.json")
    return {"release": str(out), "version": version, "anchor": str(anchor.path("dataset", version))}


def trains_policy(run):
    """Whether this run fits a selector policy: its recipes name the policy's three heads."""
    return set(selector.load()["heads"].values()) <= set(recipes(run))


def selector_fault(exc):
    return RunError("%s: %s" % (exc.code, json.dumps(exc.details)))


def fit_heads(run, split_dir, directory, label):
    """The selector policy over `split_dir`, frozen under `directory`: three heads,
    calibrated thresholds, and the runner mount that serves them."""
    if not trains_policy(run):
        return {"skipped": "run.train.recipes names no selector policy"}
    try:
        return {"selector": selector.fit(run, split_dir, directory, selector.Workbench(run), selector.load(), label)}
    except CampaignFault as exc:
        raise selector_fault(exc) from exc
    except training.TrainingError as exc:
        raise RunError(str(exc)) from exc


def stage_train(run):
    return fit_heads(run, run.path("split"), run.path("selector"), "final")


def stage_evaluate(run):
    """Qualify the final policy on the development split and compare it once on the
    held-out split. A policy short of the parity probe or a bar fails the stage, and
    no heads release follows."""
    directory = run.path("selector")
    trained = run.outputs("train").get("selector")
    if not trained:
        return {"skipped": "this run fitted no selector policy"}
    try:
        record = selector.qualify(run, run.path("split"), directory, trained, selector.Workbench(run), selector.load())
        comparison = selector.holdout(run, run.path("split"), directory, trained, selector.Workbench(run))
    except CampaignFault as exc:
        raise selector_fault(exc) from exc
    return {"policy": record["policy"], "qualification": str(directory / "qualification.json"), "measures": record["measures"],
            "parity": record["parity"], "holdout": comparison}


def stage_release_heads(run):
    if not run.outputs("train").get("selector"):
        return {"skipped": "this run fitted no selector policy"}
    version = run.setting("heads_version") or run.name
    dataset_version = run.setting("dataset_version") or run.name
    commit = (run.outputs("build").get("runner") or {}).get("lycaon_commit") or build.checkout_commit(run.checkout) or "unknown"
    out = run.path("dist", "heads-" + version)
    try:
        heads.build(run.path("selector"), version, dataset_version, commit, out)
    except heads.ReleaseError as exc:
        raise RunError(str(exc)) from exc
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


STAGE_FUNCS = {name: globals()["stage_" + name] for name in STAGES}


def plan(run, start=None, until=None, redo=()):
    """Every stage up to `until`, in order, marked run or skip. A stage runs when it is
    not done, is `start` or later, or is named in `redo`."""
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
    if "generators" in overrides and overrides["generators"] is not None:
        if "generator_weights" not in overrides:
            settings["generator_weights"] = {}
        if "archetype_generators" not in overrides:
            settings["archetype_generators"] = {}
    for key, value in overrides.items():
        if value is None:
            continue
        if "." in key:
            section, sub = key.split(".", 1)
            settings[section][sub] = value
        else:
            settings[key] = value
    return settings
