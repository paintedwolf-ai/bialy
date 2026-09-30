"""One command on one machine: `bialy run` drives a pass from repositories to a
head release, resumable, with nothing uploaded unless asked.

Every stage writes its outputs under <root>/runs/<name>/ and its status to
run.json there. A rerun skips finished stages; `--from` reruns from a stage,
`--until` stops after one. Settings come from factory.yaml's `run` section and
the flags that override it for one run.
"""
import dataclasses
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from . import anchor, config, fleet, heads, hub, judge, provenance, release, repos, serve, split, tasks

STAGES = ("check", "repos", "corpus", "tasks", "image", "warm", "plan", "drive", "collect",
          "judge", "split", "release", "train", "evaluate", "release_heads", "report")

# Recipes the train stage knows, as scripts/bialy/train.py arguments over the run's split.
RECIPES = {
    "B5": {"head": "turn-load", "families": "tools", "extra": ["--tool-truth", "consensus", "--tool-weight", "none", "--tool-negatives", "24",
                                                              "--lr", "5e-4", "--pos-weight", "6", "--seed", "11", "--batch-size", "64",
                                                              "--epochs", "45", "--patience", "8"]},
    "B7G": {"head": "guide-load", "families": "guides", "extra": ["--tool-truth", "consensus", "--tool-weight", "none", "--tool-negatives", "24",
                                                                 "--lr", "5e-4", "--pos-weight", "6", "--seed", "11", "--batch-size", "64",
                                                                 "--epochs", "45", "--patience", "8"]},
}
BACKBONE = ("convaiinnovations/laya-multilingual", "e4e9ddf21a7b1903b7acffd8814ad4307bf63a67")


class RunError(RuntimeError):
    pass


@dataclasses.dataclass
class Run:
    factory: object
    name: str
    settings: dict
    push: bool = False
    rebuild_image: bool = False
    dry_run: bool = False
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

    def path(self, *parts):
        return self.dir.joinpath(*parts)

    def setting(self, key, default=None):
        return self.settings.get(key, default)


def stamp():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


# ---- stages ---------------------------------------------------------------

def stage_check(run):
    f = run.factory
    faults = []
    for model in f.models:
        if model.hosted and not os.environ.get(model.hosted["key_env"]):
            faults.append("model %s needs %s in the environment" % (model.id, model.hosted["key_env"]))
    generators = run_generators(run)
    if not generators:
        faults.append("no generator model is usable: name hosted generators in run.generators or serve one locally")
    for model in generators:
        try:
            f.judge_for(model.id)
        except config.ConfigError as exc:
            faults.append(str(exc))
    bin_dir = Path(run.setting("lycaon_bin"))
    if run.runs("image") or (run.runs("corpus") and not run.setting("corpus_json")):
        for name in ("lycaon", "lycaon-debug"):
            if not (bin_dir / name).exists():
                faults.append("run.lycaon_bin lacks %s (a linux/amd64 build the runner image carries)" % name)
    if run.runs("image"):
        try:
            fleet.check_engine(run.setting("engine_dir"))
        except ValueError as exc:
            faults.append(str(exc))
    if run.runs("release"):
        checkout = Path(run.setting("lycaon_checkout"))
        for rel in ("scripts/bialy/train.py", "scripts/bialy/row.schema.json", "scripts/bialy/replay_eval.py"):
            if not (checkout / rel).exists():
                faults.append("run.lycaon_checkout lacks %s" % rel)
    if run.runs("image") and shutil.which("docker") is None:
        faults.append("docker is not on PATH")
    for recipe in run.setting("train", {}).get("recipes", []):
        if recipe not in RECIPES:
            faults.append("unknown training recipe %s; known: %s" % (recipe, ", ".join(sorted(RECIPES))))
    if faults:
        raise RunError("; ".join(faults))
    return {"generators": [m.id for m in generators], "judges": sorted({f.judge_for(m.id).id for m in generators}),
            "repositories": len(f.repos), "runners": int(run.setting("runners") or f.fleet["runners"])}


def run_generators(run):
    named = run.setting("generators") or []
    generators = run.factory.generators()
    if named:
        generators = [m for m in generators if m.id in named]
    return [m for m in generators if m.hosted or m.serve]


def stage_repos(run):
    problems = repos.fetch(run.factory) or repos.verify(run.factory)
    if problems:
        raise RunError("; ".join(problems))
    return {"repositories": [r.name for r in run.factory.repos]}


def stage_corpus(run):
    """corpus.json from the runners' own build, so option texts match the rows."""
    out = run.path("corpus.json")
    given = run.setting("corpus_json")
    if given:
        shutil.copy2(given, out)
    else:
        debug = Path(run.setting("lycaon_bin")) / "lycaon-debug"
        result = subprocess.run([str(debug), "decide", "corpus", "--out", str(out)], capture_output=True, text=True)
        if result.returncode != 0:
            raise RunError("lycaon-debug decide corpus failed (%s); on a host that cannot run the runner binary, set run.corpus_json"
                           % result.stderr.strip()[-300:])
    revision = json.loads(out.read_text(encoding="utf-8")).get("catalog_revision")
    return {"corpus": str(out), "catalog_revision": revision}


def scaled_factory(run):
    """The factory with generation limited to the run's cap and generator list."""
    f = run.factory
    cap = run.setting("task_cap")
    generators = {m.id for m in run_generators(run)}
    models = [m for m in f.models if "generator" not in m.roles or m.id in generators]
    archetypes = [dataclasses.replace(a, per_repo=min(a.per_repo, int(cap))) if cap else a for a in f.archetypes]
    repo_names = run.setting("repos") or []
    repo_list = [r for r in f.repos if not repo_names or r.name in repo_names]
    return dataclasses.replace(f, models=models, archetypes=archetypes, repos=repo_list)


def stage_tasks(run):
    out = run.path("tasks")
    scaled = scaled_factory(run)
    counts = tasks.generate(scaled, run.factory.root / "repos", out, seed=int(run.setting("seed", 7)),
                            workers=int(run.setting("workers", 16)))
    provenance.record(run.factory, out / "provenance.json", "tasks", uses_engine=False, seed=int(run.setting("seed", 7)), counts=counts)
    return {"tasks": str(out), "counts": counts}


def stage_image(run):
    tag = run.setting("image") or run.factory.fleet["image"]
    fleet.ensure_network(run.factory)
    present = subprocess.run(["docker", "image", "inspect", tag], capture_output=True).returncode == 0
    if present and not run.rebuild_image:
        return {"image": tag, "built": False}
    fleet.build_image(run.factory, run.setting("lycaon_bin"), run.setting("engine_dir"), tag)
    return {"image": tag, "built": True}


def stage_warm(run):
    failed = {name: code for name, code in fleet.warm_cache(run.factory, run.setting("image")).items() if code}
    if failed:
        raise RunError("cache warm failed for %s" % ", ".join(sorted(failed)))
    return {"warmed": len(run.factory.repos)}


def stage_plan(run):
    run_dir = run.path("fleet")
    run_dir.mkdir(parents=True, exist_ok=True)
    scaled = scaled_factory(run)
    decide_env = {"LYCAON_DECIDE_DISABLED": "1"}
    planned = fleet.plan_shards(scaled, run.path("tasks"), run_dir, decide_env, run.setting("repos") or None)
    tasks_record = run.path("tasks", "provenance.json")
    provenance.record(run.factory, run_dir / "provenance.json", "fleet", bin_dir=run.setting("lycaon_bin"), tasks=str(run.path("tasks")),
                      pilot=None, decide_env=decide_env, image=provenance.image(run.setting("image") or run.factory.fleet["image"]),
                      tasks_provenance=json.loads(tasks_record.read_text()) if tasks_record.exists() else None)
    return {"planned": planned}


def stage_drive(run):
    fleet.ensure_network(run.factory)
    done, failed = fleet.run(run.factory, run.path("fleet"), int(run.setting("runners") or run.factory.fleet["runners"]), [], run.setting("image"))
    if not done:
        raise RunError("no shard produced rows (%d failed)" % failed)
    return {"shards_done": done, "shards_failed": failed}


def stage_collect(run):
    out, stats = fleet.collect(run.path("fleet"), run.setting("pass_name", "engine-off"), run.path("tasks"))
    if not stats["rows"]:
        raise RunError("collect found no rows")
    return {"rows": str(out), "stats": stats}


def stage_judge(run):
    corpus = json.loads(run.path("corpus.json").read_text(encoding="utf-8"))
    rows_in = run.setting("rows") or str(run.path("fleet", "rows.jsonl"))
    out = run.path("judged.jsonl")
    scaled = scaled_factory(run)
    report = judge.run(scaled, corpus, rows_in, str(out), workers=int(run.setting("workers", 16)))
    report["provenance"] = provenance.record(run.factory, str(out) + ".provenance.json", "judge", rows=rows_in, units=list(judge.UNITS),
                                             corpus_revision=corpus["catalog_revision"],
                                             corpus_sha256=provenance.file_sha256(run.path("corpus.json")))["at"]
    (run.path("judged.jsonl.agreement.json")).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return {"judged": str(out), "report": {k: v for k, v in report.items() if k != "provenance"}}


def stage_split(run):
    counts = split.split(run.factory, [str(run.path("judged.jsonl"))], run.path("split"))
    return {"split": str(run.path("split")), "counts": counts}


def stage_release(run):
    version = run.setting("dataset_version") or "dataset-" + run.name
    out = run.path("dist", "dataset-" + version)
    checkout = Path(run.setting("lycaon_checkout"))
    stages = [p for p in (run.path("tasks", "provenance.json"), run.path("fleet", "provenance.json"),
                          run.path("judged.jsonl.provenance.json")) if p.exists()]
    driven = [run.path("fleet", "tasks-driven.jsonl")] if run.path("fleet", "tasks-driven.jsonl").exists() else []
    if not driven:
        raise RunError("no tasks-driven.jsonl; the release names every task the run drove")
    release.build(run.factory, run.path("split"), version, out, checkout / "scripts/bialy/row.schema.json", run.path("corpus.json"),
                  run.setting("code_ref", "HEAD"), agreement=[str(run.path("judged.jsonl.agreement.jsonl"))],
                  stages=[str(p) for p in stages], driven=[str(p) for p in driven], coderank=None, stopping=run.setting("stopping", ""))
    return {"release": str(out), "version": version, "anchor": str(anchor.path("dataset", version))}


def train_env(run):
    settings = run.setting("train", {})
    env = dict(os.environ, HF_HOME=str(run.factory.root / "hf"), PYTHONUNBUFFERED="1",
               LYCAON_DECIDE_MODEL_ID=BACKBONE[0], OMP_NUM_THREADS=str(settings.get("threads", 8)))
    device = settings.get("device", "auto")
    if device and device != "auto":
        env["LYCAON_DECIDE_DEVICE"] = device
    return env


def ensure_backbone(run):
    home = run.factory.root / "hf"
    snapshot = home / "hub" / ("models--" + BACKBONE[0].replace("/", "--")) / "snapshots" / BACKBONE[1]
    if snapshot.exists():
        return snapshot
    subprocess.run(["uvx", "--from", "huggingface_hub", "hf", "download", BACKBONE[0], "--revision", BACKBONE[1]],
                   check=True, env=dict(os.environ, HF_HOME=str(home)), capture_output=True, text=True)
    refs = snapshot.parents[1] / "refs"
    refs.mkdir(parents=True, exist_ok=True)
    (refs / "main").write_text(BACKBONE[1], encoding="utf-8")
    return snapshot


def trainer_python(run):
    """The interpreter the trainer runs under: the run's venv, made on first use."""
    settings = run.setting("train", {})
    venv = run.factory.root / "venvs" / "train"
    python = venv / "bin" / "python"
    if python.exists():
        return python
    requirements = Path(settings.get("requirements") or Path(__file__).resolve().parents[2] / "train-host" / "requirements-cuda.txt")
    subprocess.run(["uv", "venv", "-p", str(settings.get("python", "3.12")), str(venv)], check=True, capture_output=True, text=True)
    install = ["uv", "pip", "install", "-r", str(requirements)]
    if settings.get("torch_index"):
        install += ["--index-url", settings["torch_index"], "--index-strategy", "unsafe-best-match"]
    subprocess.run(install, check=True, env=dict(os.environ, VIRTUAL_ENV=str(venv)), capture_output=True, text=True)
    return python


def stage_train(run):
    settings = run.setting("train", {})
    recipes = settings.get("recipes") or []
    if not recipes:
        return {"skipped": "run.train.recipes is empty"}
    checkout = Path(run.setting("lycaon_checkout"))
    heads_dir = run.path("heads")
    heads_dir.mkdir(parents=True, exist_ok=True)
    corpus = run.path("corpus.json")
    independent = independent_corpus(run, corpus)
    env = train_env(run)
    ensure_backbone(run)
    python = trainer_python(run)
    trained = {}
    budget = float(settings.get("max_hours", 24)) * 3600
    started = time.time()
    for recipe in recipes:
        spec = RECIPES[recipe]
        out = heads_dir / ("%s.safetensors" % spec["head"])
        if out.exists():
            trained[recipe] = {"head": str(out), "reused": True}
            continue
        remaining = budget - (time.time() - started)
        if remaining <= 0:
            raise RunError("training budget of %s hours spent before %s" % (settings.get("max_hours", 24), recipe))
        args = [str(python), "scripts/bialy/train.py", "--corpus", str(independent), "--train", str(run.path("split", "train.jsonl")),
                "--val", str(run.path("split", "val.jsonl")), "--families", spec["families"], *spec["extra"],
                "--label", "%s-turn-load-%s-%s" % (run.name, recipe, spec["head"]), "--out", str(out)]
        log = heads_dir / ("%s.log" % recipe)
        with open(log, "w", encoding="utf-8") as fh:
            try:
                result = subprocess.run(args, cwd=str(checkout), env=env, stdout=fh, stderr=subprocess.STDOUT, timeout=remaining)
            except subprocess.TimeoutExpired as exc:
                raise RunError("recipe %s exceeded the training budget; see %s" % (recipe, log)) from exc
        if result.returncode != 0 or not out.exists():
            raise RunError("recipe %s failed (exit %s); see %s" % (recipe, result.returncode, log))
        trained[recipe] = {"head": str(out), "log": str(log), "seconds": round(time.time() - started)}
    return {"heads": trained}


def independent_corpus(run, corpus):
    """The corpus in the independent encoding the turn heads train over."""
    out = run.path("independent-corpus.json")
    if out.exists():
        return out
    checkout = Path(run.setting("lycaon_checkout"))
    script = Path(__file__).resolve().parents[2] / "train-host" / "independent_corpus.py"
    result = subprocess.run([sys.executable, str(script), str(corpus), str(out)], cwd=str(checkout), capture_output=True, text=True)
    if result.returncode != 0:
        raise RunError("independent corpus failed: %s" % result.stderr.strip()[-300:])
    return out


def stage_evaluate(run):
    launcher = run.setting("engine_launcher")
    if not launcher or not Path(launcher).exists():
        return {"skipped": "run.engine_launcher names no engine that loads the trained heads on this machine"}
    checkout = Path(run.setting("lycaon_checkout"))
    evals = {}
    for name in ("val", "holdout"):
        rows = run.path("split", "%s.jsonl" % name)
        out = run.path("eval", "%s.json" % name)
        out.parent.mkdir(parents=True, exist_ok=True)
        args = [sys.executable, "scripts/bialy/replay_eval.py", "--corpus", str(run.path("corpus.json")), "--examples", str(rows),
                "--engine", launcher, "--tool-truth", "consensus", "--json", str(out)]
        result = subprocess.run(args, cwd=str(checkout), capture_output=True, text=True)
        if result.returncode != 0:
            raise RunError("replay on %s failed: %s" % (name, result.stderr.strip()[-300:]))
        evals[name] = str(out)
    return {"evals": evals}


def stage_release_heads(run):
    heads_dir = run.path("heads")
    if not any(heads_dir.glob("*.safetensors")) if heads_dir.exists() else True:
        return {"skipped": "no heads were trained"}
    version = run.setting("heads_version") or "heads-" + run.name
    dataset_version = run.setting("dataset_version") or "dataset-" + run.name
    state = run.load_state()
    evals = (state["stages"].get("evaluate", {}).get("outputs") or {}).get("evals") or {}
    commit = subprocess.run(["git", "-C", run.setting("lycaon_checkout"), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    out = run.path("dist", "heads-" + version)
    heads.build(heads_dir, version, dataset_version, commit or "unknown", evals, {}, "base checkpoint", {}, out)
    return {"release": str(out), "version": version, "anchor": str(anchor.path("heads", version))}


def usage_summary(run):
    path = run.path("usage.jsonl")
    if not path.exists():
        return {}
    totals = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            entry = json.loads(line)
        except ValueError:
            continue
        usage = entry.get("usage") or {}
        t = totals.setdefault(entry.get("model", "?"), {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0})
        t["calls"] += 1
        t["prompt_tokens"] += int(usage.get("prompt_tokens") or 0)
        t["completion_tokens"] += int(usage.get("completion_tokens") or 0)
    return totals


def estimated_spend(run, totals):
    """Dollars, when models declare hosted.price_per_million {input, output}."""
    spend = 0.0
    priced = False
    for model in run.factory.models:
        price = (model.hosted or {}).get("price_per_million")
        key = (model.hosted or {}).get("model", model.id)
        t = totals.get(key) or totals.get(model.id)
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
    else:
        lines.append("No provider calls were recorded.")
    outputs = state["stages"]
    dataset = (outputs.get("release", {}).get("outputs") or {})
    head_release = (outputs.get("release_heads", {}).get("outputs") or {})
    lines += ["", "## Publishing", ""]
    published = {}
    for kind, rel in (("dataset", dataset), ("heads", head_release)):
        if not rel.get("release"):
            continue
        if run.push and publish:
            published[kind] = hub.publish(kind, rel["release"], rel["version"], push=True)
            lines.append("- %s %s pushed: %s" % (kind, rel["version"], published[kind].get("revision")))
        else:
            lines.append("- `bialy publish %s --release %s --version %s --push`" % (kind, rel["release"], rel["version"]))
    if not dataset and not head_release:
        lines.append("Nothing to publish.")
    run.path("REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"report": str(run.path("REPORT.md")), "published": published}


STAGE_FUNCS = {
    "check": stage_check, "repos": stage_repos, "corpus": stage_corpus, "tasks": stage_tasks, "image": stage_image,
    "warm": stage_warm, "plan": stage_plan, "drive": stage_drive, "collect": stage_collect, "judge": stage_judge,
    "split": stage_split, "release": stage_release, "train": stage_train, "evaluate": stage_evaluate,
    "release_heads": stage_release_heads, "report": stage_report,
}


def plan(run, start=None, until=None):
    """The stages this invocation will run, in order."""
    state = run.load_state()
    names = list(STAGES)
    if until:
        names = names[:names.index(until) + 1]
    out = []
    for name in names:
        done = state["stages"].get(name, {}).get("status") == "done"
        if start and STAGES.index(name) >= STAGES.index(start):
            done = False
        out.append((name, "skip" if done else "run"))
    return out


def execute(run, start=None, until=None, funcs=None):
    funcs = funcs or STAGE_FUNCS
    os.environ["BIALY_USAGE_JSONL"] = str(run.path("usage.jsonl"))
    state = run.load_state()
    if start:
        for name in STAGES[STAGES.index(start):]:
            state["stages"].pop(name, None)
        run.save_state(state)
    for name, action in plan(run, start, until):
        if action == "skip":
            continue
        started = time.time()
        print("%s: %s" % (run.name, name), flush=True)
        try:
            check_spend(run)
            outputs = funcs[name](run)
            state = run.load_state()
            state["stages"][name] = {"status": "done", "at": stamp(), "seconds": round(time.time() - started), "outputs": outputs}
            run.save_state(state)
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
    settings = dict(factory.run)
    for key, value in overrides.items():
        if value is not None:
            settings[key] = value
    return settings
