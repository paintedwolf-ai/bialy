"""The factory's configuration: models, workspaces, archetypes, and the host.

Everything is read from config/ and validated once, so a run never starts
with an unpinned repository or a model whose outputs may not train an open
model.
"""

import ipaddress
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .faults import MODEL_FAILURE_CODES

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "config"

PERMISSIVE = {"MIT", "Apache-2.0", "BSD-2-Clause", "BSD-3-Clause", "ISC", "Unlicense", "MIT OR Apache-2.0", "Unlicense OR MIT"}
# Project workflows the runner image installs in every workspace.
WORKFLOWS = ("implement-dispatch", "plan-auto", "recon-auto", "security-auto")
# Where a task's session runs: a checkout of a pinned public repository, or an
# empty directory a request starts a new project in, for one stack.
WORKSPACE_KINDS = ("repository", "stack")
# What a hosted model can be asked to spend on reasoning; none turns it off.
REASONING_EFFORTS = ("none", "low", "medium", "high")


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class Model:
    id: str
    hf: str
    revision: str
    family: str
    license: str
    roles: tuple
    serve: dict | None
    replicas: tuple = ()
    # A provider endpoint serving the pinned weights, instead of a local server:
    # {provider, base_url, model, key_env}. The key is read from key_env at call time.
    # Optional: price_per_million {input, output}, and reasoning_effort (none, low,
    # medium, high), the effort every call to the model asks for: the sessions it
    # drives and the factory's judging and writing.
    hosted: dict | None = None

    @property
    def reasoning_effort(self):
        return (self.hosted or {}).get("reasoning_effort")

    @property
    def port(self):
        return int(self.serve["port"])

    @property
    def wire_model(self):
        return ((self.serve or {}).get("external") or {}).get("model", self.id)

    def tunnels(self):
        external = (self.serve or {}).get("external")
        if external and not external.get("ssh"):
            raise ConfigError("model %s: replicas.local.yaml must give the tunnel (ssh, remote, key) to its external server" % self.id)
        return ([dict(external, port=self.port)] if external else []) + list(self.replicas)

    def ports(self):
        """Every bridge port this model answers on: its own server, then its replicas."""
        return [self.port] + [int(r["port"]) for r in self.replicas] if self.serve else []


@dataclass(frozen=True)
class Repo:
    """A public repository pinned by commit; its sessions start in a fresh checkout."""
    name: str
    url: str
    commit: str
    spdx: str
    language: str
    split: str
    license_files: dict
    kind = "repository"

    def describe(self):
        return "the %s repository (%s)" % (self.name, self.language)


@dataclass(frozen=True)
class Stack:
    """A toolchain for greenfield work; its sessions start in an empty directory. `warm`
    is a shell command run once in an empty directory to fill the package cache with
    what new projects on this stack commonly install."""
    name: str
    language: str
    brief: str
    split: str
    warm: str | None = None
    kind = "stack"

    def describe(self):
        return "a new %s project (%s)" % (self.name, self.brief)


@dataclass(frozen=True)
class Archetype:
    """A kind of request. `workspace` is the kind of workspace its tasks run in, and
    `per_workspace` how many each such workspace gets. `new_files` lets a request in a
    repository name files the repository does not have yet, as long as they would sit
    in a directory it has; requests in a stack's empty directory always may."""
    id: str
    brief: str
    workspace: str
    per_workspace: int
    workflows: dict
    follow_up_rate: float
    prompt_timeout: str
    workflow_prompt_timeout: str
    new_files: bool = False
    reference_rate: float = 0.0
    # The follow-ups a task may carry; every follow-up is a retention decision
    # with the earlier turns to look back on.
    max_follow_ups: int = 2

    def timeout(self, workflow):
        """The per-prompt budget for one of this archetype's tasks."""
        return self.workflow_prompt_timeout if workflow else self.prompt_timeout


def minutes(duration):
    """A budget written as `<n>m` or `<n>h`, in minutes."""
    value = str(duration).strip()
    if len(value) < 2 or value[-1] not in "mh" or not value[:-1].isdigit() or int(value[:-1]) <= 0:
        raise ConfigError("durations are written as <n>m or <n>h, not %r" % duration)
    return int(value[:-1]) * (60 if value[-1] == "h" else 1)


@dataclass
class Factory:
    root: Path
    network: dict
    fleet: dict
    split: dict
    judge: dict
    models: list = field(default_factory=list)
    repos: list = field(default_factory=list)
    stacks: list = field(default_factory=list)
    # What a greenfield request is about: {"domains": [...], "scales": [...]}, drawn per batch.
    seeds: dict = field(default_factory=dict)
    archetypes: list = field(default_factory=list)
    languages: dict = field(default_factory=dict)
    # Settings for `bialy run`, from factory.yaml's run section over RUN_DEFAULTS.
    run: dict = field(default_factory=dict)

    def model(self, model_id):
        for m in self.models:
            if m.id == model_id:
                return m
        raise ConfigError("unknown model %r" % model_id)

    def workspaces(self):
        """Every workspace a task can run in: the repositories, then the stacks."""
        return [*self.repos, *self.stacks]

    def workspace(self, name):
        for w in self.workspaces():
            if w.name == name:
                return w
        raise ConfigError("unknown workspace %r" % name)

    def archetypes_for(self, workspace):
        return [a for a in self.archetypes if a.workspace == workspace.kind]

    def holdout(self):
        """The held-out workspaces: none of their rows trains a head."""
        return {w.name for w in self.workspaces() if w.split == "holdout"}

    def generators(self):
        return [m for m in self.models if "generator" in m.roles]

    def teachers(self):
        return [m for m in self.models if "teacher" in m.roles or "generator" in m.roles]

    def students(self):
        return [m for m in self.models if "student" in m.roles]

    def provider_id(self, model):
        return model.hosted["provider"] + "-" + model.id.replace(".", "-") if model.hosted else "vllm-" + model.id.replace(".", "-")

    def judge_for(self, generator_model):
        """The judge for a row: a model of another family than the one that drove it."""
        family = self.model(generator_model).family if generator_model in {m.id for m in self.models} else None
        for m in self.models:
            if "judge" in m.roles and m.family != family:
                return m
        raise ConfigError("no judge outside family %r" % family)

    def served(self):
        """The models this host serves itself."""
        return [m for m in self.models if m.serve]

    def base_url(self, model, port=None):
        return "http://%s:%d/v1" % (self.network["gateway"], port or model.port)

    def chats(self, model):
        """One chat client per endpoint answering for `model`: its local server and
        replicas, or its provider."""
        from .llm import Chat

        if model.hosted:
            key = os.environ.get(model.hosted["key_env"], "")
            if not key:
                raise ConfigError("model %s: set %s to call %s" % (model.id, model.hosted["key_env"], model.hosted["provider"]))
            return [Chat(model.hosted["base_url"], model.hosted["model"], api_key=key, hosted=True, reasoning_effort=model.reasoning_effort)]
        return [Chat(self.base_url(model, port), model.wire_model) for port in model.ports()]


HUB_ENV = {"dataset_repo": "BIALY_HF_DATASET_REPO", "model_repo": "BIALY_HF_MODEL_REPO", "code_repo": "BIALY_CODE_REPO"}


def hub():
    """config/hub.yaml, with each value overridable from the environment."""
    values = _read("hub.yaml")
    for key, var in HUB_ENV.items():
        if os.environ.get(var):
            values[key] = os.environ[var]
    return values


def _read(name):
    with open(CONFIG / name, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def local_deployment():
    """This deployment's replicas and external-server tunnels by model id, from the
    uncommitted replicas.local.yaml."""
    path = CONFIG / "replicas.local.yaml"
    raw = (yaml.safe_load(path.read_text(encoding="utf-8")) or {}) if path.exists() else {}
    return raw.get("replicas") or {}, raw.get("external") or {}


RUN_DEFAULTS = {
    "lycaon_checkout": "../paintedwolf-code", "generators": [], "generator_weights": {}, "archetype_generators": {},
    "local_workers_only": False, "judges": [], "workspaces": [], "archetypes": [], "task_cap": None, "task_share": None, "seed": 7, "workers": 16,
    "runners": None, "spend_ceiling_usd": 0, "scanner": "pinned", "scanner_candidate": None, "dataset_version": None, "heads_version": None, "code_ref": "HEAD", "stopping": "",
    "task_timeout": None, "tasks_per_runner": None,
    "fail_fast": {"enabled": True, "summary_length_limit": 2, "capture_turn_limit": 3, "task_failure_codes": []},
    "preflight": {"enabled": True, "timeout_seconds": 300},
    "engine_on": {"enabled": True, "deadline_ms": 60000, "heads": None},
    "skillreq": {"writer": None, "families": {"clear": 2, "nearmiss": 1, "multi": 1}, "none_families": 24, "per_family": 6},
    "coderank": {"per_repo": 40},
    "evaluate": {"rows": 500},
    "judge": {"enabled": True},
    # The retention head: how its sessions are shaped, how its head is searched, and how
    # its verdict is read (src/bialy/retention.py).
    "retention": {"follow_up_rate": None, "follow_ups": None,
                  "search": {"lr": [2e-4, 5e-4, 1e-3], "pos_weight": [2.0, 6.0], "truth": ["needed", "later"]},
                  "bootstrap": 500, "min_rows": 20},
    "train": {"recipes": ["B5", "B7G", "E4", "code-rank"], "device": "auto", "max_hours": 24, "threads": 8, "python": "3.12",
              "requirements": None, "torch_index": None, "epochs": None},
    "custom_model": {"enabled": False, "diagnostic_only": False, "student": "gemma-4-e4b-it", "teacher": "gemma-4-31b-it", "min_score": 3,
                     "epoch": None, "tool_surface": "legacy", "host_commit": None, "quality_reports": None, "quality_judge": False, "passes": None,
                     # How the trained student will be served: llama.cpp's server applies the
                     # checkpoint's own chat template, so the trained tokens are the served ones.
                     "student_serving": {"provider": "student", "kind": "openai-compatible", "model": "gemma-4-e4b-it",
                                         "base_url": "http://127.0.0.1:8080/v1", "context_window": 32768},
                     "preload_from_judge": False, "max_routine_reads_per_session": 6, "shadow": False, "repair": False},
}

# Map-valued run settings whose inner keys are model or archetype ids, not fixed setting names.
FREEFORM_MAP_SETTINGS = {"generator_weights", "archetype_generators"}


def run_settings(raw):
    """factory.yaml's run section over the defaults; paths resolve against the repository.
    Unknown keys are refused, so a misspelt setting never silently keeps its default."""
    settings = json.loads(json.dumps(RUN_DEFAULTS))
    for key, value in (raw or {}).items():
        if key not in settings:
            raise ConfigError("run.%s is not a setting" % key)
        if key in FREEFORM_MAP_SETTINGS:
            if value is not None and not isinstance(value, dict):
                raise ConfigError("run.%s must be a mapping" % key)
            settings[key] = dict(value or {})
        elif isinstance(settings[key], dict):
            if value is not None and not isinstance(value, dict):
                raise ConfigError("run.%s is a section of settings, not a single value" % key)
            for k, v in (value or {}).items():
                if k not in settings[key]:
                    raise ConfigError("run.%s.%s is not a setting" % (key, k))
                settings[key][k] = v
        else:
            settings[key] = value
    if settings["task_timeout"] is not None and minutes(str(settings["task_timeout"])) <= 0:
        raise ConfigError("run.task_timeout must be positive")
    for key in ("tasks_per_runner",):
        if settings[key] is not None and int(settings[key]) < 1:
            raise ConfigError("run.%s must be positive" % key)
    if any(int(settings["fail_fast"][k]) < 1 for k in ("summary_length_limit", "capture_turn_limit")):
        raise ConfigError("run.fail_fast thresholds must be positive")
    isolated = settings["fail_fast"]["task_failure_codes"]
    if not isinstance(isolated, list) or any(not isinstance(code, str) or code not in MODEL_FAILURE_CODES for code in isolated):
        raise ConfigError("run.fail_fast.task_failure_codes must list supported model failure codes: turn_closeout_tool_call")
    if isolated and settings["tasks_per_runner"] != 1:
        raise ConfigError("run.fail_fast.task_failure_codes requires tasks_per_runner: 1")
    if int(settings["preflight"]["timeout_seconds"]) < 1:
        raise ConfigError("run.preflight.timeout_seconds must be positive")
    for key in ("lycaon_checkout", "scanner_candidate"):
        value = settings.get(key)
        if value:
            settings[key] = str(value) if Path(str(value)).is_absolute() else str((ROOT / str(value)).resolve())
    return settings


def load():
    factory = _read("factory.yaml")
    replicas, tunnels = local_deployment()
    out = Factory(root=Path(factory["root"]), network=factory["network"], fleet=factory["fleet"],
                  split=factory["split"], judge=factory["judge"], run=run_settings(factory.get("run")))
    # Docker gives every runner an address on the bridge; beyond the subnet, runs fail to start.
    hosts = ipaddress.ip_network(out.network["subnet"]).num_addresses - 3
    if int(out.fleet["runners"]) > hosts:
        raise ConfigError("fleet.runners %s exceeds the %d runner addresses in %s" % (out.fleet["runners"], hosts, out.network["subnet"]))
    for raw in _read("models.yaml")["models"]:
        if raw.get("outputs_trainable") is not True or not str(raw.get("reviewed", "")).strip():
            raise ConfigError("model %s: a person must review its license and set outputs_trainable" % raw.get("id"))
        if len(raw.get("revision", "")) != 40:
            raise ConfigError("model %s: pin a full 40-character revision" % raw.get("id"))
        serve, hosted = raw.get("serve"), raw.get("hosted")
        if (serve is None) == (hosted is None):
            raise ConfigError("model %s: give exactly one of serve (a local server) or hosted (a provider endpoint)" % raw.get("id"))
        if hosted and (not all(hosted.get(k) for k in ("provider", "base_url", "model", "key_env")) or replicas.get(raw["id"])):
            raise ConfigError("model %s: hosted needs provider, base_url, model, and key_env, and takes no replicas" % raw.get("id"))
        if hosted and hosted.get("reasoning_effort") not in (None, *REASONING_EFFORTS):
            raise ConfigError("model %s: hosted.reasoning_effort is one of %s" % (raw.get("id"), ", ".join(REASONING_EFFORTS)))
        external = (serve or {}).get("external")
        if external:
            # The durable pins live here; the deployment's tunnel stays out of the repository.
            if set(external) != {"model", "artifact", "artifact_sha256"} or not all(external.values()) or not re.fullmatch(r"[0-9a-f]{64}", str(external["artifact_sha256"])):
                raise ConfigError("model %s: serve.external pins model (the server's alias), artifact (file name) and artifact_sha256; the tunnel belongs in replicas.local.yaml" % raw.get("id"))
            tunnel = tunnels.get(raw["id"])
            if tunnel:
                if any(not tunnel.get(k) for k in ("ssh", "remote", "key")):
                    raise ConfigError("model %s: replicas.local.yaml external needs ssh, remote and key" % raw["id"])
                serve = dict(serve, external=dict(external, **tunnel))
        out.models.append(Model(id=raw["id"], hf=raw["hf"], revision=raw["revision"], family=raw["family"],
                                license=raw["license"], roles=tuple(raw["roles"]), serve=serve,
                                replicas=tuple(replicas.get(raw["id"]) or ()), hosted=hosted))
    unknown = (set(replicas) | set(tunnels)) - {m.id for m in out.models}
    if unknown:
        raise ConfigError("replicas.local.yaml names unknown models: %s" % ", ".join(sorted(unknown)))
    not_external = set(tunnels) & {m.id for m in out.models if not (m.serve or {}).get("external")}
    if not_external:
        raise ConfigError("replicas.local.yaml gives external tunnels to models the factory serves itself: %s" % ", ".join(sorted(not_external)))
    for raw in _read("repos.yaml")["repos"]:
        repo = Repo(**raw)
        if len(repo.commit) != 40 or repo.spdx not in PERMISSIVE or repo.split not in ("train", "holdout"):
            raise ConfigError("repository %s: pin a commit, a permissive license, and a split" % repo.name)
        if not repo.license_files or any(len(str(d)) != 64 for d in repo.license_files.values()):
            raise ConfigError("repository %s: pin the sha256 of each license text a person read" % repo.name)
        out.repos.append(repo)
    greenfield = _read("stacks.yaml")
    for raw in greenfield["stacks"]:
        stack = Stack(name=raw["name"], language=raw["language"], brief=" ".join(raw["brief"].split()), split=raw["split"],
                      warm=" ".join(raw["warm"].split()) if raw.get("warm") else None)
        if stack.split not in ("train", "holdout"):
            raise ConfigError("stack %s: split is train or holdout" % stack.name)
        out.stacks.append(stack)
    out.seeds = {key: list(greenfield.get(key) or []) for key in ("domains", "scales")}
    if out.stacks and not all(out.seeds.values()):
        raise ConfigError("stacks.yaml needs domains and scales to seed greenfield requests")
    arch = _read("archetypes.yaml")
    defaults = arch["defaults"]
    out.languages = defaults["languages"]
    for raw in arch["archetypes"]:
        workflows = raw.get("workflows") or {}
        if set(workflows) - set(WORKFLOWS) or sum(workflows.values()) > 1.0 or min(workflows.values(), default=0) < 0:
            raise ConfigError("archetype %s: workflows must be installed ones with shares summing to at most 1" % raw["id"])
        workspace = raw.get("workspace", "repository")
        if workspace not in WORKSPACE_KINDS:
            raise ConfigError("archetype %s: workspace is one of %s" % (raw["id"], ", ".join(WORKSPACE_KINDS)))
        if workspace == "stack" and "new_files" in raw:
            raise ConfigError("archetype %s: a stack's workspace starts empty, so every file is new; drop new_files" % raw["id"])
        timeout = str(raw.get("prompt_timeout", defaults["prompt_timeout"]))
        workflow_timeout = str(defaults["workflow_prompt_timeout"])
        minutes(timeout), minutes(workflow_timeout)
        out.archetypes.append(Archetype(id=raw["id"], brief=" ".join(raw["brief"].split()), workspace=workspace,
                                        per_workspace=int(raw["per_workspace"]), workflows=workflows,
                                        follow_up_rate=float(raw.get("follow_up_rate", defaults["follow_up_rate"])),
                                        max_follow_ups=int(raw.get("max_follow_ups", defaults.get("max_follow_ups", 2))),
                                        prompt_timeout=timeout, workflow_prompt_timeout=max(timeout, workflow_timeout, key=minutes),
                                        new_files=workspace == "stack" or bool(raw.get("new_files", False)),
                                        reference_rate=float(raw.get("reference_rate", 0.0))))
    names = [w.name for w in out.workspaces()]
    if len(set(names)) != len(names):
        raise ConfigError("repository and stack names must be unique together: a workspace's name is its rows' identity")
    for kind in WORKSPACE_KINDS:
        if any(w.kind == kind for w in out.workspaces()) and not any(a.workspace == kind for a in out.archetypes):
            raise ConfigError("no archetype writes requests for a %s workspace" % kind)
    return out
