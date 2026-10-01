"""The factory's configuration: models, workspaces, archetypes, and the host.

Everything is read from config/ and validated once, so a run never starts
with an unpinned repository or a model whose outputs may not train an open
model.
"""

import ipaddress
import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "config"

PERMISSIVE = {"MIT", "Apache-2.0", "BSD-2-Clause", "BSD-3-Clause", "ISC", "Unlicense", "MIT OR Apache-2.0", "Unlicense OR MIT"}
# Project workflows the runner image installs in every workspace.
WORKFLOWS = ("implement-dispatch",)
# Where a task's session runs: a checkout of a pinned public repository, or an
# empty directory a request starts a new project in, for one stack.
WORKSPACE_KINDS = ("repository", "stack")


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
    hosted: dict | None = None

    @property
    def port(self):
        return int(self.serve["port"])

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

    def repo(self, name):
        for r in self.repos:
            if r.name == name:
                return r
        raise ConfigError("unknown repository %r" % name)

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
            return [Chat(model.hosted["base_url"], model.hosted["model"], api_key=key, hosted=True)]
        return [Chat(self.base_url(model, port), model.id) for port in model.ports()]


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


def local_replicas():
    """This deployment's replicas by model id, from the uncommitted replicas.local.yaml."""
    path = CONFIG / "replicas.local.yaml"
    if not path.exists():
        return {}
    return (yaml.safe_load(path.read_text(encoding="utf-8")) or {}).get("replicas") or {}


RUN_DEFAULTS = {
    "lycaon_checkout": "../paintedwolf-code", "generators": [], "judges": [], "workspaces": [], "task_cap": None, "seed": 7, "workers": 16,
    "runners": None, "spend_ceiling_usd": 0, "scanner": "pinned", "scanner_candidate": None, "dataset_version": None, "heads_version": None, "code_ref": "HEAD", "stopping": "",
    "engine_on": {"enabled": True, "deadline_ms": 60000},
    "skillreq": {"writer": None, "families": {"clear": 2, "nearmiss": 1, "multi": 1}, "none_families": 24, "per_family": 6},
    "coderank": {"per_repo": 40},
    "train": {"recipes": ["B5", "B7G", "E4", "code-rank"], "device": "auto", "max_hours": 24, "threads": 8, "python": "3.12",
              "requirements": None, "torch_index": None, "epochs": None},
}


def run_settings(raw):
    """factory.yaml's run section over the defaults; paths resolve against the repository.
    Unknown keys are refused, so a misspelt setting never silently keeps its default."""
    settings = json.loads(json.dumps(RUN_DEFAULTS))
    for key, value in (raw or {}).items():
        if key not in settings:
            raise ConfigError("run.%s is not a setting" % key)
        if key == "engine_on" and isinstance(value, bool):
            settings[key]["enabled"] = value
        elif isinstance(settings[key], dict):
            for k, v in (value or {}).items():
                if k not in settings[key]:
                    raise ConfigError("run.%s.%s is not a setting" % (key, k))
                settings[key][k] = v
        else:
            settings[key] = value
    for key in ("lycaon_checkout", "scanner_candidate"):
        value = settings.get(key)
        if value:
            settings[key] = str(value) if Path(str(value)).is_absolute() else str((ROOT / str(value)).resolve())
    return settings


def load():
    factory = _read("factory.yaml")
    replicas = local_replicas()
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
        out.models.append(Model(id=raw["id"], hf=raw["hf"], revision=raw["revision"], family=raw["family"],
                                license=raw["license"], roles=tuple(raw["roles"]), serve=serve,
                                replicas=tuple(replicas.get(raw["id"]) or ()), hosted=hosted))
    unknown = set(replicas) - {m.id for m in out.models}
    if unknown:
        raise ConfigError("replicas.local.yaml names unknown models: %s" % ", ".join(sorted(unknown)))
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
                                        prompt_timeout=timeout, workflow_prompt_timeout=max(timeout, workflow_timeout, key=minutes),
                                        new_files=workspace == "stack" or bool(raw.get("new_files", False))))
    names = [w.name for w in out.workspaces()]
    if len(set(names)) != len(names):
        raise ConfigError("repository and stack names must be unique together: a workspace's name is its rows' identity")
    for kind in WORKSPACE_KINDS:
        if any(w.kind == kind for w in out.workspaces()) and not any(a.workspace == kind for a in out.archetypes):
            raise ConfigError("no archetype writes requests for a %s workspace" % kind)
    return out
