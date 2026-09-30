"""The factory's configuration: models, repositories, archetypes, and the host.

Everything is read from config/ and validated once, so a run never starts
with an unpinned repository or a model whose outputs may not train an open
model.
"""

import ipaddress
import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "config"

PERMISSIVE = {"MIT", "Apache-2.0", "BSD-2-Clause", "BSD-3-Clause", "ISC", "Unlicense", "MIT OR Apache-2.0", "Unlicense OR MIT"}
# Project workflows the runner image installs in every checkout.
WORKFLOWS = ("implement-dispatch",)


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
    name: str
    url: str
    commit: str
    spdx: str
    language: str
    split: str
    license_files: dict


@dataclass(frozen=True)
class Archetype:
    id: str
    brief: str
    per_repo: int
    workflows: dict
    follow_up_rate: float


@dataclass
class Factory:
    root: Path
    network: dict
    fleet: dict
    split: dict
    judge: dict
    models: list = field(default_factory=list)
    repos: list = field(default_factory=list)
    archetypes: list = field(default_factory=list)
    languages: dict = field(default_factory=dict)

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

    def generators(self):
        return [m for m in self.models if "generator" in m.roles]

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


def load():
    factory = _read("factory.yaml")
    replicas = local_replicas()
    out = Factory(root=Path(factory["root"]), network=factory["network"], fleet=factory["fleet"],
                  split=factory["split"], judge=factory["judge"])
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
    arch = _read("archetypes.yaml")
    defaults = arch["defaults"]
    out.languages = defaults["languages"]
    for raw in arch["archetypes"]:
        workflows = raw.get("workflows") or {}
        if set(workflows) - set(WORKFLOWS) or sum(workflows.values()) > 1.0 or min(workflows.values(), default=0) < 0:
            raise ConfigError("archetype %s: workflows must be installed ones with shares summing to at most 1" % raw["id"])
        out.archetypes.append(Archetype(id=raw["id"], brief=raw["brief"].strip(), per_repo=int(raw["per_repo"]),
                                        workflows=workflows, follow_up_rate=float(raw.get("follow_up_rate", defaults["follow_up_rate"]))))
    names = [r.name for r in out.repos]
    if len(set(names)) != len(names):
        raise ConfigError("repository names must be unique")
    return out
