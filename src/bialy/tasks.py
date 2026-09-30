"""Write the requests generated sessions answer.

For every repository and archetype, a generator model reads facts about the
repository (its files, README, and recent history) and writes requests a
developer could make. Requests that name files the repository does not have
are dropped, near-duplicates share a prompt group so a split never puts them
on both sides, and each task is assigned the model that will drive its
session, the workflow it starts, if any, and follow-ups.
"""

import concurrent.futures
import hashlib
import json
import random
import re
import subprocess
from pathlib import Path

import yaml

from .config import ROOT, ConfigError
from .llm import Chat

WORKFLOW_ROOT = ROOT / "runner" / "workflows"

SYSTEM = """You write realistic requests that developers type to an AI coding assistant working in their repository.
Write each request the way a busy engineer would: some terse, some detailed, some with pasted error output, some polite, some blunt.
Never tell the assistant which tool or command to use to do the work unless a developer naturally would ("run the tests", "check git blame").
Refer only to files, functions, and types that appear in the facts you are given.
Reply with one JSON object only: {"tasks": [{"prompt": "...", "follow_ups": ["..."]}]}. follow_ups is a list of zero to two later messages in the same conversation."""

REPLY_SCHEMA = {
    "type": "object", "required": ["tasks"], "additionalProperties": False,
    "properties": {"tasks": {"type": "array", "items": {
        "type": "object", "required": ["prompt", "follow_ups"], "additionalProperties": False,
        "properties": {"prompt": {"type": "string"}, "follow_ups": {"type": "array", "items": {"type": "string"}, "maxItems": 2}}}}},
}

PATHLIKE = re.compile(r"(?<![\w/.-])((?:[\w.-]+/)+[\w.-]+\.\w+|[\w-]+\.(?:py|go|js|ts|tsx|rs|rb|java|kt|php|md|toml|yaml|yml|json|lock|cfg|ini|txt))(?![\w/])")
LANGUAGE_NAMES = {"es": "Spanish", "de": "German", "fr": "French", "pt": "Portuguese", "ja": "Japanese", "zh": "Chinese", "ko": "Korean", "it": "Italian"}


def repo_files(path):
    out = subprocess.run(["git", "-C", str(path), "ls-files"], check=True, capture_output=True, text=True).stdout
    return [line for line in out.splitlines() if line]


def repo_facts(path, rng):
    """What the generator may refer to: a sample of the tree, the README, and recent commits."""
    files = repo_files(path)
    code = [f for f in files if not f.startswith((".github/", "docs/")) and "/fixtures/" not in f]
    sample = sorted(rng.sample(code, min(180, len(code))))
    readme = ""
    for name in ("README.md", "README.rst", "README", "readme.md"):
        if (path / name).exists():
            readme = (path / name).read_text(encoding="utf-8", errors="replace")[:3000]
            break
    log = subprocess.run(["git", "-C", str(path), "log", "--format=%h %s", "-n", "25"], capture_output=True, text=True).stdout
    return "Files (a sample of %d):\n%s\n\nREADME (start):\n%s\n\nRecent commits:\n%s" % (len(files), "\n".join(sample), readme, log)


def mentioned_paths_exist(prompt, files):
    """Every path-like token names a file the repository has (by full path or basename)."""
    names = set(files) | {f.rsplit("/", 1)[-1] for f in files}
    for token in PATHLIKE.findall(prompt):
        token = token.strip("./")
        if token not in names and not any(f.endswith("/" + token) for f in files):
            return False
    return True


def shingles(text, n=3):
    words = re.findall(r"\w+", text.lower())
    return {" ".join(words[i:i + n]) for i in range(max(1, len(words) - n + 1))}


def minhash(text, seeds):
    grams = shingles(text)
    return tuple(min(int(hashlib.blake2b((s + g).encode(), digest_size=8).hexdigest(), 16) for g in grams) for s in seeds)


def group_prompts(prompts, threshold=0.6, bands=16, rows_per_band=4):
    """Near-duplicate groups by MinHash LSH over word 3-shingles: prompts whose estimated
    Jaccard similarity reaches `threshold` share a group id."""
    seeds = [str(i) for i in range(bands * rows_per_band)]
    signatures = [minhash(p, seeds) for p in prompts]
    parent = list(range(len(prompts)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    buckets = {}
    for i, sig in enumerate(signatures):
        for b in range(bands):
            key = (b, sig[b * rows_per_band:(b + 1) * rows_per_band])
            buckets.setdefault(key, []).append(i)
    for members in buckets.values():
        for j in members[1:]:
            a, b = find(members[0]), find(j)
            if a != b and sum(x == y for x, y in zip(signatures[a], signatures[j], strict=True)) / len(seeds) >= threshold:
                parent[b] = a
    return [find(i) for i in range(len(prompts))]


def provider_id(model_id):
    """The provider instance a runner's sidecar declares for a served model."""
    return "vllm-" + model_id.replace(".", "-")


def clean(text):
    """Model text as valid UTF-8: a lone surrogate from a broken token becomes "?"."""
    return text.encode("utf-8", "replace").decode("utf-8").strip()


def generate_batch(chat, facts, archetype, count, language, seed):
    lang = ("Write every request and follow-up in %s." % LANGUAGE_NAMES[language]) if language else "Write in English."
    user = "%s\n\nWrite %d different requests of this kind:\n%s\n%s" % (facts, count, archetype.brief, lang)
    # Writing requests needs variety, not deliberation: sample hot, skip the reasoning phase.
    value = chat.json(SYSTEM, user, temperature=0.9, seed=seed, thinking=False, max_tokens=4096, schema=REPLY_SCHEMA)
    tasks = value.get("tasks") if isinstance(value, dict) else value
    out = []
    for t in tasks or []:
        if isinstance(t, dict) and isinstance(t.get("prompt"), str):
            follow = [clean(f) for f in t.get("follow_ups") or [] if isinstance(f, str) and f.strip()][:2]
            out.append({"prompt": clean(t["prompt"]), "follow_ups": follow})
    return out


def generate(factory, repos_dir, out_dir, seed=7, workers=48):
    """Write tasks/<repo>.jsonl for every configured repository. Each generated batch is
    appended to raw/<repo>.jsonl as it completes, so an interrupted run resumes where it
    stopped and the final files are rebuilt from every batch."""
    out_dir = Path(out_dir)
    raw_dir = out_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    generators = factory.generators()
    chats = {m.id: Chat(factory.base_url(m), m.id) for m in generators}
    done = set()
    for path in raw_dir.glob("*.jsonl"):
        for line in path.read_text(encoding="utf-8").splitlines():
            done.add(json.loads(line)["job"])
    jobs = []
    for repo in factory.repos:
        rng = random.Random("%s:%s" % (seed, repo.name))
        facts = repo_facts(Path(repos_dir) / repo.name, rng)
        for arch in factory.archetypes:
            remaining, n = arch.per_repo, 0
            while remaining > 0:
                count = min(6, remaining)
                other = rng.random() < factory.languages["other_rate"]
                language = rng.choice(factory.languages["choices"]) if other else None
                author = generators[(n + len(jobs)) % len(generators)].id
                key = "%s/%s/%d" % (repo.name, arch.id, n)
                job_seed = rng.randrange(1 << 30)
                if key not in done:
                    jobs.append((key, repo, arch, facts, count, language, author, job_seed))
                remaining -= count
                n += 1
    print("%d batches to write, %d already written" % (len(jobs), len(done)), flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(generate_batch, chats[j[6]], j[3], j[2], j[4], j[5], j[7]): j for j in jobs}
        for fut in concurrent.futures.as_completed(futures):
            key, repo, arch, _, _, language, author, _ = futures[fut]
            try:
                batch = fut.result()
            except Exception as exc:  # a malformed reply loses one batch, not the run
                print("batch %s failed: %s" % (key, exc), flush=True)
                continue
            with open(raw_dir / (repo.name + ".jsonl"), "a", encoding="utf-8") as fh:
                fh.write(json.dumps({"job": key, "archetype": arch.id, "lang": language or "en", "author": author, "tasks": batch}, ensure_ascii=False) + "\n")
    counts = {}
    for repo in factory.repos:
        raw = []
        path = raw_dir / (repo.name + ".jsonl")
        for line in (path.read_text(encoding="utf-8").splitlines() if path.exists() else []):
            entry = json.loads(line)
            raw += [{**t, "archetype": entry["archetype"], "lang": entry["lang"], "author": entry["author"]} for t in entry["tasks"]]
        counts[repo.name] = finalize(factory, repo, raw, repo_files(Path(repos_dir) / repo.name), out_dir / (repo.name + ".jsonl"), seed)
    return counts


def workflow_versions(factory):
    """Pin tasks to the manifests installed by the runner image."""
    versions = {}
    for name in sorted({name for arch in factory.archetypes for name in arch.workflows}):
        path = WORKFLOW_ROOT / name / "workflow.yaml"
        try:
            manifest = yaml.safe_load(path.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError) as exc:
            raise ConfigError("cannot read workflow manifest %s: %s" % (path, exc)) from exc
        if not isinstance(manifest, dict) or manifest.get("id") != name:
            raise ConfigError("workflow manifest %s must declare id %s" % (path, name))
        version = manifest.get("version")
        if not isinstance(version, str) or not version.strip() or version != version.strip():
            raise ConfigError("workflow manifest %s must declare a nonempty version string" % path)
        versions[name] = version
    return versions


def finalize(factory, repo, raw, files, out_path, seed):
    """Validate, group, and assign one repository's tasks, then write them."""
    versions = workflow_versions(factory)
    rng = random.Random("%s:assign:%s" % (seed, repo.name))
    def usable(text):
        return 12 <= len(text) <= 2400 and mentioned_paths_exist(text, files)

    # Follow-ups pass the same gate as the request; a task keeps only those that do.
    kept = [dict(t, follow_ups=[f for f in t["follow_ups"] if usable(f)]) for t in raw if usable(t["prompt"])]
    groups = group_prompts([t["prompt"] for t in kept])
    generators = [m.id for m in factory.generators()]
    archetypes = {a.id: a for a in factory.archetypes}
    seen_groups = set()
    tasks = []
    for t, g in zip(kept, groups, strict=True):
        group_key = "%s:%s" % (repo.name, hashlib.sha1(kept[g]["prompt"].encode()).hexdigest()[:12])
        if group_key in seen_groups:
            continue  # a near-duplicate adds no diversity
        seen_groups.add(group_key)
        arch = archetypes[t["archetype"]]
        names = sorted(arch.workflows)
        pick = rng.choices([None, *names], [1 - sum(arch.workflows.values()), *(arch.workflows[n] for n in names)])[0]
        follow = t["follow_ups"] if rng.random() < arch.follow_up_rate else []
        tid = hashlib.sha1(("%s\n%s" % (repo.name, t["prompt"])).encode()).hexdigest()[:16]
        model = rng.choice(generators)
        task = {
            "id": tid, "prompt": t["prompt"], "follow_ups": [{"prompt": f} for f in follow],
            "provider_id": provider_id(model), "model": model,
            "meta": {"task_id": tid, "repo": repo.name, "split": repo.split, "archetype": arch.id, "prompt_group": group_key,
                     "lang": t["lang"], "author_model": t["author"], "workflow": pick},
        }
        if pick:
            task["workflow"] = pick
            task["workflow_version"] = versions[pick]
            task["meta"]["workflow_version"] = versions[pick]
        tasks.append(task)
    with open(out_path, "w", encoding="utf-8") as fh:
        for task in tasks:
            fh.write(json.dumps(task, ensure_ascii=False) + "\n")
    return len(tasks)


def rebase(factory, src_dir, out_dir, skip_ids):
    """Copy a task set for another run: drop tasks already settled, and move each task's
    driving model to the generator at the same position in the configuration, so a
    change of generators keeps the balance across tasks."""
    src_dir, out_dir = Path(src_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    generators = [m.id for m in factory.generators()]
    counts = {}
    for path in sorted(src_dir.glob("*.jsonl")):
        tasks = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        before = sorted({t["model"] for t in tasks})
        mapping = {old: generators[i % len(generators)] for i, old in enumerate(before)}
        kept = []
        for t in tasks:
            if t["id"] in skip_ids:
                continue
            t["model"] = mapping[t["model"]]
            t["provider_id"] = provider_id(t["model"])
            kept.append(json.dumps(t, ensure_ascii=False))
        (out_dir / path.name).write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")
        counts[path.stem] = len(kept)
    return counts
