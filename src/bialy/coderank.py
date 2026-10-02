"""Training pairs for the code-rank head.

The code-rank head reranks code units (definitions, repomap tags, search hits)
against a request. Its data starts from the units Painted Wolf Code's own
parsers harvest from each pinned repository (`decide-rerank harvest`). A pair
is a request whose answer is one unit, from two sources:

  docs    the unit's leading comment as the request: human-written, free,
          English.
  model   a pinned open-weights generator writes a request for the unit in a
          language and register drawn from a panel, plus the one-to-three-word
          query a person would type into project search.

A request that shares a word with its unit's identifier or file name is
dropped from both sources: text matching already ranks that unit first, so the
pair would teach the head nothing.

`decide-rerank eval --no-engine` then turns each repository's pairs into the
candidate sets a site would rank (the dumps `train_rerank.py` reads). Held-out
repositories contribute evaluation pairs only.
"""

import concurrent.futures
import json
import os
import random
import re
import subprocess
from pathlib import Path

LANGUAGES = ["en", "en", "en", "de", "es", "fr", "pt", "ja", "zh", "ko"]
REGISTERS = [
    "a short question a developer asks a coding assistant",
    "one terse imperative line, lowercase, no punctuation",
    "two sentences with a little context about what they are working on",
    "a bug report style request naming the behaviour, not the code",
]
SITES = ("summarize_definitions", "summarize_structure", "repomap_tags", "project_search")

SYSTEM = """You write the requests a software developer would type into a coding assistant while working inside a codebase.
Each request must be something a real person would write to find or understand the code unit shown, in the requested language and register.
Never use the unit's identifier, its file name, or any token of them. Describe purpose and behaviour, not names."""

REPLY_SCHEMA = {
    "type": "object", "additionalProperties": False, "required": ["task", "query"],
    "properties": {"task": {"type": "string"}, "query": {"type": "string"}},
}


def read_jsonl(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def write_jsonl(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def ident_tokens(symbol):
    """Lowercase tokens of an identifier: camelCase, snake_case, and the whole."""
    parts = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", symbol)
    parts = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", parts)
    toks = {t.lower() for t in re.split(r"[^A-Za-z0-9]+", parts) if len(t) > 2}
    toks.add(symbol.lower())
    return toks


def mentions_identifier(text, unit):
    words = {w.lower() for w in re.findall(r"[A-Za-z0-9_]+", text)}
    banned = ident_tokens(unit["symbol"]) | ident_tokens(os.path.basename(unit["file"]).split(".")[0])
    return bool(words & banned)


def prose(text):
    """True when the text reads as words rather than code."""
    tokens = text.split()
    if len(tokens) < 3:
        return False
    wordy = sum(1 for t in tokens if re.fullmatch(r"[A-Za-z][A-Za-z'’-]*[.,;:!?]?", t))
    return wordy / len(tokens) >= 0.8 and not re.search(r"[{}();=<>\[\]]", text)


def pair(unit, task, query, lang, source):
    return {"repo": unit["repo"], "file": unit["file"], "line": unit["line"], "symbol": unit["symbol"],
            "task": task, "query": query, "lang": lang, "source": source}


def doc_pairs(units, min_words=6):
    """The leading comment's first sentence as the request, when it reads as prose and
    names neither the unit nor its file."""
    out = []
    for u in units:
        doc = (u.get("doc") or "").strip()
        if not doc:
            continue
        first = re.split(r"(?<=[.!?])\s", doc, maxsplit=1)[0].strip()
        first = re.sub(r"^" + re.escape(u["symbol"]) + r"\s*", "", first)
        if not prose(first) or mentions_identifier(first, u) or len(first.split()) < min_words:
            continue
        query = " ".join([w for w in re.findall(r"[A-Za-z][A-Za-z0-9]+", first.lower()) if len(w) > 3][:3])
        out.append(pair(u, first, query, "en", "docs"))
    return out


def unit_prompt(unit, lang, register):
    return (f"Language: {lang}\nRegister: {register}\n\n"
            f"Code unit ({unit['lang']}, kind {unit['kind']}, in {unit['file']}):\n{unit['code'][:900]}\n"
            + (f"Comment above it: {unit['doc'][:300]}\n" if unit.get("doc") else "")
            + "\nReply with `task`: the request, in that language and register; and `query`: one to three "
              "English words a person would type into project search to find this.")


def model_pair(chat, unit, lang, register, seed):
    """One generated pair, marked when its request names the unit, or None for an empty reply."""
    reply = chat.json(SYSTEM, unit_prompt(unit, lang, register), temperature=0.8, seed=seed, thinking=False,
                      max_tokens=400, schema=REPLY_SCHEMA)
    task, query = str(reply.get("task", "")).strip(), str(reply.get("query", "")).strip()
    if not task:
        return None
    return dict(pair(unit, task, query, lang, "model:" + chat.model), names_unit=mentions_identifier(task, unit))


def endpoints(factory, writer=None):
    """Every endpoint of the writer, or of every generator, so requests spread over replicas."""
    models = [writer] if writer else factory.generators()
    return [chat for m in models for chat in factory.chats(m)]


def model_pairs(factory, units, count, seed=7, workers=64, writer=None):
    """`count` generated pairs whose requests do not name their unit, from a seeded order
    of `units`, and every pair along the way whose request does.

    Each unit's language, register, and generator come from the seed. The prompt cannot
    stop a reply from naming its unit, so named replies are kept apart rather than
    prevented. Units are drawn in waves until `count` unnamed pairs exist or the units
    run out; the named pairs returned are those drawn up to the last unnamed one kept."""
    rng = random.Random(seed)
    sample = [u for u in units if u.get("code")]
    rng.shuffle(sample)
    chats = endpoints(factory, writer)
    jobs = [(u, rng.choice(LANGUAGES), rng.choice(REGISTERS), rng.randrange(1 << 30), chats[i % len(chats)])
            for i, u in enumerate(sample)]
    kept, cursor = {}, 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        while cursor < len(jobs):
            short = count - sum(not r["names_unit"] for r in kept.values())
            wave = range(cursor, min(len(jobs), cursor + max(2 * short, workers)))
            futures = {i: pool.submit(model_pair, jobs[i][4], *jobs[i][:4]) for i in wave}
            for i, fut in futures.items():
                try:
                    row = fut.result()
                except Exception as exc:  # a malformed reply loses one pair, not the repository
                    print("pair failed: %s" % exc, flush=True)
                    continue
                if row:
                    kept[i] = row
            cursor = wave.stop
            if sum(not r["names_unit"] for r in kept.values()) >= count:
                break
    ordered = [kept[i] for i in sorted(kept)]
    unnamed = [r for r in ordered if not r["names_unit"]][:count]
    last = ordered.index(unnamed[-1]) if len(unnamed) == count else len(ordered) - 1
    return unnamed, [r for r in ordered[:last + 1] if r["names_unit"]]


def harvest(factory, decide_rerank, out_dir):
    """Units for every pinned repository, from its seed clone at the pinned commit."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    counts = {}
    for repo in factory.repos:
        target = out_dir / (repo.name + ".jsonl")
        subprocess.run([decide_rerank, "harvest", "--repo", str(factory.root / "repos" / repo.name), "--name", repo.name,
                        "--out", str(target)], check=True, capture_output=True)
        counts[repo.name] = sum(1 for _ in open(target, encoding="utf-8"))
    return counts


def pairs(factory, units_dir, out_dir, per_repo, seed=7, writer=None):
    """Doc and generated pairs for every repository; held-out repositories get the same
    sources so they can measure both."""
    units_dir, out_dir = Path(units_dir), Path(out_dir)
    counts = {}
    for repo in factory.repos:
        units = read_jsonl(units_dir / (repo.name + ".jsonl"))
        docs = doc_pairs(units)
        generated, named = model_pairs(factory, units, per_repo, seed=hash_seed(seed, repo.name), writer=writer)
        write_jsonl(out_dir / ("docs-%s.jsonl" % repo.name), docs)
        write_jsonl(out_dir / ("model-%s.jsonl" % repo.name), generated)
        # Requests that name their unit: realistic, and for measuring the blend, not training.
        write_jsonl(out_dir / ("model-named-%s.jsonl" % repo.name), named)
        counts[repo.name] = {"docs": len(docs), "model": len(generated), "model_named": len(named), "split": repo.split}
        print(repo.name, counts[repo.name], flush=True)
    return counts


def hash_seed(seed, name):
    return random.Random("%s:%s" % (seed, name)).randrange(1 << 30)


def dumps(factory, decide_rerank, units_dir, pairs_dir, out_dir):
    """Every site's lexical candidate sets for every repository's pairs, without an engine:
    the rows `train_rerank.py` labels from the site's own structure."""
    units_dir, pairs_dir, out_dir = Path(units_dir), Path(pairs_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = [(repo, site, source) for repo in factory.repos for site in SITES for source in ("docs", "model")]

    def one(job):
        repo, site, source = job
        dump = out_dir / ("%s-%s-%s.jsonl" % (site, source, repo.name))
        result = subprocess.run([decide_rerank, "eval", "--no-engine", "--site", site, "--repo", str(factory.root / "repos" / repo.name),
                                 "--name", repo.name, "--units", str(units_dir / (repo.name + ".jsonl")),
                                 "--pairs", str(pairs_dir / ("%s-%s.jsonl" % (source, repo.name))),
                                 "--dump", str(dump), "--json", str(dump.with_suffix(".report.json"))],
                                capture_output=True, text=True)
        return job, result.returncode, (result.stdout + result.stderr).strip().splitlines()[-1:] or [""]

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        for (repo, site, source), code, tail in pool.map(one, jobs):
            print("%s %s %s exit=%d %s" % (repo.name, site, source, code, tail[0][:120]), flush=True)
