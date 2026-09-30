"""Skill-ranking requests: rows that teach the rank head which skill a request needs.

Sessions label few skills: a turn reads at most one, and the common skills
(verifying a change, reading history) crowd out the specialised ones, so the
head ranks generic skills first even when a specialised one fits. A writer
model drafts requests for every skill in families of four kinds:

  clear     the skill's procedure is plainly what the request needs
  nearmiss  the request sounds close to the skill but needs something else
  multi     the request needs the skill and one other named skill
  none      a request no skill helps with

No request names any skill; the judges then score every skill card against
each request (`bialy judge --unit skills`), so a label never comes from the
writer's intent. A family is one writer call, and whole families go to
training or evaluation, so paraphrases never cross the split.
"""

import concurrent.futures
import hashlib
import json
import random
import re
from pathlib import Path

from .coderank import LANGUAGES, REGISTERS
from .tasks import LANGUAGE_NAMES, clean

SYSTEM = """You write realistic requests that developers type to an AI coding assistant working in their repository.
Write the way working engineers do: some terse, some detailed, some with pasted output, some polite, some blunt.
Never name a skill, procedure, or tool, and never quote the descriptions you are given; describe the work itself.
Reply with one JSON object only: {"requests": ["...", ...]}."""

REPLY_SCHEMA = {"type": "object", "required": ["requests"], "additionalProperties": False,
                "properties": {"requests": {"type": "array", "items": {"type": "string"}}}}

KINDS = ("clear", "nearmiss", "multi")


def brief(kind, skill, other, count):
    """The writer's instruction for one family."""
    if kind == "clear":
        return ("Write %d different requests for which the following procedure is plainly the right way to do the work:\n%s"
                % (count, skill["description"]))
    if kind == "nearmiss":
        return ("Write %d different requests that touch the same area as the following procedure but that it would not help with; "
                "each should need something else, such as this other procedure or none at all:\nNot this: %s\nPerhaps this: %s"
                % (count, skill["description"], other["description"]))
    if kind == "multi":
        return ("Write %d different requests that need both of the following procedures to carry out:\n1. %s\n2. %s"
                % (count, skill["description"], other["description"]))
    return ("Write %d different requests that no special procedure helps with: quick questions, small edits, explanations, "
            "or everyday changes an assistant does directly." % count)


def names_skill(text, name):
    """The request spells out the skill's name, with hyphens or spaces."""
    words = name.split("-")
    return bool(re.search(r"\b" + r"[\s-]+".join(map(re.escape, words)) + r"\b", text, re.IGNORECASE))


def offered_from(path):
    """The candidates of the first complete coordinator turn in a rows file."""
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            if r["host"] == "coordinator" and not r["partial"] and r["offered"]["loadable"]:
                return r["offered"]
    raise ValueError("%s has no complete coordinator turn" % path)


def family_id(seed, skill, kind, n):
    return hashlib.sha256(("%s:%s:%s:%d" % (seed, skill, kind, n)).encode()).hexdigest()[:16]


def row(request, fam, i, writer, repo, lang, revision, meta, offered=None):
    """A pw-decide-row/1 row the judge and the rank trainer read: one coordinator turn with
    the request as its only state, and no behaviour. `offered` carries a real turn's
    candidates when the row's tools are to be judged too."""
    session = "skillreq-" + fam
    return {
        "schema": "pw-decide-row/1", "receipt": i, "session": session, "root_session": session, "opening_message_id": "%s-%d" % (fam, i),
        "host": "coordinator", "surface": "implement_investigate", "catalog_revision": revision, "project": repo.name, "model": writer,
        "partial": False, "lang": lang, "engine": {"state": "abstained", "reason": "generated request", "preloaded": [], "omitted": []},
        "offered": offered or {"floor": [], "loadable": [], "guides": []},
        "state": {"host": "coordinator", "user": request, "surface": "implement_investigate", "posture": "build", "root_count": 1, "workers_in_flight": 0},
        "labels": {"tools": [], "requests": [], "requested_names": [], "requested_groups": [], "skills": [], "kind": None, "guides": {}},
        "meta": dict(meta, source="skillreq", family=fam, repo=repo.name),
    }


def plan(factory, skills, families, none_families, seed, only=(), repos=()):
    """Every family to write: (skill, kind, other skill, family index, repo, language, register)."""
    rng = random.Random("%s:plan" % seed)
    pool = [r for r in factory.repos if not repos or r.name in repos]
    names = sorted(skills)
    out = []
    for name in names:
        if only and name not in only:
            continue
        for kind in KINDS:
            for n in range(families[kind]):
                other = skills[rng.choice([m for m in names if m != name])]
                out.append((name, kind, other["name"], n, rng.choice(pool), rng.choice(LANGUAGES), rng.choice(REGISTERS)))
    for n in range(none_families):
        out.append(("", "none", "", n, rng.choice(pool), rng.choice(LANGUAGES), rng.choice(REGISTERS)))
    return out


def write(factory, corpus, out_path, writer, families, none_families, per_family, seed=7, eval_fraction=0.2, only=(), workers=16,
          repos=(), split=None, offered=None):
    """Write the families to `out_path` as rows, each with meta.split train or eval, or all
    with `split` (an acceptance set written after the selection rules were fixed)."""
    skills = {s["name"]: s for s in corpus["skills"]}
    model = next(m for m in factory.models if m.id == writer)
    chat = factory.chats(model)[0]
    jobs = plan(factory, skills, families, none_families, seed, only, repos)

    def work(job):
        name, kind, other, n, repo, lang, register = job
        fam = family_id(seed, name or "none", kind, n)
        language = "Write in %s." % LANGUAGE_NAMES[lang] if lang != "en" else "Write in English."
        user = "The developer works in %s (%s). Style: %s.\n%s\n%s" % (
            repo.name, repo.language, register, brief(kind, skills.get(name), skills.get(other), per_family), language)
        value = chat.json(SYSTEM, user, temperature=0.9, seed=int(fam[:8], 16), thinking=False, max_tokens=4096, schema=REPLY_SCHEMA)
        requests = [clean(r) for r in (value.get("requests") or []) if isinstance(r, str) and 12 <= len(r.strip()) <= 1500]
        requests = [r for r in requests if not any(names_skill(r, n) for n in skills)]
        chosen = split or ("eval" if int(fam[8:12], 16) / 0xFFFF < eval_fraction else "train")
        meta = {"skill": name or None, "other": other or None, "kind": kind, "split": chosen, "writer": writer}
        return [row(r, fam, i, writer, repo, lang, corpus["catalog_revision"], meta, offered) for i, r in enumerate(requests)]

    rows, failures = [], 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        for fut in concurrent.futures.as_completed([pool.submit(work, j) for j in jobs]):
            try:
                rows += fut.result()
            except Exception as exc:  # a malformed reply loses one family, not the run
                failures += 1
                print("family failed: %s" % str(exc)[:160], flush=True)
    rows.sort(key=lambda r: (r["meta"]["family"], r["receipt"]))
    Path(out_path).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    kinds = {}
    for r in rows:
        kinds[r["meta"]["kind"]] = kinds.get(r["meta"]["kind"], 0) + 1
    return {"rows": len(rows), "families": len(jobs), "failed": failures, "kinds": kinds,
            "eval_rows": sum(1 for r in rows if r["meta"]["split"] == "eval")}
