"""Score skill and tool cards 0..4 against a turn's request, with an open-weights judge.

What a session did labels only a few cards: a turn reads at most one skill, and
a request_tools need names a few tools. The judge scores every skill card and
every offered loadable tool card against each completed turn's request, and
every other loadable tool card against each request_tools need. Tools a
session called are not labels on their own: driving models make habitual
orientation calls, and a head trained on them learns the habit.

A row is judged by a model of another family than the one that drove its
session. Decoding is greedy with a fixed seed, and candidates are shown in an
order shuffled per row and unit, recorded on the row, so position bias
averages out across the set instead of favouring the catalog's first cards.
A sample of rows is judged again by the other family to report agreement.
"""

import concurrent.futures
import hashlib
import json
import random
import threading
from pathlib import Path

from .llm import account_failure

SYSTEM = """You judge which skills and tools a coding assistant will need to carry out a developer's request.
For each numbered candidate you are given its description. Score relevance from 0 to 4: 0 the request will never need it, 1 unlikely, 2 possible, 3 likely, 4 certain.
A skill is a procedure the assistant reads before working; score it by whether following it would carry out this request, not by shared words.
Judge from the request alone, as the assistant's first step, before any tool has run.
Reply with one JSON object only, mapping every candidate number to its score: {"scores": {"1": 0, "2": 3, ...}}"""


SKILL_SYSTEM = """You judge which skill procedures are relevant to carrying out a developer's request. Each numbered candidate has a description. Evaluate every candidate independently against its described purpose.
Score 0 unrelated, 1 unlikely to help this work, 2 possibly useful but incidental or uncertain, 3 likely useful for doing a requested operation, 4 directly matches the work or a necessary part of completing it.
A direct match can be implicit in the requested operation; the developer need not name the skill or every necessary step. Several specialized procedures can be relevant to one request. Do not suppress a direct match because another skill also applies.
General advice that could accompany almost any edit is not by itself a likely match. For a general verification procedure, distinguish making a routine edit from a request whose work includes testing, diagnosing a failure, confirming behavior, checking a build, or demonstrating correctness. Judge other procedures on their own descriptions, not this verification distinction.
Use only the request and the candidate descriptions, not assumptions about future unrelated work. Shared words alone do not establish relevance. A request may have no relevant skills.
Reply with one JSON object only, mapping every candidate number to its score: {"scores": {"1": 0, "2": 3, ...}}"""


TOOL_SYSTEM = """You judge which additional tools a coding assistant will need to carry out a developer's request.
The assistant already has the tools listed as available. Each numbered candidate is a tool it would have to load first; you are given its description.
Score each candidate from 0 to 4: 0 the request will never need it, 1 unlikely, 2 possible, 3 likely, 4 certain.
Score 3 or 4 only when carrying out the request will likely call that tool itself, not an available one. Where several candidates do the same job, score only the one the request most directly calls for above 2.
Judge from the request alone, as the assistant's first step, before any tool has run.
Reply with one JSON object only, mapping every candidate number to its score: {"scores": {"1": 0, "2": 3, ...}}"""


def order_seed(row, unit):
    return int(hashlib.sha256(("%s:%s:%s" % (row["session"], row["receipt"], unit)).encode()).hexdigest()[:8], 16)


def score_cards(chat, task, cards, seed, system=SYSTEM, available=()):
    """{name: level} for every card, shown in a seeded shuffled order."""
    names = sorted(cards)
    random.Random(seed).shuffle(names)
    listing = "\n\n".join("%d. %s" % (i + 1, cards[name]) for i, name in enumerate(names))
    context = "\n\nAvailable: %s" % ", ".join(available) if available else ""
    user = "Request:\n%s%s\n\nCandidates:\n\n%s" % (task, context, listing)
    numbers = {str(i + 1): {"type": "integer", "minimum": 0, "maximum": 4} for i in range(len(names))}
    schema = {"type": "object", "required": ["scores"], "additionalProperties": False,
              "properties": {"scores": {"type": "object", "required": list(numbers), "additionalProperties": False, "properties": numbers}}}
    # The verdict is decoded straight into the schema, without a reasoning phase: scoring
    # sixty cards after reasoning can exhaust the output budget before any score appears.
    value = chat.json(system, user, temperature=0.0, seed=seed, schema=schema, thinking=False)
    raw = value.get("scores", value) if isinstance(value, dict) else {}
    out = {}
    for i, name in enumerate(names):
        level = raw.get(str(i + 1))
        if isinstance(level, (int, float)) and 0 <= level <= 4:
            out[name] = int(round(level))
    if len(out) < len(names) * 0.9:
        raise ValueError("judge scored %d of %d candidates" % (len(out), len(names)))
    return out


UNITS = ("skills", "tools", "requests")


def attempt(score, tries=3):
    """A judge call, tried again after a failure: most failures (rate limits, a malformed
    verdict) pass on a second try."""
    for n in range(tries):
        try:
            return score()
        except Exception as exc:
            if account_failure(exc) or n == tries - 1:
                raise


def judge_row(chats, row, skill_cards, tool_cards, judge_model, units=UNITS, lenient=False):
    """Score the row's units with one judge. Strict, a failed call fails the row; lenient,
    it leaves that piece (a unit, or one need) unscored, listed in judge.missed."""
    # Rows spread over every server of the judge model; the weights are the same on each.
    servers = chats[judge_model]
    chat = servers[order_seed(row, "server") % len(servers)]
    labels = row["labels"]
    loadable = [t for t in row["offered"]["loadable"] if t in tool_cards]
    missed = []

    def piece(name, score, store, clear):
        try:
            store(attempt(score))
        except Exception as exc:
            if account_failure(exc) or not lenient:
                raise
            clear()
            missed.append(name)

    if "skills" in units and not row["partial"] and skill_cards:
        piece("skills", lambda: score_cards(chat, row["state"]["user"], skill_cards, order_seed(row, "skills"), system=SKILL_SYSTEM),
              lambda v: labels.__setitem__("skill_scores", v), lambda: labels.pop("skill_scores", None))
    if "tools" in units and not row["partial"] and loadable:
        piece("tools", lambda: score_cards(chat, row["state"]["user"], {t: tool_cards[t] for t in loadable}, order_seed(row, "tools"),
                                           system=TOOL_SYSTEM, available=row["offered"]["floor"]),
              lambda v: labels.__setitem__("tool_scores", v), lambda: labels.pop("tool_scores", None))
    if "requests" in units:
        for n, request in enumerate(labels["requests"]):
            rest = {t: tool_cards[t] for t in loadable if t not in set(request["exact"])}
            if rest:
                piece("need%d" % n, lambda need=request["need"], rest=rest, n=n: score_cards(chat, need, rest, order_seed(row, "need%d" % n)),
                      lambda v, request=request: request.__setitem__("scores", v), lambda request=request: request.__setitem__("scores", None))
    earlier = (row.get("judge") or {}).get("units", []) if set(units) != set(UNITS) else []
    judged = sorted(set(earlier) | set(units), key=UNITS.index)
    reasoning = "%s reasoning effort" % (chat.reasoning_effort or "low") if chat.hosted else "no reasoning"
    row["judge"] = {"model": judge_model, "units": judged, "decoding": "greedy, seeded order, schema-constrained, %s" % reasoning}
    if missed:
        row["judge"]["missed"] = missed
    return row


def signature(factory, corpus, rows_in, units, fraction):
    """What a run's labels depend on: its inputs, units, judges, and prompts. A checkpoint
    resumes only under the same signature."""
    prompts = hashlib.sha256((SYSTEM + SKILL_SYSTEM + TOOL_SYSTEM).encode()).hexdigest()
    with open(rows_in, "rb") as fh:
        inputs = hashlib.sha256(fh.read()).hexdigest()
    judges = [{"id": m.id, "revision": m.revision, "hosted": (m.hosted or {}).get("model")} for m in factory.models if "judge" in m.roles]
    return {"inputs_sha256": inputs, "units": list(units), "judges": judges, "prompts_sha256": prompts,
            "second_fraction": fraction, "corpus_revision": corpus.get("catalog_revision")}


def row_key(row):
    return "%s:%s" % (row["session"], row["receipt"])


def run(factory, corpus, rows_in, rows_out, workers=48, second=None, units=UNITS, rounds=4):
    """Judge every row on `units`, keeping labels from units judged before; write judged
    rows, and the second-judge sample beside them (<rows_out>.agreement.jsonl, one line
    per doubly judged card) so the agreement a release reports can be recomputed from
    the release itself. A doubly judged row also carries the second judge's scores
    (labels.second_scores).

    Each judged row is appended to <rows_out>.partial.jsonl as it finishes, under a
    manifest (<rows_out>.manifest.json) naming the run's signature; a run stopped part
    way resumes from there without judging a finished row again, and a rerun whose
    inputs, units, judges, or prompts differ is refused instead of mixing labels."""
    skill_cards = {s["name"]: s["card"] for s in corpus["skills"]}
    tool_cards = {t["name"]: t["card"] for t in corpus["tools"] if t.get("card")}
    chats = {m.id: factory.chats(m) for m in factory.models if "judge" in m.roles}
    rows = [json.loads(line) for line in open(rows_in, encoding="utf-8") if line.strip()]
    fraction = factory.judge["second_judge_fraction"] if second is None else second
    sig = signature(factory, corpus, rows_in, units, fraction)
    manifest_path, partial_path = Path(str(rows_out) + ".manifest.json"), Path(str(rows_out) + ".partial.jsonl")
    manifest = {"signature": sig, "attempts": 0, "completed": False}
    done = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["signature"] != sig:
            raise ValueError("%s was started with different inputs, units, judges, or prompts; write to a new path" % rows_out)
        if partial_path.exists():
            for line in partial_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    entry = json.loads(line)
                    done[entry["key"]] = entry
    manifest["attempts"] += 1
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    failures = 0
    stop = threading.Event()
    fatal_error = None

    def score_row(row):
        primary = factory.judge_for(row["model"]).id
        # A pass over some units keeps the others' labels, so it must be the same judge.
        if set(units) != set(UNITS) and (row.get("judge") or {}).get("model", primary) != primary:
            raise ValueError("row %s was judged by %s, not %s" % (row["session"], row["judge"]["model"], primary))
        untouched = json.loads(json.dumps(row))
        out = judge_row(chats, row, skill_cards, tool_cards, primary, units)
        other = None
        if random.Random(order_seed(row, "second")).random() < fraction:
            others = [m.id for m in factory.models if "judge" in m.roles and m.id != primary]
            if others:
                other = judge_row(chats, untouched, skill_cards, tool_cards, others[0], units, lenient=True)
        return out, other

    def work(row):
        if stop.is_set():
            raise concurrent.futures.CancelledError()
        try:
            return score_row(row)
        except Exception as exc:
            if account_failure(exc):
                stop.set()
            raise

    # A row that fails (a rate limit, an unparseable verdict) is tried again in the next
    # round; only a row that fails every round is left out, and a resumed run tries it again.
    pending = [r for r in rows if row_key(r) not in done]
    with open(partial_path, "a", encoding="utf-8") as checkpoint:
        for _ in range(rounds):
            failed = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {pool.submit(work, r): r for r in pending}
                for fut in concurrent.futures.as_completed(futures):
                    try:
                        row, other = fut.result()
                    except concurrent.futures.CancelledError:
                        continue
                    except Exception as exc:
                        failed.append(futures[fut])
                        if account_failure(exc):
                            fatal_error = str(exc)[:200]
                        print("judge failed: %s" % str(exc)[:200], flush=True)
                        continue
                    found = []
                    if other is not None:
                        row["labels"]["second_scores"] = merged_second(row["labels"].get("second_scores"), second_scores(other, units), units)
                        found = agreement_records(row, other, units)
                    entry = {"key": row_key(row), "row": row, "records": found}
                    checkpoint.write(json.dumps(entry, ensure_ascii=False) + "\n")
                    checkpoint.flush()
                    done[entry["key"]] = entry
            pending = [r for r in rows if row_key(r) not in done] if stop.is_set() else failed
            if not pending or stop.is_set():
                break
    failures = len(pending)
    judged = sorted((e["row"] for e in done.values()), key=lambda r: (r["root_session"], r["receipt"]))
    records = [rec for e in done.values() for rec in e["records"]]
    pairs = [(r["first_score"], r["second_score"]) for r in records]
    with open(rows_out, "w", encoding="utf-8") as fh:
        for row in judged:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    records.sort(key=lambda r: (r["session"], r["receipt"], r["unit"], r["candidate"]))
    with open(str(rows_out) + ".agreement.jsonl", "w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    manifest.update(completed=failures == 0, rows=len(judged), failed=failures, fatal_error=fatal_error)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return {"rows": len(judged), "failed": failures, "second_judge_pairs": len(pairs), "weighted_kappa": weighted_kappa(pairs),
            "attempts": manifest["attempts"], "fatal_error": fatal_error}


def second_scores(b, units=UNITS):
    """The second judge's scores on a row, kept beside the first judge's so a consumer can
    require both to agree; a piece it could not score is absent (a need is null) and
    listed in `missed`."""
    labels = b["labels"]
    out = {"model": b["judge"]["model"], "missed": b["judge"].get("missed", [])}
    if "skills" in units and labels.get("skill_scores"):
        out["skills"] = labels["skill_scores"]
    if "tools" in units and labels.get("tool_scores"):
        out["tools"] = labels["tool_scores"]
    if "requests" in units:
        out["requests"] = [r.get("scores") for r in labels["requests"]]
    return out


def merged_second(earlier, now, units):
    """A pass over some units replaces only those units' second scores; the rest, and what
    the second judge missed in them, stay as an earlier pass left them."""
    if not earlier or set(units) == set(UNITS) or earlier.get("model") != now["model"]:
        return now
    keep = {"skills": ("skills",), "tools": ("tools",), "requests": ("requests",)}
    out = dict(now)
    for unit in UNITS:
        if unit not in units:
            for key in keep[unit]:
                if key in earlier:
                    out[key] = earlier[key]
    kept_missed = [m for m in earlier.get("missed", []) if (m if not m.startswith("need") else "requests") not in units]
    out["missed"] = kept_missed + now.get("missed", [])
    return out


def agreement_records(a, b, units=UNITS):
    """One record per card both judges scored on the same row, over the units this pass
    judged: the second judge's copy carries the first judge's earlier labels."""
    base = {"session": a["session"], "receipt": a["receipt"], "first_judge": a["judge"]["model"], "second_judge": b["judge"]["model"]}
    out = []
    for unit, key in (("skills", "skill_scores"), ("tools", "tool_scores")):
        if unit not in units:
            continue
        sa, sb = a["labels"].get(key) or {}, b["labels"].get(key) or {}
        out += [dict(base, unit=unit, candidate=k, first_score=sa[k], second_score=sb[k]) for k in sorted(sa) if k in sb]
    for n, (ra, rb) in enumerate(zip(a["labels"]["requests"], b["labels"]["requests"], strict=True)):
        if "requests" not in units:
            break
        xa, xb = ra.get("scores") or {}, rb.get("scores") or {}
        out += [dict(base, unit="need%d" % n, candidate=k, first_score=xa[k], second_score=xb[k]) for k in sorted(xa) if k in xb]
    return out


def weighted_kappa(pairs, levels=5):
    """Quadratic-weighted Cohen's kappa over paired 0..levels-1 scores."""
    if not pairs:
        return None
    n = len(pairs)
    observed = [[0.0] * levels for _ in range(levels)]
    for x, y in pairs:
        observed[x][y] += 1
    rows = [sum(r) for r in observed]
    cols = [sum(observed[i][j] for i in range(levels)) for j in range(levels)]
    num = den = 0.0
    for i in range(levels):
        for j in range(levels):
            w = ((i - j) ** 2) / ((levels - 1) ** 2)
            num += w * observed[i][j]
            den += w * rows[i] * cols[j] / n
    return round(1 - num / den, 4) if den else None
