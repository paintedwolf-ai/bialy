"""bialy: the decision-head dataset factory.

  bialy repos fetch|verify
  bialy serve start|stop|status
  bialy tasks --out DIR [--seed N]
  bialy fleet image --lycaon-bin DIR --engine DIR [--tag NAME --decide-deadline-ms N --config-dir DIR]
  bialy fleet warm [--image TAG]
  bialy fleet plan --run NAME --tasks DIR [--pilot DIR] [--repo NAME ...] [--image TAG --lycaon-bin DIR]
  bialy fleet run --run NAME [--runners N] [--pilot DIR --image NAME]
  bialy fleet replan --run NAME --tasks DIR [--pilot DIR]
  bialy fleet reap --run NAME
  bialy fleet salvage --run NAME --into NAME --tasks DIR --out DIR
  bialy coderank harvest --decide-rerank BIN --out DIR
  bialy coderank pairs --units DIR --out DIR [--per-repo N --seed N]
  bialy coderank dumps --decide-rerank BIN --units DIR --pairs DIR --out DIR
  bialy collect --run NAME --pass NAME --tasks DIR
  bialy judge --corpus FILE --rows FILE --out FILE [--unit skills|tools|requests ...]
  bialy split --out DIR ROWS...
  bialy release --split DIR --version V --code-ref REF --out DIR --schema FILE --corpus FILE --driven FILE ...
                   [--agreement FILE ...] [--coderank DIR] [--stage FILE ...] [--stopping TEXT]
  bialy release-heads --heads DIR --version V --dataset-version V --engine-commit SHA --out DIR --eval SET=FILE ...
                         [--baseline SET=FILE ... --baseline-label TEXT] [--rerank NAME=FILE ...]
  bialy publish dataset|heads --release DIR --version V [--push]
  bialy fetch dataset|heads --version V --out DIR
  bialy verify dataset|heads --release DIR [--unanchored]
  bialy audit dataset --release DIR [--unanchored] [--rejudge N --endpoint URL --model ID]
  bialy audit heads --release DIR --dataset DIR --lycaon CHECKOUT --engine LAUNCHER
  bialy check
  bialy run --run NAME [--until STAGE] [--from STAGE] [--redo STAGE ...] [--dry-run] [--push] [--rebuild-image]
            [--task-cap N] [--repo NAME ...] [--runners N] [--epochs N] [--no-engine-on]
  bialy run-status --run NAME

Runs live under <root>/runs/<name>; the root and everything else come from config/.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

from . import audit, coderank, config, fleet, heads, hub, judge, provenance, release, repos, serve, skillreq, split, tasks
from . import run as runmod


def main(argv=None):
    ap = argparse.ArgumentParser(prog="bialy", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("repos")
    p.add_argument("action", choices=("fetch", "verify"))
    p = sub.add_parser("serve")
    p.add_argument("action", choices=("start", "stop", "status"))
    p = sub.add_parser("tasks")
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=7)
    p = sub.add_parser("fleet")
    p.add_argument("action", choices=("image", "warm", "plan", "replan", "run", "reap", "salvage"))
    p.add_argument("--into", help="salvage: run name that keeps the live containers' rows")
    p.add_argument("--out", help="salvage: task directory for the tasks still to drive")
    p.add_argument("--run")
    p.add_argument("--tasks")
    p.add_argument("--repo", action="append")
    p.add_argument("--runners", type=int)
    p.add_argument("--pilot", help="directory with bialy, model/, and heads/ for an engine-on pass")
    p.add_argument("--lycaon-bin")
    p.add_argument("--engine")
    p.add_argument("--tag", help="image tag (default: the configured runner image)")
    p.add_argument("--image", help="runner image for this run")
    p.add_argument("--decide-deadline-ms", type=int, help="raise every turn decision's deadline in the staged catalog")
    p.add_argument("--config-dir", help="the binaries' commit's lycaon/config tree, for --decide-deadline-ms")
    p = sub.add_parser("coderank")
    p.add_argument("action", choices=("harvest", "pairs", "dumps"))
    p.add_argument("--decide-rerank", help="the decide-rerank binary, built from the engine commit the runners carry")
    p.add_argument("--units")
    p.add_argument("--pairs")
    p.add_argument("--out", required=True)
    p.add_argument("--per-repo", type=int, default=1500, help="pairs: generated pairs per repository")
    p.add_argument("--seed", type=int, default=7)
    p = sub.add_parser("collect")
    p.add_argument("--run", required=True)
    p.add_argument("--pass", dest="pass_name", required=True)
    p.add_argument("--tasks", required=True, help="the task directory the run was planned from")
    p = sub.add_parser("judge")
    p.add_argument("--corpus", required=True)
    p.add_argument("--rows", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--workers", type=int, default=48)
    p.add_argument("--second-fraction", type=float, help="rows the second judge also scores (default: factory.yaml judge.second_judge_fraction)")
    p.add_argument("--unit", action="append", choices=judge.UNITS, help="judge only these units, keeping earlier labels (default: every unit)")
    p = sub.add_parser("skillreq")
    p.add_argument("--corpus", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--writer", required=True, help="a model with the writer role")
    p.add_argument("--families", default="clear=4,nearmiss=2,multi=2", help="families per skill of each kind")
    p.add_argument("--none-families", type=int, default=40)
    p.add_argument("--per-family", type=int, default=5)
    p.add_argument("--eval-fraction", type=float, default=0.2)
    p.add_argument("--skill", action="append", default=[], help="only these skills (a pilot)")
    p.add_argument("--repo", action="append", default=[], help="only these repositories")
    p.add_argument("--split", help="put every row in this split instead of dividing train and eval")
    p.add_argument("--offered-from", help="a rows file whose first coordinator turn's candidates every row offers, so its tools can be judged")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--workers", type=int, default=16)
    p = sub.add_parser("split")
    p.add_argument("--out", required=True)
    p.add_argument("rows", nargs="+")
    p = sub.add_parser("release")
    p.add_argument("--split", required=True)
    p.add_argument("--version", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--agreement", action="append", default=[], help="a judge pass's <rows>.agreement.jsonl second-judge sample; every pass the rows went through")
    p.add_argument("--schema", required=True, help="row.schema.json from the Painted Wolf Code commit that exported the rows")
    p.add_argument("--corpus", required=True, help="corpus.json from the same build")
    p.add_argument("--code-ref", required=True, help="the tag of this repository the release names, such as v1")
    p.add_argument("--coderank", help="the code-rank pairs directory (bialy coderank pairs)")
    p.add_argument("--stage", action="append", default=[], help="a stage's provenance.json, read into the release's recipe")
    p.add_argument("--driven", action="append", default=[], required=True, help="a run's tasks-driven.jsonl; every run the rows came from")
    p.add_argument("--stopping", default="", help="why generation stopped where it did, when it stopped before the plan ended")
    p = sub.add_parser("release-heads")
    p.add_argument("--heads", required=True, help="directory of trained .safetensors heads")
    p.add_argument("--version", required=True)
    p.add_argument("--dataset-version", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--engine-commit", required=True, help="the Painted Wolf Code commit the heads were trained and replayed with")
    p.add_argument("--eval", action="append", default=[], help="SET=replay_eval JSON for these heads")
    p.add_argument("--baseline", action="append", default=[], help="SET=replay_eval JSON for a baseline a reader can reproduce")
    p.add_argument("--baseline-label", default="base checkpoint", help="what the baseline rows are called on the card")
    p.add_argument("--rerank", action="append", default=[], help="NAME=decide-rerank eval JSON for the code-rank head")
    p = sub.add_parser("publish")
    p.add_argument("kind", choices=sorted(hub.KINDS))
    p.add_argument("--release", required=True)
    p.add_argument("--version", required=True)
    p.add_argument("--repo", help="override the repository in config/hub.yaml")
    p.add_argument("--push", action="store_true", help="upload; without it, list what would be sent")
    p = sub.add_parser("fetch")
    p.add_argument("kind", choices=sorted(hub.KINDS))
    p.add_argument("--version", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--repo")
    unanchored = "check a local build before its anchor is committed; a published release always has one"
    p = sub.add_parser("verify")
    p.add_argument("kind", choices=sorted(hub.KINDS))
    p.add_argument("--release", required=True)
    p.add_argument("--unanchored", action="store_true", help=unanchored)
    p = sub.add_parser("audit")
    p.add_argument("kind", choices=("dataset", "heads"))
    p.add_argument("--release", required=True)
    p.add_argument("--unanchored", action="store_true", help=unanchored)
    p.add_argument("--dataset", help="heads: the dataset release the heads trained on")
    p.add_argument("--lycaon", help="heads: a Painted Wolf Code checkout at the commit the heads name")
    p.add_argument("--rejudge", type=int, default=0, help="score this many released rows again through --endpoint")
    p.add_argument("--endpoint", help="OpenAI-compatible base URL serving --model")
    p.add_argument("--model", help="the pinned judge the endpoint serves, as rows name it")
    p.add_argument("--engine", help="heads: a bialy launcher that loads the released heads")
    sub.add_parser("check", help="validate the configuration and pins, offline")
    p = sub.add_parser("run", help="drive a whole pass on this machine, resumable; uploads only with --push")
    p.add_argument("--run", required=True)
    p.add_argument("--until", choices=runmod.STAGES, help="stop after this stage")
    p.add_argument("--from", dest="start", choices=runmod.STAGES, help="rerun from this stage, discarding later results")
    p.add_argument("--redo", action="append", choices=runmod.STAGES, default=[], help="rerun this stage alone, keeping the others (repeatable)")
    p.add_argument("--dry-run", action="store_true", help="print the stages this invocation would run")
    p.add_argument("--push", action="store_true", help="publish the dataset and heads at the end")
    p.add_argument("--rebuild-image", action="store_true")
    p.add_argument("--task-cap", type=int, help="at most this many tasks per archetype per repository")
    p.add_argument("--repo", action="append", help="only these repositories (repeatable)")
    p.add_argument("--runners", type=int)
    p.add_argument("--epochs", type=int, help="cap every recipe's epochs, for a short check of the training stages")
    p.add_argument("--no-engine-on", action="store_true", help="skip the second pass that drives with the pilot heads")
    p = sub.add_parser("run-status")
    p.add_argument("--run", required=True)
    args = ap.parse_args(argv)
    factory = config.load()
    runs = factory.root / "runs"

    if args.cmd in ("run", "run-status"):
        overrides = {} if args.cmd == "run-status" else {"task_cap": args.task_cap, "repos": args.repo, "runners": args.runners,
                                                          "train.epochs": args.epochs, "engine_on.enabled": False if args.no_engine_on else None}
        run = runmod.Run(factory=factory, name=args.run, settings=runmod.settings_for(factory, overrides),
                         push=getattr(args, "push", False), rebuild_image=getattr(args, "rebuild_image", False),
                         until=getattr(args, "until", None))
        if args.cmd == "run-status":
            state = run.load_state()
            for name in runmod.STAGES:
                s = state["stages"].get(name)
                print("%-14s %s" % (name, "%s at %s (%ss)%s" % (s["status"], s["at"], s.get("seconds", "?"), "  " + s["error"] if s.get("error") else "") if s else "pending"))
            return 0
        if args.dry_run:
            for name, action in runmod.plan(run, args.start, args.until, args.redo):
                print("%-14s %s" % (name, action))
            return 0
        try:
            state = runmod.execute(run, args.start, args.until, redo=args.redo)
        except runmod.RunError as exc:
            print(exc, file=sys.stderr)
            return 1
        report = state["stages"].get("report", {}).get("outputs", {}).get("report")
        print("run %s finished%s" % (args.run, "; report at " + report if report else ""))
        return 0
    if args.cmd == "check":
        print("%d models, %d repositories (%d held out), %d archetypes; hub %s, %s" % (
            len(factory.models), len(factory.repos), sum(r.split == "holdout" for r in factory.repos), len(factory.archetypes),
            config.hub()["dataset_repo"], config.hub()["model_repo"]))
        return 0
    if args.cmd == "release-heads":
        pairs = lambda items: dict(item.split("=", 1) for item in items)  # noqa: E731
        print(heads.build(args.heads, args.version, args.dataset_version, args.engine_commit, pairs(args.eval), pairs(args.baseline),
                          args.baseline_label, pairs(args.rerank), args.out))
        return 0
    if args.cmd == "fetch":
        print(hub.fetch(args.kind, args.version, args.out, args.repo))
        return 0
    if args.cmd == "verify":
        names = hub.check(args.kind, args.release, anchored=not args.unanchored)
        print("%s release complete: %d files match SHA256SUMS%s" % (
            args.kind, len(names), "; not checked against an anchor" if args.unanchored else ", which matches its anchor in releases/"))
        return 0
    if args.cmd == "audit" and args.kind == "heads":
        if not (args.dataset and args.lycaon and args.engine):
            ap.error("audit heads needs --dataset, --lycaon, and --engine")
        print(json.dumps(audit.heads(args.release, args.dataset, args.lycaon, args.engine), indent=2))
        return 0
    if args.cmd == "audit":
        report = audit.dataset(args.release, anchored=not args.unanchored)
        if args.rejudge:
            if not args.endpoint or not args.model:
                ap.error("--rejudge needs --endpoint and --model")
            report["rejudge"] = audit.rejudge(args.release, args.endpoint, args.model, args.rejudge)
        print(json.dumps(report, indent=2))
        return 0
    if args.cmd == "publish":
        print(json.dumps(hub.publish(args.kind, args.release, args.version, args.repo, args.push), indent=2))
        return 0
    if args.cmd == "repos":
        problems = repos.fetch(factory) if args.action == "fetch" else repos.verify(factory)
        for line in problems:
            print(line)
        return 1 if problems else 0
    if args.cmd == "serve":
        for model in factory.served():
            if args.action == "start":
                serve.start(factory, model)
                serve.start_tunnels(factory, model)
            elif args.action == "stop":
                serve.stop_tunnels(factory, model)
                serve.stop(factory, model)
        if args.action == "start":
            while not all(serve.ready(factory, m) for m in factory.served()):
                if not all(serve.running(factory, m) for m in factory.served()):
                    print("a server exited; see %s/logs" % factory.root)
                    return 1
                time.sleep(10)
        for model in factory.served():
            print("%s running=%s ready=%s" % (model.id, serve.running(factory, model), serve.ready(factory, model)))
        return 0
    if args.cmd == "tasks":
        counts = tasks.generate(factory, factory.root / "repos", args.out, seed=args.seed)
        provenance.record(factory, "%s/provenance.json" % args.out, "tasks", uses_engine=False, seed=args.seed, counts=counts)
        print(json.dumps(counts, indent=2))
        return 0
    if args.cmd == "coderank":
        out = Path(args.out)
        if args.action == "harvest":
            counts = coderank.harvest(factory, args.decide_rerank, out)
        elif args.action == "pairs":
            counts = coderank.pairs(factory, args.units, out, args.per_repo, args.seed)
        else:
            counts = coderank.dumps(factory, args.decide_rerank, args.units, args.pairs, out)
        provenance.record(factory, out / "provenance.json", "coderank-" + args.action, uses_engine=args.action != "pairs",
                          bin_dir=os.path.dirname(os.path.abspath(args.decide_rerank)) if args.decide_rerank else None,
                          counts=counts, per_repo=args.per_repo, seed=args.seed, units=args.units, pairs=args.pairs,
                          decide_rerank=provenance.file_sha256(args.decide_rerank) if args.decide_rerank else None)
        print(json.dumps(counts, indent=2) if counts else "done")
        return 0
    if args.cmd == "fleet":
        if args.action == "warm":
            fleet.ensure_network(factory)
            failed = {name: code for name, code in fleet.warm_cache(factory, args.image).items() if code}
            print("cache warmed for %d repositories%s" % (len(factory.repos), "; failed: %s" % failed if failed else ""))
            return 1 if failed else 0
        if args.action == "image":
            fleet.ensure_network(factory)
            if args.decide_deadline_ms and not args.config_dir:
                ap.error("--decide-deadline-ms needs --config-dir")
            fleet.build_image(factory, args.lycaon_bin, args.engine, args.tag, args.decide_deadline_ms, args.config_dir)
            return 0
        run_dir = runs / args.run
        if args.action == "plan":
            run_dir.mkdir(parents=True, exist_ok=True)
            planned = fleet.plan_shards(factory, args.tasks, run_dir, fleet.pilot_env(args.pilot), args.repo)
            tasks_record = os.path.join(args.tasks, "provenance.json")
            provenance.record(factory, run_dir / "provenance.json", "fleet", bin_dir=args.lycaon_bin, tasks=str(args.tasks), pilot=args.pilot,
                              decide_env=fleet.pilot_env(args.pilot), image=provenance.image(args.image or factory.fleet["image"]),
                              tasks_provenance=json.load(open(tasks_record)) if os.path.exists(tasks_record) else None)
            print("%d shards planned" % planned)
            return 0
        if args.action == "replan":
            removed, planned = fleet.replan(factory, args.tasks, run_dir, fleet.pilot_env(args.pilot))
            record = run_dir / "provenance.json"
            doc = json.loads(record.read_text()) if record.exists() else {}
            doc.setdefault("serving_changes", []).append({"replanned_shards": planned, "serving": provenance.serving(factory),
                                                          "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
            record.write_text(json.dumps(doc, indent=2) + "\n")
            print("replanned %d unstarted shards" % removed)
            return 0
        if args.action == "run":
            fleet.ensure_network(factory)
            mounts = [(args.pilot, fleet.PILOT_MOUNT)] if args.pilot else []
            try:
                done, failed = fleet.run(factory, run_dir, args.runners, mounts, args.image)
            except fleet.AccountRefused as exc:
                print("fleet stopped: %s" % exc, file=sys.stderr)
                return 1
            print("shards done=%d failed=%d" % (done, failed))
            return 0 if not failed else 1
        if args.action == "salvage":
            saved, settled = fleet.salvage(run_dir, runs / args.into)
            counts = tasks.rebase(factory, args.tasks, args.out, settled)
            print("salvaged %d shards; %d tasks settled; still to drive: %s" % (saved, len(settled), json.dumps(counts)))
            return 0
        for name in fleet.reap(factory, run_dir):
            print("removed", name)
        return 0
    if args.cmd == "collect":
        out, stats = fleet.collect(runs / args.run, args.pass_name, args.tasks)
        print(out, json.dumps(stats))
        return 0
    if args.cmd == "skillreq":
        with open(args.corpus, encoding="utf-8") as fh:
            corpus = json.load(fh)
        families = {k: int(v) for k, v in (kv.split("=") for kv in args.families.split(","))}
        report = skillreq.write(factory, corpus, args.out, args.writer, families, args.none_families, args.per_family, seed=args.seed,
                                eval_fraction=args.eval_fraction, only=tuple(args.skill), workers=args.workers, repos=tuple(args.repo),
                                split=args.split, offered=skillreq.offered_from(args.offered_from) if args.offered_from else None)
        report["provenance"] = provenance.record(factory, args.out + ".provenance.json", "skillreq", uses_engine=False, writer=args.writer,
                                                 families=families, none_families=args.none_families, per_family=args.per_family,
                                                 seed=args.seed, corpus_revision=corpus["catalog_revision"], repos=args.repo, split=args.split)["at"]
        print(json.dumps(report))
        return 0
    if args.cmd == "judge":
        with open(args.corpus, encoding="utf-8") as fh:
            corpus = json.load(fh)
        units = tuple(args.unit or judge.UNITS)
        report = judge.run(factory, corpus, args.rows, args.out, workers=args.workers, second=args.second_fraction, units=units)
        report["provenance"] = provenance.record(factory, args.out + ".provenance.json", "judge", rows=args.rows, units=list(units),
                                                 corpus_revision=corpus["catalog_revision"], corpus_sha256=provenance.file_sha256(args.corpus))["at"]
        print(json.dumps(report))
        with open(args.out + ".agreement.json", "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)
        return 0
    if args.cmd == "split":
        print(json.dumps(split.split(factory, args.rows, args.out), indent=2))
        return 0
    if args.cmd == "release":
        print(release.build(factory, args.split, args.version, args.out, args.schema, args.corpus, args.code_ref, args.agreement,
                            args.stage, driven=args.driven, coderank=args.coderank, stopping=args.stopping))
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
