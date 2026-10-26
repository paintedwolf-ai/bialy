#!/usr/bin/env bash
# Unattended retention pilot on a GPU host: preflight the whole data path on
# stores earlier runs kept, then start `bialy run` detached with retries.
#
#   retention_pilot.sh preflight [--run NAME] [--rebuild]  build, check, and dry-run export/train/eval; no provider calls
#   retention_pilot.sh launch    [--run NAME] [--rebuild]  preflight, then start the run detached and return
#   retention_pilot.sh resume    [--run NAME]              rebuild from the synced checkout, re-export the shards the
#                                                          engine-off pass kept, and continue the run detached
#   retention_pilot.sh status    [--run NAME]              the run's stage table and log tail
#
# --rebuild rebuilds the binaries, image, and corpus from the synced checkout
# even when an earlier invocation of the same run already built them. resume
# always rebuilds, and keeps every stage the run has finished: the shards already
# driven are exported again by the new binary and never driven twice.
#
# Layout (BIALY_RUN_HOME, default ~/bialy-run): bialy/ (this repository),
# paintedwolf-code/ (the pinned checkout), root/ (the factory root named in
# bialy/config/factory.yaml), .env (FIREWORKS_API_KEY, mode 600). The fleet
# stages need root for overlay mounts, so the run itself goes through sudo -n.
set -euo pipefail

HOME_DIR="${BIALY_RUN_HOME:-$HOME/bialy-run}"
RUN_NAME="retention-pilot"
WORKSPACES=(httpx cobra flask python-cli zod vue-vite)
# Both passes: the second drives the same tasks with the pilot head deciding, which
# is the comparison the verdict reads.
RUN_FLAGS=(--recipes R1 --no-judge --follow-up-rate 1.0 --follow-ups 3 --task-cap 2 --runners 4 --spend-ceiling 40)
ATTEMPTS=3

REBUILD=()
cmd="${1:-}"; shift || true
while [[ $# -gt 0 ]]; do
  case "$1" in
    --run) RUN_NAME="$2"; shift 2 ;;
    --rebuild) REBUILD=(--from build --rebuild-image); shift ;;
    *) echo "unknown argument $1" >&2; exit 2 ;;
  esac
done
[[ -n "$cmd" ]] || { sed -n '2,12p' "$0"; exit 2; }

BIALY="$HOME_DIR/bialy"
CHECKOUT="$HOME_DIR/paintedwolf-code"
ROOT="$(cd "$BIALY" && python3 -c "import yaml,sys; print(yaml.safe_load(open('config/factory.yaml'))['root'])")"
RUN_DIR="$ROOT/runs/$RUN_NAME"
LOG="$HOME_DIR/$RUN_NAME.log"
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

# The fleet stages need root, and everything under the factory root is then
# root-owned, so preflight and launch run as root with the user's HOME and
# tooling, the way run-desk2.sh did. The key stays in the environment of this
# process tree only.
if [[ "$cmd" != "status" && "$(id -u)" != 0 ]]; then
  [[ -f "$HOME_DIR/.env" ]] || { echo "missing $HOME_DIR/.env"; exit 1; }
  set -a
  # shellcheck disable=SC1091
  . "$HOME_DIR/.env"
  set +a
  exec sudo -n env HOME="$HOME" PATH="$PATH" BIALY_RUN_HOME="$HOME_DIR" FIREWORKS_API_KEY="${FIREWORKS_API_KEY:?}" "$0" "$cmd" --run "$RUN_NAME" ${REBUILD:+--rebuild}
fi

run_bialy() {
  (cd "$BIALY" && uv run bialy run --run "$RUN_NAME" "${RUN_FLAGS[@]}" "${WORKSPACES[@]/#/--workspace=}" "$@")
}

preflight() {
  echo "== preflight $RUN_NAME at $(date -u +%FT%TZ)"
  [[ -n "${FIREWORKS_API_KEY:-}" ]] || { echo "FIREWORKS_API_KEY is not set"; exit 1; }
  (cd "$BIALY" && uv run ruff check src tests && uv run pytest -q tests && uv run bialy check)
  echo "== build, image, and corpus (no provider calls)"
  run_bialy --until corpus "${REBUILD[@]}"
  REBUILD=()
  local debug="$RUN_DIR/build/bin/lycaon-debug"
  [[ -x "$debug" ]] || { echo "no lycaon-debug at $debug"; exit 1; }
  echo "== dry data path over the stores earlier runs kept"
  local dry="$RUN_DIR/preflight"
  mkdir -p "$dry"
  : > "$dry/retention.jsonl"
  local stores=0
  while IFS= read -r store; do
    [[ "$store" == "$RUN_DIR"/* ]] && continue
    stores=$((stores + 1))
    "$debug" decide export --db "$store" --out /dev/null --retention "$dry/one.jsonl" 2>>"$dry/export.log" || { echo "export failed on $store"; exit 1; }
    cat "$dry/one.jsonl" >> "$dry/retention.jsonl"
  done < <(find "$ROOT/runs" -path '*/shards/*/store.db' -type f 2>/dev/null | sort)
  rm -f "$dry/one.jsonl"
  echo "exported from $stores stores:"
  python3 - "$dry/retention.jsonl" <<'PY'
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
cands = sum(len(r["candidates"]) for r in rows)
needed = sum(sum(1 for v in r["labels"]["needed"].values() if v) for r in rows)
later = sum(sum(1 for v in r["labels"]["later"].values() if v) for r in rows)
recon = sum(1 for r in rows if r["reconstructed"])
print("  rows=%d reconstructed=%d candidates=%d needed=%d (%.1f%%) later=%d (%.1f%%)" % (
    len(rows), recon, cands, needed, 100.0 * needed / cands if cands else 0, later, 100.0 * later / cands if cands else 0))
by = {}
for r in rows:
    for v in r["labels"]["by"].values():
        for b in v:
            by[b] = by.get(b, 0) + 1
print("  needed by:", json.dumps(by))
if len(rows) < 2:
    print("  too few rows for a dry training pass; the live run will supply them")
    sys.exit(3)
# A deterministic split for the smoke pass: every other row validates.
with open(sys.argv[1] + ".train", "w") as tr, open(sys.argv[1] + ".val", "w") as va:
    for i, r in enumerate(rows):
        (va if i % 2 else tr).write(json.dumps(r) + "\n")
PY
  local split_rc=$?
  if [[ $split_rc -eq 3 ]]; then
    echo "== dry training skipped (not enough rows)"; return 0
  fi
  echo "== dry training (one epoch) and replay through the host engine"
  local python="$ROOT/venvs/train/bin/python"
  [[ -x "$python" ]] || { echo "no trainer venv at $python; a previous run's train stage makes it"; exit 1; }
  local snapshot
  snapshot="$(find "$ROOT/hf/hub/models--convaiinnovations--laya-multilingual/snapshots" -mindepth 1 -maxdepth 1 -type d | head -1)"
  [[ -n "$snapshot" ]] || { echo "no backbone snapshot under $ROOT/hf"; exit 1; }
  (cd "$CHECKOUT" && env HF_HOME="$ROOT/hf" HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 LYCAON_DECIDE_MODEL_ID=convaiinnovations/laya-multilingual \
    "$python" scripts/bialy/train.py --corpus "$RUN_DIR/corpus.json" --train "$dry/retention.jsonl.train" --val "$dry/retention.jsonl.val" \
    --families retention --epochs 1 --batch-size 16 --model "$snapshot" --label preflight --out "$dry/retention.safetensors" 2>&1 | tail -8)
  [[ -f "$dry/retention.safetensors" ]] || { echo "dry training wrote no head"; exit 1; }
  (cd "$BIALY" && uv run python - "$CHECKOUT" "$RUN_DIR" "$ROOT" "$dry" <<'PY'
import sys
from pathlib import Path
from bialy import build
checkout, run_dir, root, dry = map(Path, sys.argv[1:5])
engine = build.host_engine(checkout, run_dir / "build" / "host-engine", root / "build-cache")["binary"]
model = build.model_dir(next((root / "hf/hub/models--convaiinnovations--laya-multilingual/snapshots").iterdir()), run_dir / "build" / "model")
build.launcher(engine, model, {"retention": str(dry / "retention.safetensors")}, dry / "launcher.sh")
print("launcher", dry / "launcher.sh")
PY
  )
  (cd "$CHECKOUT" && "$python" scripts/bialy/retention_eval.py --corpus "$RUN_DIR/corpus.json" --examples "$dry/retention.jsonl.val" \
    --fit --bootstrap 200 --engine "$dry/launcher.sh" --json "$dry/retention-eval.json" > "$dry/retention-eval.log")
  python3 - "$dry/retention-eval.json" <<'PY'
import json, sys
r = json.load(open(sys.argv[1]))
o = r["overall"]
print("  dry replay:", json.dumps({k: o.get(k) for k in ("rows", "needed_rate", "auc", "abstained", "latency_ms")}))
print("  engine at shipped policy:", json.dumps(o["engine"]))
print("  fitted:", json.dumps({k: r["fitted"][k] for k in ("drop_below", "confidence_floor")}), json.dumps(r["fitted"]["tally"]))
for name, t in o["structural"].items():
    print("  structural %-14s released=%s needed=%s objective/request=%s" % (name, t["released"], t["needed_released"], t["objective_per_request"]))
print("  recommendation:", json.dumps(r["recommendation"]["facts"]))
PY
  echo "== preflight passed"
}

launch() {
  preflight
  start_detached
}

resume() {
  echo "== resume $RUN_NAME at $(date -u +%FT%TZ)"
  [[ -n "${FIREWORKS_API_KEY:-}" ]] || { echo "FIREWORKS_API_KEY is not set"; exit 1; }
  if pgrep -f "bialy run --run $RUN_NAME " >/dev/null; then
    echo "a bialy run of $RUN_NAME is still running; stop it before resuming"; exit 1
  fi
  [[ -f "$RUN_DIR/run.json" ]] || { echo "no run at $RUN_DIR to resume"; exit 1; }
  (cd "$BIALY" && uv run ruff check src tests && uv run pytest -q tests && uv run bialy check)
  echo "== rebuilding the binaries, image, and corpus from the synced checkout (no provider calls)"
  run_bialy --redo check --redo build --redo image --redo corpus --rebuild-image --until corpus
  local debug="$RUN_DIR/build/bin/lycaon-debug"
  [[ -x "$debug" ]] || { echo "no lycaon-debug at $debug"; exit 1; }
  echo "== exporting the engine-off shards again with the rebuilt binary"
  (cd "$BIALY" && uv run bialy fleet reexport --run "$RUN_NAME/fleet-off" --lycaon-debug "$debug")
  start_detached
}

start_detached() {
  echo "== launching $RUN_NAME detached; log $LOG"
  # The quoted script is the detached process's own; its expansions happen there.
  # shellcheck disable=SC2016
  setsid nohup bash -c '
    cd "$1"; attempts="$2"; shift 2
    for attempt in $(seq 1 "$attempts"); do
      echo "== attempt $attempt $(date -u +%FT%TZ)"
      if uv run bialy run "$@"; then
        echo "== finished $(date -u +%FT%TZ)"; exit 0
      fi
      echo "== attempt $attempt failed; resuming in 120 s"; sleep 120
    done
    echo "== gave up after $attempts attempts $(date -u +%FT%TZ)"; exit 1
  ' bash "$BIALY" "$ATTEMPTS" --run "$RUN_NAME" "${RUN_FLAGS[@]}" "${WORKSPACES[@]/#/--workspace=}" >>"$LOG" 2>&1 < /dev/null &
  disown
  sleep 2
  echo "started; follow with: tail -f $LOG; status: $0 status --run $RUN_NAME"
}

status() {
  (cd "$BIALY" && sudo -n env HOME="$HOME" PATH="$PATH" uv run bialy run-status --run "$RUN_NAME") || true
  echo "---- $LOG"
  sudo -n tail -20 "$LOG" 2>/dev/null || echo "(no log yet)"
  sudo -n test -f "$RUN_DIR/REPORT.md" && sudo -n sed -n '/## Retention viability/,/## Publishing/p' "$RUN_DIR/REPORT.md"
  true
}

case "$cmd" in
  preflight) preflight ;;
  launch) launch ;;
  resume) resume ;;
  status) status ;;
  *) echo "unknown command $cmd" >&2; exit 2 ;;
esac
