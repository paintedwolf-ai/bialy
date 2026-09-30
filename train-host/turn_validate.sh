#!/bin/bash
# turn_validate.sh GPU RECIPE RUN: score a completed independent tool head on all
# original validation rows. Training must be tracked as RECIPE-RUN by jobctl.sh.
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "$0")" && pwd)
GPU=$1; RECIPE=$2; RUN=$3
R=${R:-/scratch/bialy}; export R
"$SCRIPT_DIR/jobctl.sh" wait "$RECIPE-$RUN"
export CUDA_VISIBLE_DEVICES=$GPU HF_HOME=$R/hf HF_HUB_OFFLINE=1 LYCAON_DECIDE_DEVICE=cuda
PY=$R/venvs/train/bin/python
OUT=$R/eval/$RECIPE-$RUN
mkdir -p "$R/eval"
mkdir "$OUT" || { echo "$OUT exists; validation outputs are immutable" >&2; exit 1; }
"$PY" "$SCRIPT_DIR/turn_forward_probe.py" --trainer "$R/lycaon" \
  --corpus "$R/data/independent-corpus.json" --rows "$R/data/judged/val.jsonl" \
  --head "$R/out/$RECIPE-$RUN/turn-load.safetensors" --out "$OUT/predictions.jsonl" --limit 0
"$PY" "$SCRIPT_DIR/turn_score.py" --trainer "$R/lycaon" \
  --corpus "$R/data/independent-corpus.json" --rows "$R/data/judged/val.jsonl" \
  --predictions "$OUT/predictions.jsonl" --out "$OUT/report.json"
