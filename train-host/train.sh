#!/bin/bash
# train.sh GPU RECIPE RUN: train one head into out/RECIPE-RUN, which must not exist (one
# launch owns a run). Recipes name every argument; data comes from data/ as setup.sh placed it:
#   data/original  session rows judged with the first pair (open1's split)
#   data/judged    session rows judged by the hosted pair (tools, skills, needs)
#   data/skillreq  generated skill requests (train and eval families) judged by the hosted pair
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "$0")" && pwd)
GPU=$1; RECIPE=$2; RUN=$3
R=${R:-/scratch/bialy}; L=$R/lycaon; PY=$R/venvs/train/bin/python; C=$R/data/corpus.json; OUT=$R/out/$RECIPE-$RUN
export OMP_NUM_THREADS=${THREADS:-10} MKL_NUM_THREADS=${THREADS:-10} CUDA_VISIBLE_DEVICES=$GPU HF_HOME=$R/hf HF_HUB_OFFLINE=1 \
  LYCAON_DECIDE_MODEL_ID=convaiinnovations/laya-multilingual LYCAON_DECIDE_DEVICE=cuda PYTHONUNBUFFERED=1
mkdir -p $R/out
mkdir $OUT 2>/dev/null || { echo "$OUT exists; choose a new run name" >&2; exit 1; }
cd $L
TURN="--holdout-pack painted-wolf/browser --families tools,guides,kind --seed 11 --batch-size 32"
RANK="--families skills,requests --skill-scored 4 --skill-zeros 3 --seed 11 --batch-size 32"
case $RECIPE in
  B3|B4|B5|B6)
      C=$R/data/independent-corpus.json
      TRUTH=consensus; [[ $RECIPE = B4 ]] && TRUTH=called
      NEGATIVES=8; LR=5e-4
      [[ $RECIPE = B5 || $RECIPE = B6 ]] && NEGATIVES=24
      [[ $RECIPE = B6 ]] && LR=1e-4
      $PY -c 'import json,sys; c=json.load(open(sys.argv[1])); assert c["questions"]["tools"].get("independent"), "independent corpus required"' "$C"
      exec $PY scripts/decide/train.py --corpus "$C" --train "$R/data/judged/train.jsonl" --val "$R/data/judged/val.jsonl" \
        --families tools --tool-truth "$TRUTH" --tool-weight none --tool-negatives "$NEGATIVES" --lr "$LR" --pos-weight 6 --seed 11 \
        --batch-size 64 --epochs 45 --patience 8 --label "open1-turn-load-$RECIPE-$RUN-independent" --out "$OUT/turn-load.safetensors" ;;
  # Turn-load. A2 is rev4a's recipe on the open sessions; B2 the same on judged labels.
  A2) exec $PY scripts/decide/train.py --corpus $C --train $R/data/original/train.jsonl --val $R/data/original/val.jsonl $TURN \
        --tool-weight none --tool-truth called --label turn-load-a2 --out $OUT/turn-load.safetensors ;;
  B2) exec $PY scripts/decide/train.py --corpus $C --train $R/data/judged/train.jsonl --val $R/data/judged/val.jsonl $TURN \
        --tool-weight none --tool-truth consensus --label turn-load-b2 --out $OUT/turn-load.safetensors ;;
  # Unit-rank. E counts a session's skill reads over the judges; E1 and E2 add the generated
  # skill requests, all of them or half the families.
  E|E0)  exec $PY scripts/decide/train.py --corpus $C --train $R/data/judged/train.jsonl --val $R/data/selection.jsonl $RANK \
        --rank-levels skills-blended --label "open1-unit-rank-$(echo "$RECIPE" | tr A-Z a-z)-$RUN" --out $OUT/unit-rank.safetensors ;;
  E6) exec $PY scripts/decide/train.py --corpus $C --train $R/derived/hard-skill-train.jsonl --val $R/data/selection.jsonl \
        --families skills,requests --skill-scored 8 --skill-zeros 12 --seed 11 --batch-size 32 \
        --rank-levels skills-blended --label "open1-unit-rank-e6-$RUN" --out $OUT/unit-rank.safetensors ;;
  E1|E2|E3|E4|E5)
      FRACTION=0.5; [[ $RECIPE = E1 || $RECIPE = E3 ]] && FRACTION=1.0
      LEVELS=skills-blended
      if [[ $RECIPE = E3 || $RECIPE = E4 || $RECIPE = E5 ]]; then
        RANK="--families skills,requests --skill-scored 8 --skill-zeros 12 --seed 11 --batch-size 32"
      fi
      [[ $RECIPE = E5 ]] && LEVELS=judged
      $PY "$SCRIPT_DIR/mix.py" $OUT/train.jsonl $FRACTION 7 $R/data/judged/train.jsonl $R/data/skillreq/train.judged.jsonl
      exec $PY scripts/decide/train.py --corpus $C --train $OUT/train.jsonl --val $R/data/selection.jsonl $RANK \
        --rank-levels "$LEVELS" --label "open1-unit-rank-$(echo "$RECIPE" | tr A-Z a-z)-$RUN" --out $OUT/unit-rank.safetensors ;;
  *) echo "unknown recipe $RECIPE" >&2; exit 2 ;;
esac
