#!/bin/bash
# eval.sh GPU NAME TURN_HEAD RANK_HEAD KIND CORPUS ROWS TOOL_TRUTH [ROSTER]: one calibration,
# replay, or skill-discovery run through the CUDA engine; writes eval/NAME.{txt,json,err}.
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "$0")" && pwd)
GPU=$1; NAME=$2; TL=$3; UR=$4; KIND=$5; CORPUS=$6; ROWS=$7; TRUTH=$8; ROSTER=${9:-}
R=${R:-/scratch/bialy}; ENGINE=${ENGINE:-/root/engine/lycaon/internal/decide/native/target/release/bialy}
SNAP=$R/hf/hub/models--convaiinnovations--laya-multilingual/snapshots/e4e9ddf21a7b1903b7acffd8814ad4307bf63a67
mkdir -p $R/eval $R/launchers; L=$R/launchers/$NAME
printf '#!/bin/sh\nexport CUDA_VISIBLE_DEVICES=%s\nexec %s serve --model %s/ --model-id convaiinnovations/laya-multilingual --device cuda --head-max-len 512 --head turn-load=%s --head unit-rank=%s\n' \
  "$GPU" "$ENGINE" "$SNAP" "$TL" "$UR" > $L
chmod +x $L; cd $R/lycaon
case $KIND in
  calibrate) python3 scripts/bialy/calibrate.py --corpus $CORPUS --examples $ROWS --engine $L --tool-truth $TRUTH > $R/eval/$NAME.txt 2> $R/eval/$NAME.err ;;
  replay) python3 scripts/bialy/replay_eval.py --corpus $CORPUS --examples $ROWS --engine $L --holdout-pack painted-wolf/browser --tool-truth $TRUTH --json $R/eval/$NAME.json > /dev/null 2> $R/eval/$NAME.err ;;
  rank-dump) python3 "$SCRIPT_DIR/rank_eval.py" dump --trainer "$R/lycaon" --corpus "$CORPUS" --rows "$ROWS" --engine "$L" --head-file "turn-load=$TL" --head-file "unit-rank=$UR" --out "$R/eval/$NAME.jsonl" > "$R/eval/$NAME.txt" 2> "$R/eval/$NAME.err" ;;
  discovery) python3 scripts/bialy/skill_discovery.py --corpus $CORPUS --examples $ROWS --engine $L --train $R/data/judged/train.jsonl ${ROSTER:+--roster $ROSTER} --json $R/eval/$NAME.json > $R/eval/$NAME.txt 2> $R/eval/$NAME.err ;;
  *) echo "unknown evaluation kind $KIND" >&2; exit 2 ;;
esac
