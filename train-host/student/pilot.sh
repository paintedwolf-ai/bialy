#!/bin/bash
# shellcheck source-path=SCRIPTDIR
# Run every declared student pilot against the factory's frozen splits.
set -euo pipefail
# shellcheck source=lib.sh
. "$(cd -- "$(dirname -- "$0")" && pwd)/lib.sh"
require STUDENT_CHECKPOINT_ROOT HOST_COMMIT PW_TRAINING_SOURCE_KEY
SPLITS=${1:?frozen split directory}
BIALY_CODE=$(cd -- "$STUDENT_HOME/../.." && pwd)
student() { uv run --project "$BIALY_CODE" --locked python -m bialy.student_host "$@"; }
declared=$(student students)
mapfile -t models <<<"$declared"
status=0
for model in "${models[@]}"; do
  if ! result=$(student create --student "$model" --root "$RUNS" --checkout "$SRC/paintedwolf-code-$HOST_COMMIT" \
    --models "$MODELS" --splits "$SPLITS/$model" --mirror "$STUDENT_CHECKPOINT_ROOT" --pilot); then
    status=1
    continue
  fi
  if ! directory=$(python3 -c 'import json,sys; print(json.load(sys.stdin)["directory"])' <<<"$result"); then
    status=1
    continue
  fi
  if student train "$directory"; then
    student export "$directory" --llama "$LLAMA" --revision "$LLAMA_COMMIT" || status=1
  else
    status=1
  fi
done
exit "$status"
