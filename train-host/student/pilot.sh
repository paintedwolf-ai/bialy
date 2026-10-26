#!/bin/bash
# shellcheck source-path=SCRIPTDIR
# Run both student pilots against the factory's frozen splits.
set -euo pipefail
# shellcheck source=lib.sh
. "$(cd -- "$(dirname -- "$0")" && pwd)/lib.sh"
require STUDENT_CHECKPOINT_ROOT HOST_COMMIT PW_TRAINING_SOURCE_KEY
SPLITS=${1:?frozen split directory}
BIALY_CODE=$(cd -- "$STUDENT_HOME/../.." && pwd)
student() { uv run --project "$BIALY_CODE" --locked python -m bialy.student_host "$@"; }
status=0
for model in e4b 12b; do
  result=$(student create --student "$model" --root "$RUNS" --checkout "$SRC/paintedwolf-code-$HOST_COMMIT" \
    --models "$MODELS" --splits "$SPLITS/$model" --mirror "$STUDENT_CHECKPOINT_ROOT" --pilot)
  directory=$(python3 -c 'import json,sys; print(json.load(sys.stdin)["directory"])' <<<"$result")
  if student train "$directory"; then
    student export "$directory" --llama "$LLAMA" --revision "$LLAMA_COMMIT" || status=1
  else
    status=1
  fi
done
exit "$status"
