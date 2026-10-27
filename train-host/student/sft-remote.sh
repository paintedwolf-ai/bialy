#!/bin/bash
# shellcheck source-path=SCRIPTDIR
# Train every declared student with configured machines and durable checkpoint copies.
set -euo pipefail
# shellcheck source=lib.sh
. "$(cd -- "$(dirname -- "$0")" && pwd)/lib.sh"
require STUDENT_REMOTE_CONFIG PW_TRAINING_SOURCE_KEY
SPLITS=${1:?frozen paired split directory}
BIALY_CODE=$(cd -- "$STUDENT_HOME/../.." && pwd)
student() { uv run --project "$BIALY_CODE" --locked python -m bialy.student_host "$@"; }
declared=$(student students)
mapfile -t models <<<"$declared"
status=0
for model in "${models[@]}"; do
  student remote \
    --configuration "$STUDENT_REMOTE_CONFIG" --student "$model" --splits "$SPLITS/$model" \
    --source-key "$PW_TRAINING_SOURCE_KEY" --out "$RUNS/remote" || status=1
done
exit "$status"
