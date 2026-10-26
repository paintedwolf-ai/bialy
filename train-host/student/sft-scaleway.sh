#!/bin/bash
# shellcheck source-path=SCRIPTDIR
# Train both students with lease-owned rentals and durable checkpoint copies.
set -euo pipefail
# shellcheck source=lib.sh
. "$(cd -- "$(dirname -- "$0")" && pwd)/lib.sh"
require STUDENT_RENTAL_CONFIG PW_TRAINING_SOURCE_KEY
SPLITS=${1:?frozen paired split directory}
BIALY_CODE=$(cd -- "$STUDENT_HOME/../.." && pwd)
status=0
for model in e4b 12b; do
  uv run --project "$BIALY_CODE" --locked python -m bialy.student_host rent \
    --configuration "$STUDENT_RENTAL_CONFIG" --student "$model" --splits "$SPLITS/$model" \
    --source-key "$PW_TRAINING_SOURCE_KEY" --out "$RUNS/rentals" || status=1
done
exit "$status"
