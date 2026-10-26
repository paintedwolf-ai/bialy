#!/bin/bash
# shellcheck source-path=SCRIPTDIR
# sft-scaleway.sh RUN SPLIT_DIR HOST_COMMIT: train the student on a rented
# Scaleway GPU server, run from your own machine.
#
# SPLIT_DIR is a directory on the training host holding train.jsonl, val.jsonl
# and holdout.jsonl. HOST_COMMIT is the full Painted Wolf Code commit whose
# trainer runs. The server is deleted with its volumes and IP on every exit,
# including failures; KEEP_SERVER=1 keeps it for debugging. The adapter run
# returns to the training host under RUNS/RUN, where merge, GGUF export and
# evaluation run.
set -euo pipefail
# shellcheck source=lib.sh
. "$(cd -- "$(dirname -- "$0")" && pwd)/lib.sh"
require SCW_PROFILE SCW_ZONE SCW_TYPE SCW_IMAGE SCW_ROOT_VOLUME SCW_HOURLY_EUR
RUN=${1:?run name}; SPLITS=${2:?split directory on the training host}; COMMIT=${3:?host commit}
[[ "$COMMIT" =~ ^[0-9a-f]{40}$ ]] || { echo "HOST_COMMIT must be a full commit" >&2; exit 2; }
TRAIN_HOST=$STUDENT_TRAIN_HOST
SCW=(scw -p "$SCW_PROFILE")
STAGE=$(mktemp -d "${TMPDIR:-/tmp}/pw-sft-$RUN.XXXX")
SSH=(ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30)
SERVER=""
started=$(date +%s)

cleanup() {
  status=$?
  if [ -n "$SERVER" ] && [ "${KEEP_SERVER:-0}" != 1 ]; then
    step "deleting server $SERVER"
    "${SCW[@]}" instance server delete "$SERVER" zone="$SCW_ZONE" with-volumes=all with-ip=true force-shutdown=true || true
    if "${SCW[@]}" instance server get "$SERVER" zone="$SCW_ZONE" >/dev/null 2>&1; then
      echo "SERVER $SERVER STILL EXISTS: delete it in the Scaleway console to stop billing" >&2; status=1
    else
      echo "server deleted"
    fi
  fi
  hours=$(awk -v s="$started" -v e="$(date +%s)" 'BEGIN {printf "%.2f", (e - s) / 3600}')
  echo "elapsed ${hours} h, about EUR $(awk -v h="$hours" -v r="$SCW_HOURLY_EUR" 'BEGIN {printf "%.2f", h * r}') at the configured rate"
  rm -rf "$STAGE"
  exit "$status"
}
trap cleanup EXIT INT TERM

step "staging splits and the student inventory from the training host"
rsync -a "$TRAIN_HOST:$SPLITS/train.jsonl" "$TRAIN_HOST:$SPLITS/val.jsonl" "$TRAIN_HOST:$SPLITS/holdout.jsonl" "$STAGE/"
rsync -a "$TRAIN_HOST:$MODELS/$STUDENT_DIR.sha256" "$STAGE/"
wc -l "$STAGE"/*.jsonl

step "creating $SCW_TYPE in $SCW_ZONE"
created=$("${SCW[@]}" instance server create type="$SCW_TYPE" zone="$SCW_ZONE" image="$SCW_IMAGE" \
  root-volume="$SCW_ROOT_VOLUME" ip=new name="pw-sft-$RUN" tags.0=paintedwolf-sft -o json)
SERVER=$(python3 -c 'import json,sys; print(json.load(sys.stdin)["id"])' <<<"$created")
IP=$(python3 -c 'import json,sys; d=json.load(sys.stdin); print((d.get("public_ip") or d["public_ips"][0])["address"])' <<<"$created")
GPU_HOST=root@$IP
echo "server $SERVER"
for _ in $(seq 1 60); do "${SSH[@]}" -o ConnectTimeout=10 "$GPU_HOST" true 2>/dev/null && break; sleep 10; done

step "bootstrap"
"${SSH[@]}" "$GPU_HOST" bash -s -- "$COMMIT" "$STUDENT_REPO" "$STUDENT_REV" "$STUDENT_DIR" "$HOST_URL" "$PIPE" <<'REMOTE'
set -euo pipefail
COMMIT=$1; REPO=$2; REV=$3; DIR=$4; URL=$5; PIPE=$6
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1
export PATH=$HOME/.local/bin:$PATH
mkdir -p /scratch/data /scratch/models /scratch/runs
git clone -q "$URL" /scratch/paintedwolf-code
git -C /scratch/paintedwolf-code fetch -q origin "$COMMIT" && git -C /scratch/paintedwolf-code checkout -q "$COMMIT"
cd /scratch/paintedwolf-code && uv sync -q --project "$PIPE" --locked
uv run --project "$PIPE" --locked python -c "import torch; assert torch.cuda.is_available(), 'no CUDA'; print('torch', torch.__version__, torch.cuda.get_device_name(0))"
uv tool install -q huggingface_hub >/dev/null 2>&1
hf download "$REPO" --revision "$REV" --local-dir "/scratch/models/$DIR" >/dev/null
rm -rf "/scratch/models/$DIR/.cache"
REMOTE

step "uploading splits and verifying the student against the training host's inventory"
rsync -a "$STAGE"/*.jsonl "$GPU_HOST:/scratch/data/"
rsync -a "$STAGE/$STUDENT_DIR.sha256" "$GPU_HOST:/scratch/models/"
"${SSH[@]}" "$GPU_HOST" "cd /scratch/models/$STUDENT_DIR && sha256sum -c --quiet ../$STUDENT_DIR.sha256 && echo 'student matches the inventory'"

step "training"
TRAIN="cd /scratch/paintedwolf-code && PATH=\$HOME/.local/bin:\$PATH uv run --project $PIPE --locked \
python $PIPE/train.py --mode sft --model /scratch/models/$STUDENT_DIR \
--train /scratch/data/train.jsonl --validation /scratch/data/val.jsonl --holdout /scratch/data/holdout.jsonl \
--out /scratch/runs/$RUN --device cuda:0 --targets '$STUDENT_TARGETS' --max-length 8192 --checkpoint-every 50"
"${SSH[@]}" "$GPU_HOST" "$TRAIN" || true
for attempt in 1 2 3; do
  "${SSH[@]}" "$GPU_HOST" "test -f /scratch/runs/$RUN/result.json" && break
  echo "no result yet; resume attempt $attempt"
  "${SSH[@]}" "$GPU_HOST" "$TRAIN --resume" || true
done
"${SSH[@]}" "$GPU_HOST" "test -f /scratch/runs/$RUN/result.json" || { echo "training did not complete" >&2; exit 1; }

step "returning the run to the training host"
rsync -a --exclude checkpoints "$GPU_HOST:/scratch/runs/$RUN/" "$STAGE/$RUN/"
rsync -a "$STAGE/$RUN/" "$TRAIN_HOST:$RUNS/$RUN/"
"${SSH[@]}" "$TRAIN_HOST" "ls -la $RUNS/$RUN/adapter && python3 -c \"import json; r=json.load(open('$RUNS/$RUN/result.json')); print('history', r['history'])\""
