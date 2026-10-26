#!/bin/bash
# shellcheck source-path=SCRIPTDIR
# pilot.sh RUN: exercise the whole student pipeline on the training host with a
# run's qualified rows. Trains with a planned pause, hard-kills the trainer after
# a checkpoint, resumes to completion, merges, converts to GGUF, and loads it.
# The point is recoverability and export parity, not model quality.
set -euo pipefail
# shellcheck source=lib.sh
. "$(cd -- "$(dirname -- "$0")" && pwd)/lib.sh"
require STUDENT_PILOT_HOLDOUT
RUN=${1:?run name}
HOST=$SRC/paintedwolf-code
OUT=$RUNS/$RUN-pilot
SPLITS=$DATA/$RUN-pilot
MODEL=$MODELS/$STUDENT_DIR
py() { (cd "$HOST" && uv run --project "$PIPE" --locked python "$@"); }

step "pilot splits from the run's qualified rows"
LATEST=$(sudo cat "$FACTORY/runs/$RUN/custom_model/latest.json" | python3 -c 'import json,sys; print(json.load(sys.stdin)["directory"])')
mkdir -p "$SPLITS"
# The factory writes as root; the pilot keeps its own copy of the rows.
# shellcheck disable=SC2024
sudo cat "$LATEST/sft.qualified-review.jsonl" > "$SPLITS/qualified.jsonl"
python3 - "$SPLITS" "$STUDENT_PILOT_HOLDOUT" <<'EOF'
import collections, json, sys
data, held = sys.argv[1], sys.argv[2]
rows = [json.loads(line) for line in open(data + "/qualified.jsonl") if line.strip()]
groups = collections.defaultdict(list)
for row in rows:
    groups[(row.get("meta") or {}).get("workspace"), row["split_group"]].append(row)
holdout = [k for k in groups if k[0] == held]
rest = sorted((k for k in groups if k not in holdout), key=lambda k: -len(groups[k]))
if not holdout or len(rest) < 2:
    sys.exit("pilot needs a held-out group and two training groups; have %d rows in %d groups" % (len(rows), len(groups)))
for name, keys in {"holdout.jsonl": holdout, "val.jsonl": rest[-1:], "train.jsonl": rest[:-1]}.items():
    with open(data + "/" + name, "w") as out:
        for k in keys:
            for row in groups[k]:
                out.write(json.dumps(row) + "\n")
    print(name, sum(len(groups[k]) for k in keys), "rows")
EOF

ARGS=("$PIPE/train.py" --mode sft --model "$MODEL" --train "$SPLITS/train.jsonl"
      --validation "$SPLITS/val.jsonl" --holdout "$SPLITS/holdout.jsonl" --out "$OUT" --device cuda:0
      --targets "$STUDENT_TARGETS" --max-length 32768 --checkpoint-every 5)

step "first invocation: 10 updates, then a planned pause"
rm -rf "$OUT"
py "${ARGS[@]}" --max-updates 10 | tail -20
python3 -c "import json,sys; p=json.load(open(sys.argv[1])); print({k: p.get(k) for k in ('status','step','total_steps','median_update_seconds','peak_cuda_bytes','peak_process_rss_bytes')})" "$OUT/progress.json"

step "hard kill after a completed checkpoint"
(cd "$HOST" && exec uv run --project "$PIPE" --locked python "${ARGS[@]}" --resume) > "$OUT.kill.log" 2>&1 &
TRAINER=$!
before=$(find "$OUT/checkpoints" -mindepth 1 -maxdepth 1 | wc -l)
for _ in $(seq 1 600); do
  [ "$(find "$OUT/checkpoints" -mindepth 1 -maxdepth 1 | wc -l)" -gt "$before" ] && break
  kill -0 "$TRAINER" 2>/dev/null || break
  sleep 2
done
pkill -9 -P "$TRAINER" 2>/dev/null || true
kill -9 "$TRAINER" 2>/dev/null || true
wait "$TRAINER" 2>/dev/null || true
echo "killed after $(find "$OUT/checkpoints" -mindepth 1 -maxdepth 1 | wc -l) checkpoint generations"

step "resume to completion"
py "${ARGS[@]}" --resume | tail -12
test -f "$OUT/result.json"
python3 -c "import json,sys; r=json.load(open(sys.argv[1])); print('history', r['history']); p=r['resources']; print({k: p.get(k) for k in ('step','median_update_seconds','observed_update_range_seconds','peak_cuda_bytes','peak_process_rss_bytes','rejected_checkpoints')})" "$OUT/result.json"

step "merge and reload parity"
rm -rf "$OUT-merged"
py "$PIPE/export.py" --model "$MODEL" --run "$OUT" --out "$OUT-merged"
python3 -c "import json,sys; m=json.load(open(sys.argv[1])); print(m['merge_parity'], m['load_parity'], len(m['files']), 'files')" "$OUT-merged/manifest.json"

step "GGUF conversion, quantization, load probe"
rm -rf "$OUT-gguf"
py "$PIPE/gguf_export.py" --merge "$OUT-merged" --llama-cpp "$LLAMA" --revision "$LLAMA_COMMIT" \
  --quantizer "$LLAMA/build/bin/llama-quantize" --loader "$LLAMA/build/bin/llama-completion" --out "$OUT-gguf"
ls -la "$OUT-gguf"/*.gguf
python3 -c "import json,sys; m=json.load(open(sys.argv[1])); print('load_probe', m['load_probe'], 'quantized tensors', len(m['quantized']['tensors']))" "$OUT-gguf/manifest.json"

step "done"
