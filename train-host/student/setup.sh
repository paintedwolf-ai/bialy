#!/bin/bash
# shellcheck source-path=SCRIPTDIR
# Prepare pinned sources, every declared student, and the independent resource reaper.
set -euo pipefail
# shellcheck source=lib.sh
. "$(cd -- "$(dirname -- "$0")" && pwd)/lib.sh"
require HOST_COMMIT STUDENT_CHECKPOINT_ROOT STUDENT_REMOTE_CONFIG
[[ "$HOST_COMMIT" =~ ^[0-9a-f]{40}$ ]] || { echo "HOST_COMMIT must be a full commit" >&2; exit 2; }
BIALY_CODE=$(cd -- "$STUDENT_HOME/../.." && pwd)
HOST_CHECKOUT=$SRC/paintedwolf-code-$HOST_COMMIT
sudo apt-get install -y -q cmake build-essential pkg-config rsync
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
sudo install -d -o "$USER" -g "$(id -gn)" "$STUDENT_ROOT" "$MODELS" "$RUNS" "$SRC" "$DATA"
mountpoint -q "$STUDENT_CHECKPOINT_ROOT" || { echo "checkpoint storage must be mounted independently" >&2; exit 2; }
if [ ! -d "$HOST_CHECKOUT" ]; then
  git clone --no-checkout "$HOST_URL" "$HOST_CHECKOUT"
  git -C "$HOST_CHECKOUT" fetch origin "$HOST_COMMIT"
  git -C "$HOST_CHECKOUT" checkout --detach "$HOST_COMMIT"
fi
[ "$(git -C "$HOST_CHECKOUT" rev-parse HEAD)" = "$HOST_COMMIT" ]
[ -z "$(git -C "$HOST_CHECKOUT" status --porcelain)" ]
[ -d "$LLAMA/.git" ] || git clone https://github.com/ggml-org/llama.cpp "$LLAMA"
git -C "$LLAMA" fetch origin "$LLAMA_COMMIT"
git -C "$LLAMA" checkout --detach "$LLAMA_COMMIT"
cmake -S "$LLAMA" -B "$LLAMA/build" -DGGML_CUDA=OFF -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build "$LLAMA/build" --target llama-quantize llama-completion -j"$(nproc)"
docker pull "$SERVER_IMAGE"
uv tool install huggingface_hub==2.2.0
uv run --project "$BIALY_CODE" --locked python -m bialy.student_host models --root "$MODELS"
uv sync --project "$HOST_CHECKOUT/$PIPE" --locked
uv run --project "$BIALY_CODE" --locked python -m bialy.student_host install-reaper --configuration "$STUDENT_REMOTE_CONFIG"
