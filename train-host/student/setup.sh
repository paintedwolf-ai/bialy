#!/bin/bash
# shellcheck source-path=SCRIPTDIR
# setup.sh: prepare the training host. Run it there, from this checkout. It
# installs the toolchain, the serving image, llama.cpp's CPU tools, the two
# repositories, the released heads, the student checkpoint and its inventory,
# the trainer environment, and optionally an external teacher and tunnel key.
set -euo pipefail
# shellcheck source=lib.sh
. "$(cd -- "$(dirname -- "$0")" && pwd)/lib.sh"
require HOST_REF BIALY_REF

step "packages"
sudo apt-get install -y -q cmake build-essential pkg-config >/dev/null
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null
uv --version

step "directories"
sudo mkdir -p "$FACTORY" "$MODELS" "$HEADS" "$RUNS" "$SRC" "$DATA"
sudo chown -R "$USER:$USER" "$STUDENT_ROOT"
df -h "$STUDENT_ROOT" | tail -1

step "serving image"
docker pull -q "$SERVER_IMAGE"
docker image inspect "$SERVER_IMAGE" --format '{{.Id}} {{.Size}}'

step "llama.cpp $LLAMA_TAG CPU tools"
[ -d "$LLAMA/.git" ] || git clone -q https://github.com/ggml-org/llama.cpp "$LLAMA"
git -C "$LLAMA" fetch -q --tags origin
git -C "$LLAMA" checkout -q "$LLAMA_TAG"
[ "$(git -C "$LLAMA" rev-parse HEAD)" = "$LLAMA_COMMIT" ] || { echo "llama.cpp $LLAMA_TAG does not match the pinned commit" >&2; exit 1; }
cmake -S "$LLAMA" -B "$LLAMA/build" -DGGML_CUDA=OFF -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release >/dev/null
cmake --build "$LLAMA/build" --target llama-quantize llama-completion -j"$(nproc)" >/dev/null
ls -la "$LLAMA/build/bin/llama-quantize" "$LLAMA/build/bin/llama-completion"

step "repositories"
[ -d "$SRC/paintedwolf-code/.git" ] || git clone -q -b "$HOST_REF" "$HOST_URL" "$SRC/paintedwolf-code"
[ -d "$SRC/bialy/.git" ] || git clone -q -b "$BIALY_REF" "$BIALY_URL" "$SRC/bialy"
git -C "$SRC/paintedwolf-code" log --oneline -1
git -C "$SRC/bialy" log --oneline -1

step "bialy environment and released heads"
(cd "$SRC/bialy" && uv sync -q --extra hub && uv run bialy check)
mkdir -p "$HEADS/$HEADS_VERSION"
(cd "$SRC/bialy" && uv run bialy fetch heads --version "$HEADS_VERSION" --out "$HEADS/$HEADS_VERSION") | tail -1

uv tool install -q huggingface_hub >/dev/null 2>&1 || true
if [ -n "${TEACHER_FILE:-}" ]; then
  require TEACHER_REPO TEACHER_REV TEACHER_SHA256
  step "external teacher artifact"
  hf download "$TEACHER_REPO" "$TEACHER_FILE" --revision "$TEACHER_REV" --local-dir "$MODELS" >/dev/null
  echo "$TEACHER_SHA256  $MODELS/$TEACHER_FILE" | sha256sum -c -
fi

step "student checkpoint and inventory"
hf download "$STUDENT_REPO" --revision "$STUDENT_REV" --local-dir "$MODELS/$STUDENT_DIR" >/dev/null
rm -rf "$MODELS/$STUDENT_DIR/.cache"
if find "$MODELS/$STUDENT_DIR" -type l | grep -q .; then echo "symlinks in the student checkpoint" >&2; exit 1; fi
(cd "$MODELS/$STUDENT_DIR" && find . -type f -print0 | sort -z | xargs -0 sha256sum) > "$MODELS/$STUDENT_DIR.sha256"
wc -l "$MODELS/$STUDENT_DIR.sha256"

step "trainer environment"
(cd "$SRC/paintedwolf-code" && uv sync -q --project "$PIPE" --locked \
  && uv run --project "$PIPE" --locked python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())")

if [ -n "${TUNNEL_PORT:-}" ]; then
  step "tunnel key restricted to 127.0.0.1:$TUNNEL_PORT"
  if [ ! -f "$HOME/.ssh/bialy_tunnel" ]; then
    ssh-keygen -q -t ed25519 -N '' -C bialy-tunnel -f "$HOME/.ssh/bialy_tunnel"
    printf 'restrict,port-forwarding,permitopen="127.0.0.1:%s" %s\n' "$TUNNEL_PORT" "$(cat "$HOME/.ssh/bialy_tunnel.pub")" >> "$HOME/.ssh/authorized_keys"
  fi
fi

step "done"
du -sh "$STUDENT_ROOT"/* 2>/dev/null || true
