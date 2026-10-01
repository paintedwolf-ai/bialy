#!/bin/bash
# setup.sh LYCAON_TAR DATA_DIR: prepare a GPU host to train and evaluate heads. The trainer
# comes from a Painted Wolf Code checkout (scripts/bialy at a recorded commit, as a tar),
# the backbone from its pinned revision.
set -euxo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "$0")" && pwd)
R=${R:-/scratch/bialy}; mkdir -p $R/{hf,venvs,logs,data,lycaon,out,eval,launchers}
export PATH=/root/.local/bin:$PATH HF_HOME=$R/hf
command -v uv || curl -LsSf https://astral.sh/uv/install.sh | sh
tar -xzf "$1" -C $R/lycaon
cp -r "$2"/. $R/data/
uvx --from huggingface_hub hf download convaiinnovations/laya-multilingual --revision e4e9ddf21a7b1903b7acffd8814ad4307bf63a67 > $R/logs/dl-laya.log 2>&1
# Offline loads resolve `main`; point it at the pinned revision.
H=$R/hf/hub/models--convaiinnovations--laya-multilingual; mkdir -p $H/refs; printf e4e9ddf21a7b1903b7acffd8814ad4307bf63a67 > $H/refs/main
uv venv -p 3.12 $R/venvs/train && VIRTUAL_ENV=$R/venvs/train uv pip install -r "$SCRIPT_DIR/requirements-cuda.txt"
VIRTUAL_ENV=$R/venvs/train uv pip freeze > "$R/logs/requirements-resolved.txt"
$R/venvs/train/bin/python -c 'import torch; print("torch", torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())'
