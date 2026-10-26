#!/bin/bash
# The layout below is read by the scripts that source this file.
# shellcheck source-path=SCRIPTDIR disable=SC2034
# Shared settings for the student scripts: pinned inputs from pins.env, this
# deployment from student.local.env, and the working layout under STUDENT_ROOT.
STUDENT_HOME=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=pins.env
. "$STUDENT_HOME/pins.env"
if [ ! -f "$STUDENT_HOME/student.local.env" ]; then
  echo "copy student.example.env to student.local.env and fill in this deployment" >&2
  exit 2
fi
# shellcheck source=student.example.env
. "$STUDENT_HOME/student.local.env"

require() {
  local name
  for name in "$@"; do
    [ -n "${!name:-}" ] || { echo "student.local.env must set $name" >&2; exit 2; }
  done
}
require STUDENT_TRAIN_HOST STUDENT_ROOT

SRC=$STUDENT_ROOT/src
MODELS=$STUDENT_ROOT/models
RUNS=$STUDENT_ROOT/runs
DATA=$STUDENT_ROOT/data
HEADS=$STUDENT_ROOT/heads
LLAMA=$STUDENT_ROOT/llama.cpp
FACTORY=$STUDENT_ROOT/bialy
PIPE=scripts/bialy/custom_model
export PATH="$HOME/.local/bin:$PATH"

step() { echo; echo "=== $(date -u +%FT%TZ) $*"; }
