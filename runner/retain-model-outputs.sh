#!/bin/bash
# store.db holds the digest of each model output; its body is a file beside the store.
set -euo pipefail

out=$1
for dir in /cfg/projects/*/model-content; do
  [ -d "$dir" ] || continue
  project=$(basename "$(dirname "$dir")")
  mkdir -p "$out/projects/$project"
  cp -R "$dir" "$out/projects/$project/"
done
