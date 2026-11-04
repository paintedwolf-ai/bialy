#!/bin/bash
# store.db holds the digest of each model output; its body is a file beside the store.
set -euo pipefail

out=$1
root=${2:-/cfg}
for dir in "$root"/projects/*/model-content; do
  [ -d "$dir" ] || continue
  project=$(basename "$(dirname "$dir")")
  mkdir -p "$out/projects/$project/model-content"
  cp -R "$dir"/. "$out/projects/$project/model-content/"
done
