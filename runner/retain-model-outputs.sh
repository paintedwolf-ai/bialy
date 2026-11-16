#!/bin/bash
# Copy every project's model-output bodies beside a preserved store. The store holds
# only each output's digest; host bundles read the body from these files.
set -euo pipefail

out=$1
root=${2:-/cfg}
for dir in "$root"/projects/*/model-content; do
  [ -d "$dir" ] || continue
  project=$(basename "$(dirname "$dir")")
  mkdir -p "$out/projects/$project/model-content"
  cp -R "$dir"/. "$out/projects/$project/model-content/"
done
