#!/bin/bash
# setup-repo.sh <dir>: install what the repository needs to build and test, so
# a session that runs its tests measures the request, not a missing toolchain.
# Every package manager caches under /cache, the warmed cache seen through this
# runner's own overlay, so what one runner installs never reaches another.
set -u
dir=$1
cd "$dir" || exit 1
step() { timeout 900 "$@" >>/shard/setup.log 2>&1 || echo "setup step failed: $*" >>/shard/setup.log; }

if [ -f pyproject.toml ] || [ -f setup.py ]; then
  python3 -m venv /work/.venv
  step /work/.venv/bin/pip install -q -e .
  for req in requirements/tests.txt requirements/dev.txt requirements-dev.txt requirements.txt; do
    [ -f "$req" ] && step /work/.venv/bin/pip install -q -r "$req"
  done
  step /work/.venv/bin/pip install -q pytest
fi
[ -f go.mod ] && step go mod download
[ -f Cargo.toml ] && step cargo fetch
if [ -f pnpm-lock.yaml ]; then step corepack pnpm install --frozen-lockfile
elif [ -f package-lock.json ]; then step npm ci
elif [ -f package.json ]; then step npm install
fi
[ -f Gemfile ] && step bundle install
[ -f composer.json ] && step composer install --no-interaction --no-progress
[ -f pom.xml ] && step mvn -q -Dmaven.repo.local=/cache/m2 -DskipTests dependency:go-offline
exit 0
