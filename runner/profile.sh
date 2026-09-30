# shellcheck shell=sh
# Sourced by login shells in a runner: the toolchains and the shared package caches.
export PATH="/work/.venv/bin:/opt/cargo/bin:/usr/local/go/bin:/root/go/bin:$PATH"
export RUSTUP_HOME=/opt/rustup CARGO_HOME=/cache/cargo CARGO_TARGET_DIR=/work/.target
export PIP_CACHE_DIR=/cache/pip npm_config_cache=/cache/npm GOMODCACHE=/cache/gomod GOCACHE=/cache/gobuild
export BUNDLE_PATH=/cache/bundle COMPOSER_CACHE_DIR=/cache/composer GRADLE_USER_HOME=/cache/gradle
export COREPACK_HOME=/cache/corepack MAVEN_OPTS=-Dmaven.repo.local=/cache/m2
