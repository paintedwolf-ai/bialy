# One runner: a Painted Wolf Code sidecar, the toolchains the seed repositories
# build with, and the shard entrypoint. Nothing in the image is a credential;
# the sidecar talks only to the factory's model servers and the open web.
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive LANG=C.UTF-8 LC_ALL=C.UTF-8
# The lib* and fonts-* packages are what the browser tool's headless Chrome needs.
RUN apt-get update -qq && apt-get install -y -qq --no-install-recommends \
      ca-certificates curl git build-essential pkg-config libssl-dev file procps less unzip zip xz-utils jq ripgrep sqlite3 \
      python3 python3-venv python3-pip python3-dev \
      ruby-full ruby-bundler \
      php-cli php-xml php-mbstring php-curl composer \
      openjdk-21-jdk-headless maven \
      libnss3 libatk1.0-0t64 libatk-bridge2.0-0t64 libcups2t64 libdrm2 libxkbcommon0 libxcomposite1 libxdamage1 \
      libxfixes3 libxrandr2 libgbm1 libasound2t64 libpango-1.0-0 libcairo2 libxshmfence1 libdbus-1-3 \
      fonts-liberation fonts-noto-core fonts-noto-cjk fonts-noto-color-emoji \
    && rm -rf /var/lib/apt/lists/*

# Toolchain downloads are pinned by their publishers' sha256 digests.
ARG GO_VERSION=1.26.6
ARG GO_SHA256=708effb774be8237570d0add163225abbdfaf4fca28b2611df167beba4feef89
RUN curl -fsSLo /tmp/go.tgz "https://go.dev/dl/go${GO_VERSION}.linux-amd64.tar.gz" \
    && echo "${GO_SHA256}  /tmp/go.tgz" | sha256sum -c - && tar -C /usr/local -xzf /tmp/go.tgz && rm /tmp/go.tgz
ARG RUSTUP_VERSION=1.29.1
ARG RUSTUP_SHA256=dda7234360b7f578ca8b0ddcb80145646fa61a67c1720a5abc7051b35c9fcb71
ENV RUSTUP_HOME=/opt/rustup CARGO_HOME=/opt/cargo
RUN curl -fsSLo /tmp/rustup-init "https://static.rust-lang.org/rustup/archive/${RUSTUP_VERSION}/x86_64-unknown-linux-gnu/rustup-init" \
    && echo "${RUSTUP_SHA256}  /tmp/rustup-init" | sha256sum -c - && chmod +x /tmp/rustup-init \
    && /tmp/rustup-init -y --profile minimal --no-modify-path && rm /tmp/rustup-init && chmod -R a+rwX /opt/cargo /opt/rustup
# Node.js LTS from nodejs.org: project scaffolders refuse the distribution's older major.
ARG NODE_VERSION=24.21.0
ARG NODE_SHA256=fd8e59d5a511510f6a298afb548f18c7d2b1be404d8b4a27d94fbe49f56cb2d6
RUN curl -fsSLo /tmp/node.txz "https://nodejs.org/dist/v${NODE_VERSION}/node-v${NODE_VERSION}-linux-x64.tar.xz" \
    && echo "${NODE_SHA256}  /tmp/node.txz" | sha256sum -c - && tar -C /usr/local --strip-components=1 -xJf /tmp/node.txz \
    && rm /tmp/node.txz
RUN npm install -g corepack && corepack enable

# Toolchains, and every package manager's cache under /cache. The sidecar resolves the
# user's environment from a login shell, so the same values go in its profile.
COPY profile.sh /etc/profile.d/bialy.sh
ENV PATH=/work/.venv/bin:/opt/cargo/bin:/usr/local/go/bin:/root/go/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    PIP_CACHE_DIR=/cache/pip npm_config_cache=/cache/npm GOMODCACHE=/cache/gomod GOCACHE=/cache/gobuild \
    CARGO_HOME=/cache/cargo CARGO_TARGET_DIR=/work/.target BUNDLE_PATH=/cache/bundle COMPOSER_CACHE_DIR=/cache/composer \
    GRADLE_USER_HOME=/cache/gradle COREPACK_HOME=/cache/corepack MAVEN_OPTS=-Dmaven.repo.local=/cache/m2

# Seed clones are mounted from the host, whose owner is not the container's root.
RUN git config --system --add safe.directory '*'

COPY engine/ /opt/bialy/engine/
COPY bin/ /opt/bialy/bin/
COPY --chmod=755 run-shard.sh setup-repo.sh /opt/bialy/
COPY unattended.yaml /opt/bialy/
COPY workflows/ /opt/bialy/workflows/
# The overlay holds staged configuration, such as an engine-on pass's decision deadlines.
ENV LYCAON_ENGINE_ROOT=/opt/bialy/engine LYCAON_CONFIG_ROOT=/opt/bialy/engine/overlay
ENTRYPOINT ["/opt/bialy/run-shard.sh"]
