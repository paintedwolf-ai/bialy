#!/bin/bash
# run-shard.sh: drive one shard of tasks through one sidecar, then export its rows.
#
# Mounts: /repos (read-only seed clones), /shard (this shard's tasks.jsonl and
# sidecar.env; its outputs land here), /cache (this runner's view of the package cache).
# Environment: WORKSPACE and WORKSPACE_KIND (repository or stack), COMMIT for a
# repository, PROMPT_TIMEOUT, and the model servers in /shard/providers.local.yaml and
# /shard/model-policy.yaml.
set -euo pipefail
# shellcheck source=runner/provider-credentials.sh
. /opt/bialy/provider-credentials.sh
: "${WORKSPACE:?}" "${WORKSPACE_KIND:?}" "${PROMPT_TIMEOUT:=20m}"
out=/shard
project=/work/"$WORKSPACE"
exec >>"$out/runner.log" 2>&1
echo "shard start $(date -u +%FT%TZ) workspace=$WORKSPACE kind=$WORKSPACE_KIND commit=${COMMIT:-}"

case "$WORKSPACE_KIND" in
  repository)
    # A fresh checkout at the pinned commit; the seed clone is never written.
    : "${COMMIT:?}"
    git clone -q --no-hardlinks /repos/"$WORKSPACE" "$project"
    git -C "$project" checkout -q "$COMMIT"
    ;;
  stack)
    # Greenfield: an empty directory with git initialised and nothing else.
    mkdir -p "$project"
    git -C "$project" init -q -b main
    ;;
  *)
    echo "unknown workspace kind $WORKSPACE_KIND"
    exit 1
    ;;
esac
git -C "$project" config user.email runner@paintedwolf.invalid
git -C "$project" config user.name "Painted Wolf runner"
# Project workflows a task can start; git ignores them so the workspace stays clean.
mkdir -p "$project"/.paintedwolf
cp -R /opt/bialy/workflows "$project"/.paintedwolf/workflows
echo /.paintedwolf/ >>"$project"/.git/info/exclude
if [ "$WORKSPACE_KIND" = repository ]; then
  /opt/bialy/setup-repo.sh "$project"
fi

# The sidecar's own configuration: model servers, no approval prompts, and
# whatever decision engine this pass runs with (sidecar.env).
mkdir -p /cfg
cp "$out/providers.local.yaml" "$out/model-policy.yaml" /cfg/
printf 'rules: []\nnever_ask: true\n' > /cfg/approvals.yaml
cat <<'EOF' > /cfg/web-research-config.yaml
search_enabled: true
warming: true
guess_domains: true
warming_caps:
  seed_warms_per_hour: 64
  turn_hosts: 4
  turn_probes: 6
  seed_probes: 10
  scheduled_probes: 24
EOF
set -a
# shellcheck disable=SC1091
. "$out/sidecar.env"
set +a
fixture_pid=
if [ "${BIALY_PREFLIGHT:-}" = fixture ]; then
  python3 /opt/bialy/preflight.py configure
  export LYCAON_CONFIG_ROOT=/cfg/preflight-overlay
  (clear_provider_environment; exec python3 /opt/bialy/preflight.py fixture) >"$out/fixture.log" 2>&1 &
  fixture_pid=$!
fi
if [ -n "${PW_TRAINING_CAPTURE_DIR:-}" ]; then
  mkdir -p "$PW_TRAINING_CAPTURE_DIR"
  chmod 700 "$PW_TRAINING_CAPTURE_DIR"
fi
# Record context retention candidates on every decided request, not only past the
# watermark, so every session labels retention rows (retention.jsonl below).
export LYCAON_RETENTION_RECORD=all
# A Linux Opengrep build, which only development binaries accept.
if [ -x /opt/bialy/engine/opengrep/opengrep ]; then
  export LYCAON_OPENGREP_CANDIDATE=/opt/bialy/engine/opengrep
fi
(
  clear_provider_environment
  exec env LYCAON_ADDR=127.0.0.1:8850 LYCAON_CONFIG_DIR=/cfg /opt/bialy/bin/lycaon serve
) >"$out/sidecar.log" 2>&1 &
sidecar=$!
stop_sidecar() {
  if kill -0 "$sidecar" 2>/dev/null; then
    kill -TERM "$sidecar"
    wait "$sidecar" || true
  fi
  if [ -n "$fixture_pid" ] && kill -0 "$fixture_pid" 2>/dev/null; then
    kill -TERM "$fixture_pid"
    wait "$fixture_pid" || true
  fi
}
trap stop_sidecar EXIT
for _ in $(seq 1 120); do
  curl -sf http://127.0.0.1:8850/health >/dev/null && break
  kill -0 "$sidecar" || { echo "sidecar exited during startup"; exit 1; }
  sleep 1
done
# Hosted providers: the setup shell installs keys into the sidecar's tmpfs vault.
# The sidecar and every program it spawns started without those environment keys.
# Providers can share one key, so the keys leave the environment only once every
# provider has its own.
IFS=, read -ra hosted <<<"${BIALY_HOSTED_PROVIDERS:-}"
printf 'Authorization: Bearer %s\n' "$(cat /cfg/api.token)" > /cfg/credential.headers
chmod 600 /cfg/credential.headers
for pair in "${hosted[@]}"; do
  [ -n "$pair" ] || continue
  provider=${pair%%:*}
  key_env=${pair#*:}
  if [ -z "${!key_env:-}" ]; then echo "hosted provider $provider: $key_env is not set"; exit 1; fi
  code=$(store_provider_credential "$provider" "$key_env" /cfg/credential.headers)
  [ "$code" = 200 ] || { echo "hosted provider $provider: storing the key returned HTTP $code"; exit 1; }
done
for pair in github:GITHUB_SEARCH_API_KEY gitlab:GITLAB_SEARCH_API_KEY stackexchange:STACKEXCHANGE_API_KEY; do
  provider=${pair%%:*}
  key_env=${pair#*:}
  [ -n "${!key_env:-}" ] || continue
  code=$(store_provider_credential "$provider" "$key_env" /cfg/credential.headers web-research/providers)
  [ "$code" = 200 ] || { echo "web research provider $provider: storing the key returned HTTP $code"; exit 1; }
done
clear_provider_environment
rm /cfg/credential.headers
printf '{"ready":true}\n' > "$out/ready.json"
if [ -n "${BIALY_PREFLIGHT:-}" ]; then
  python3 /opt/bialy/preflight.py before "$project"
fi

while IFS= read -r task; do
  [ -n "$task" ] || continue
  printf '%s\n' "$task" > "$out/task.part"
  command=(/opt/bialy/bin/lycaon-debug decide generate --addr http://127.0.0.1:8850 --token "$(cat /cfg/api.token)"
    --tasks "$out/task.part" --project "$project" --project-name "$WORKSPACE" --manifest "$out/task-manifest.part"
    --unattended /opt/bialy/unattended.yaml --timeout "$PROMPT_TIMEOUT")
  if [ -n "${TASK_TIMEOUT:-}" ]; then
    command=(timeout --signal=TERM --kill-after=10s "$TASK_TIMEOUT" "${command[@]}")
  fi
  code=0
  "${command[@]}" || code=$?
  if [ -f "$out/task-manifest.part" ]; then
    cat "$out/task-manifest.part" >> "$out/manifest.jsonl"
    rm "$out/task-manifest.part"
  fi
  if [ "$code" -ne 0 ]; then
    printf '{"code":"driver_exit","exit_code":%d}\n' "$code" > "$out/task-fault.json"
    break
  fi
  if ! python3 -c 'import json,sys; sys.exit(any(json.loads(l)["status"] != "settled" for l in open(sys.argv[1])))' "$out/manifest.jsonl"; then
    printf '{"code":"task_failed"}\n' > "$out/task-fault.json"
    break
  fi
done < "$out/tasks.jsonl"
if [ -n "${BIALY_PREFLIGHT:-}" ]; then
  python3 /opt/bialy/preflight.py after "$project"
fi

stop_sidecar
trap - EXIT

# A shard is finished only when every task reached the manifest; one the driver gave up
# on writes no rows.jsonl, so the run counts it failed and drives it again.
planned=$(grep -c . "$out/tasks.jsonl")
driven=$(grep -c . "$out/manifest.jsonl" 2>/dev/null || true)
if [ "${driven:-0}" -ne "$planned" ]; then
  echo "shard incomplete: $driven of $planned tasks driven"
  exit 1
fi

# Rows from the sessions this shard drove, with the retention and request rows beside
# them, and the store they came from, so a later label change can export again
# without driving anything.
grep -o '"root_session":"[^"]*"' "$out/manifest.jsonl" | cut -d'"' -f4 >"$out/roots.txt" || true
/opt/bialy/bin/lycaon-debug decide export --db /cfg/store.db --roots "$out/roots.txt" --out "$out/rows.part" \
  --retention "$out/retention.part" --requests "$out/requests.part"
sqlite3 /cfg/store.db ".backup '$out/store.db'"
if [ "${BIALY_CAMPAIGN_CAPTURE:-}" = 1 ] && [ -d /cfg/projects ]; then
  cp -R /cfg/projects "$out/projects"
fi
mv "$out/retention.part" "$out/retention.jsonl"
mv "$out/requests.part" "$out/requests.jsonl"
mv "$out/rows.part" "$out/rows.jsonl"
echo "shard done $(date -u +%FT%TZ)"

