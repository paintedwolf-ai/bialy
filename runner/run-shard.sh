#!/bin/bash
# run-shard.sh: drive one shard of tasks through one sidecar, then export its rows.
#
# Mounts: /repos (read-only seed clones), /shard (this shard's tasks.jsonl and
# sidecar.env; its outputs land here), /cache (this runner's view of the package cache).
# Environment: WORKSPACE and WORKSPACE_KIND (repository or stack), COMMIT for a
# repository, PROMPT_TIMEOUT, and the model servers in /shard/providers.local.yaml and
# /shard/model-policy.yaml.
set -euo pipefail
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
    # Greenfield: an empty directory with git initialised and nothing else, as a
    # developer starting a new project has.
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
set -a
# shellcheck disable=SC1091
. "$out/sidecar.env"
set +a
# A Linux Opengrep build, which only development binaries accept; without it the
# sidecar reports the scanner unavailable.
if [ -x /opt/bialy/engine/opengrep/opengrep ]; then
  export LYCAON_OPENGREP_CANDIDATE=/opt/bialy/engine/opengrep
fi
LYCAON_ADDR=127.0.0.1:8850 LYCAON_CONFIG_DIR=/cfg /opt/bialy/bin/lycaon serve >"$out/sidecar.log" 2>&1 &
sidecar=$!
stop_sidecar() {
  if kill -0 "$sidecar" 2>/dev/null; then
    kill -TERM "$sidecar"
    wait "$sidecar" || true
  fi
}
trap stop_sidecar EXIT
for _ in $(seq 1 120); do
  curl -sf http://127.0.0.1:8850/health >/dev/null && break
  kill -0 "$sidecar" || { echo "sidecar exited during startup"; exit 1; }
  sleep 1
done
# Hosted providers: the key goes from this container's environment into the sidecar's
# credential store on its tmpfs, then out of the environment the sessions run in.
IFS=, read -ra hosted <<<"${BIALY_HOSTED_PROVIDERS:-}"
for pair in "${hosted[@]}"; do
  [ -n "$pair" ] || continue
  provider=${pair%%:*}
  key_env=${pair#*:}
  if [ -z "${!key_env:-}" ]; then echo "hosted provider $provider: $key_env is not set"; exit 1; fi
  body=$(printf '{"api_key": "%s"}' "${!key_env}")
  code=$(curl -s -o /dev/null -w '%{http_code}' -X PUT -H "Authorization: Bearer $(cat /cfg/api.token)" \
    -H 'Content-Type: application/json' --data-binary "$body" "http://127.0.0.1:8850/v1/providers/$provider/credential")
  [ "$code" = 200 ] || { echo "hosted provider $provider: storing the key returned HTTP $code"; exit 1; }
  unset "$key_env"
done

/opt/bialy/bin/lycaon-debug decide generate --addr http://127.0.0.1:8850 --token "$(cat /cfg/api.token)" \
  --tasks "$out/tasks.jsonl" --project "$project" --project-name "$WORKSPACE" --manifest "$out/manifest.jsonl" \
  --unattended /opt/bialy/unattended.yaml --timeout "$PROMPT_TIMEOUT" || echo "driver exited $?"
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

# Rows from the sessions this shard drove, and the store they came from, so a
# later label change can export again without driving anything.
grep -o '"root_session":"[^"]*"' "$out/manifest.jsonl" | cut -d'"' -f4 >"$out/roots.txt" || true
/opt/bialy/bin/lycaon-debug decide export --db /cfg/store.db --roots "$out/roots.txt" --out "$out/rows.part"
sqlite3 /cfg/store.db ".backup '$out/store.db'"
mv "$out/rows.part" "$out/rows.jsonl"
echo "shard done $(date -u +%FT%TZ)"
