#!/bin/bash
# run-shard.sh: drive one shard of tasks through one sidecar, then export its rows.
#
# Mounts: /repos (read-only seed clones), /shard (this shard's tasks.jsonl and
# sidecar.env; its outputs land here), /cache (this runner's view of the package cache).
# Environment: REPO, COMMIT, PROMPT_TIMEOUT, and the model servers in
# /shard/providers.local.yaml and /shard/model-policy.yaml.
set -euo pipefail
: "${REPO:?}" "${COMMIT:?}" "${PROMPT_TIMEOUT:=20m}"
out=/shard
exec >>"$out/runner.log" 2>&1
echo "shard start $(date -u +%FT%TZ) repo=$REPO commit=$COMMIT"

# A fresh checkout at the pinned commit; the seed clone is never written.
git clone -q --no-hardlinks /repos/"$REPO" /work/"$REPO"
git -C /work/"$REPO" checkout -q "$COMMIT"
git -C /work/"$REPO" config user.email runner@paintedwolf.invalid
git -C /work/"$REPO" config user.name "Painted Wolf runner"
# Project workflows a task can start; git ignores them so the checkout stays clean.
mkdir -p /work/"$REPO"/.paintedwolf
cp -R /opt/bialy/workflows /work/"$REPO"/.paintedwolf/workflows
echo /.paintedwolf/ >>/work/"$REPO"/.git/info/exclude
/opt/bialy/setup-repo.sh /work/"$REPO"

# The sidecar's own configuration: model servers, no approval prompts, and
# whatever decision engine this pass runs with (sidecar.env).
mkdir -p /cfg
cp "$out/providers.local.yaml" "$out/model-policy.yaml" /cfg/
printf 'rules: []\nnever_ask: true\n' > /cfg/approvals.yaml
set -a
# shellcheck disable=SC1091
. "$out/sidecar.env"
set +a
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

/opt/bialy/bin/lycaon-debug decide generate --addr http://127.0.0.1:8850 --token "$(cat /cfg/api.token)" \
  --tasks "$out/tasks.jsonl" --project /work/"$REPO" --project-name "$REPO" --manifest "$out/manifest.jsonl" \
  --unattended /opt/bialy/unattended.yaml --timeout "$PROMPT_TIMEOUT" || echo "driver exited $?"
stop_sidecar
trap - EXIT

# Rows from the sessions this shard drove, and the store they came from, so a
# later label change can export again without driving anything.
grep -o '"root_session":"[^"]*"' "$out/manifest.jsonl" | cut -d'"' -f4 >"$out/roots.txt" || true
/opt/bialy/bin/lycaon-debug decide export --db /cfg/store.db --roots "$out/roots.txt" --out "$out/rows.part"
sqlite3 /cfg/store.db ".backup '$out/store.db'"
mv "$out/rows.part" "$out/rows.jsonl"
echo "shard done $(date -u +%FT%TZ)"
