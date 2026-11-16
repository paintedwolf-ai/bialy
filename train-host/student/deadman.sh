#!/bin/bash
# Dead-man switch of a rented machine: it ends its own rental at a deadline that only
# the controller's heartbeats move, and never past a limit fixed at the first arming.
#   arm PROVIDER TTL LIFETIME | extend TTL | watch
set -u
state=${PW_DEADMAN_STATE:-/var/lib/pw-deadman}
api=${PW_DEADMAN_API:-https://console.vast.ai/api/v0}
self=$(readlink -f "$0")
put() { printf '%s\n' "$2" >"$state/$1.new" && mv "$state/$1.new" "$state/$1"; }
# An unreadable number reads as 0, so a damaged deadline has already passed.
get() { local v; v=$(cat "$state/$1" 2>/dev/null); case $v in '' | *[!0-9]*) v=0 ;; esac; echo "$v"; }
request() {
  local url="$api/instances/$CONTAINER_ID/"
  if command -v curl >/dev/null; then
    curl -fsS -m 30 -X "$1" -H "Authorization: Bearer $CONTAINER_API_KEY" -H 'Content-Type: application/json' ${2:+-d "$2"} "$url"
  else
    python3 -c 'import sys,urllib.request as u
u.urlopen(u.Request(sys.argv[2],method=sys.argv[1],data=sys.argv[4].encode() or None,headers={"Authorization":"Bearer "+sys.argv[3],"Content-Type":"application/json"}),timeout=30)' "$1" "$url" "$CONTAINER_API_KEY" "${2:-}"
  fi
}
end_rental() {
  if [ "$(cat "$state/provider")" = vastai ]; then
    # shellcheck source=/dev/null
    . "$state/identity"
    request DELETE || request PUT '{"state":"stopped"}'
  else
    systemctl poweroff || poweroff -f
  fi
}
start() {
  if [ -d /run/systemd/system ]; then
    printf '[Service]\nExecStart=%s watch\nRestart=always\nOOMScoreAdjust=-1000\n[Install]\nWantedBy=multi-user.target\n' "$self" >/etc/systemd/system/pw-deadman.service
    systemctl daemon-reload && systemctl enable --now pw-deadman.service
  else
    setsid nohup "$self" watch >/dev/null 2>&1 </dev/null &
  fi
}
case ${1:-} in
arm)
  mkdir -p "$state" && chmod 700 "$state"
  now=$(date +%s)
  put provider "$2"
  [ -s "$state/limit" ] || put limit $((now + $4))
  [ -s "$state/deadline" ] || put deadline $((now + $3))
  if [ "$2" = vastai ]; then
    (umask 077 && printf 'CONTAINER_ID=%q\nCONTAINER_API_KEY=%q\n' "$CONTAINER_ID" "$CONTAINER_API_KEY" >"$state/identity")
    command -v curl >/dev/null || command -v python3 >/dev/null ||
      (apt-get update -q && apt-get install -y -q --no-install-recommends curl ca-certificates) >/dev/null 2>&1 &
  fi
  start
  ;;
extend)
  now=$(date +%s) limit=$(get limit) deadline=$(($(date +%s) + $2))
  [ "$deadline" -le "$limit" ] || deadline=$limit
  [ -e "$state/expired" ] || put deadline "$deadline"
  [ $((now - $(get alive))) -le 30 ] || start
  can=false
  { [ "$(cat "$state/provider")" != vastai ] || { [ -s "$state/identity" ] && { command -v curl || command -v python3; } >/dev/null; }; } && can=true
  printf '{"now":%d,"deadline":%d,"limit":%d,"watcher_age":%d,"expired":%s,"can_end_rental":%s}\n' \
    "$now" "$(get deadline)" "$limit" $((now - $(get alive))) "$([ -e "$state/expired" ] && echo true || echo false)" "$can"
  ;;
watch)
  echo -1000 >/proc/$$/oom_score_adj 2>/dev/null
  while :; do
    put alive "$(date +%s)"
    if [ "$(date +%s)" -lt "$(get deadline)" ] && [ ! -e "$state/expired" ]; then
      sleep 5
      continue
    fi
    put expired 1
    end_rental
    # The rental could not be ended from here: stop the work and keep trying.
    kill -KILL -1
    sleep 30
  done
  ;;
*) exit 2 ;;
esac
