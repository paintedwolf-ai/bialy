#!/bin/bash
# jobctl.sh start NAME -- COMMAND... | stop NAME | status | wait NAME...
# Every job runs in its own session and process group, recorded with its start time, so
# stop reaches the whole group (a wrapper and every child) and a recycled PID is never
# mistaken for the job. A name that is still running cannot be started again.
set -euo pipefail
J=${R:-/scratch/bialy}/jobs; mkdir -p $J
starttime() { awk '{print $22}' /proc/$1/stat 2>/dev/null; }
alive() { [ -f $J/$1.pid ] || return 1; read -r pid start < $J/$1.pid; [ "$(starttime $pid)" = "$start" ]; }
case ${1:-} in
  start)
    name=$2; shift 3
    if alive $name; then echo "jobctl: $name is already running" >&2; exit 1; fi
    # The log is created exclusively: a name is used once, and nothing truncates its log.
    ( set -C; : > $J/$name.log ) 2>/dev/null || { echo "jobctl: $name has run before; use a new name" >&2; exit 1; }
    setsid bash -c '
      state=$1; shift
      echo "$$ $(awk "{print \$22}" /proc/$$/stat)" > "$state.pid"
      "$@"
      rc=$?
      printf "%s\n" "$rc" > "$state.exit.tmp"
      mv "$state.exit.tmp" "$state.exit"
      exit "$rc"
    ' _ "$J/$name" "$@" >> "$J/$name.log" 2>&1 < /dev/null &
    for _ in $(seq 50); do [ -s $J/$name.pid ] && break; sleep 0.1; done
    echo "jobctl: started $name pid $(cut -d" " -f1 $J/$name.pid)" ;;
  stop)
    name=$2
    if ! alive $name; then echo "jobctl: $name is not running"; exit 0; fi
    read -r pid _ < $J/$name.pid
    kill -TERM -- -$pid 2>/dev/null || true
    for _ in $(seq 30); do pgrep -g $pid >/dev/null || break; sleep 1; done
    pgrep -g $pid >/dev/null && kill -KILL -- -$pid 2>/dev/null || true
    echo "jobctl: stopped $name (group $pid)" ;;
  status)
    for f in $J/*.pid; do [ -e "$f" ] || continue; name=$(basename $f .pid)
      if alive $name; then state=running; elif [ -f "$J/$name.exit" ]; then state="exit=$(cat "$J/$name.exit")"; else state=interrupted; fi
      echo "$name $state $(tail -c 300 $J/$name.log | tr '\n' ' ' | tail -c 120)"; done ;;
  wait)
    shift; while :; do n=0; for name in "$@"; do alive $name && n=$((n+1)); done; [ $n = 0 ] && break; sleep 30; done
    rc=0
    for name in "$@"; do
      if [ ! -f "$J/$name.exit" ]; then echo "$name interrupted"; rc=1
      elif [ "$(cat "$J/$name.exit")" != 0 ]; then echo "$name failed: $(cat "$J/$name.exit")"; rc=1
      else echo "$name completed"; fi
    done
    exit "$rc" ;;
  *) echo "usage: jobctl.sh start NAME -- CMD... | stop NAME | status | wait NAME..." >&2; exit 2 ;;
esac
