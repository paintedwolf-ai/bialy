#!/bin/bash
set -euo pipefail
mkdir -p /home/user /cfg /work /cache
chown 1000:1000 /home/user /cfg /work /cache
chown -R 1000:1000 /shard /work
export HOME=/home/user USER=user LOGNAME=user
if [ "$#" -eq 0 ]; then set -- /opt/bialy/run-shard.sh; fi
exec setpriv --reuid=1000 --regid=1000 --clear-groups --bounding-set=-all \
  --inh-caps=-all --ambient-caps=-all --no-new-privs "$@"
