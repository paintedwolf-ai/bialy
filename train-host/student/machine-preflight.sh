#!/bin/bash
# Decide, before any work starts on a training machine, whether it can run a student job:
# the tools bootstrap calls, uv at its configured path, and room for the tier.
#   machine-preflight.sh check|install WORKSPACE UV DEVICE DISK_GIB MEMORY_GIB GPU_MEMORY_GIB
# `install` adds what is missing and is only used on a rented machine; `check` changes nothing.
# Prints one JSON line of measurements and exits 1 when the machine falls short.
set -euo pipefail
mode=$1 workspace=$2 uv=$3 device=$4 need_disk=$5 need_memory=$6 need_gpu=$7
UV_VERSION=0.13.8
declare -A package=([git]=git [python3]=python3 [rsync]=rsync [cmake]=cmake [c++]=g++ [make]=make [curl]=curl)
missing() {
  local tool
  for tool in "${!package[@]}"; do command -v "$tool" >/dev/null || echo "$tool"; done | sort
}
if [ "$mode" = install ] && [ "$(id -u)" = 0 ] && command -v apt-get >/dev/null; then
  export DEBIAN_FRONTEND=noninteractive
  # The dead-man switch may be installing its own client at boot and holding the package lock.
  for _ in 1 2 3 4 5 6; do
    mapfile -t absent < <(missing)
    [ ${#absent[@]} -gt 0 ] || break
    packages=(ca-certificates)
    for tool in "${absent[@]}"; do packages+=("${package[$tool]}"); done
    { apt-get update -q && apt-get -o DPkg::Lock::Timeout=300 install -y -q --no-install-recommends "${packages[@]}"; } >&2 || sleep 10
  done
  if [ ! -x "$uv" ] && command -v curl >/dev/null; then
    curl -LsSf "https://astral.sh/uv/$UV_VERSION/install.sh" | env UV_UNMANAGED_INSTALL="$(dirname "$uv")" sh >&2 || true
  fi
fi
problems=()
mapfile -t absent < <(missing)
for tool in "${absent[@]}"; do problems+=("missing tool: $tool"); done
[ -x "$uv" ] || problems+=("uv is not executable at $uv")
mkdir -p "$workspace"
disk=$(($(df -Pk "$workspace" | awk 'NR==2 {print $4}') / 1048576))
memory=$(($(awk '/^MemTotal:/ {print $2}' /proc/meminfo) / 1048576))
gpu=0
if [[ $device == cuda* ]]; then
  index=${device#cuda}
  index=${index#:}
  if mebibytes=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i "${index:-0}" 2>/dev/null); then
    gpu=$((mebibytes / 1024))
  else
    problems+=("no GPU answers as $device")
  fi
  [ "$gpu" -ge "$need_gpu" ] || problems+=("GPU memory is $gpu GiB; the tier needs $need_gpu")
fi
[ "$disk" -ge "$need_disk" ] || problems+=("workspace has $disk GiB free; the tier needs $need_disk")
[ "$memory" -ge "$need_memory" ] || problems+=("memory is $memory GiB; the tier needs $need_memory")
list=""
for problem in "${problems[@]}"; do list+="${list:+,}\"${problem//\"/}\""; done
printf '{"disk_gib":%d,"memory_gib":%d,"gpu_memory_gib":%d,"problems":[%s]}\n' "$disk" "$memory" "$gpu" "$list"
[ ${#problems[@]} -eq 0 ]
