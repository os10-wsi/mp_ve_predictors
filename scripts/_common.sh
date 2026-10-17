#!/usr/bin/env bash
# Shared helpers for the install scripts.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
THIRD_PARTY="${VEPBENCH_THIRD_PARTY:-$REPO_ROOT/third_party}"
WEIGHTS="${VEPBENCH_WEIGHTS:-$REPO_ROOT/data/weights}"
mkdir -p "$THIRD_PARTY" "$WEIGHTS"

# Hugging Face defaults to ~/.cache/huggingface. On a cluster that is a small
# home quota, and a couple of models fill it, so default the cache to sit with
# the other weights instead. An explicit HF_HOME always wins.
export HF_HOME="${HF_HOME:-$WEIGHTS/huggingface}"
mkdir -p "$HF_HOME"

clone_repo() {
  local url="$1" name="$2" ref="${3:-}"
  local target="$THIRD_PARTY/$name"
  if [ -d "$target/.git" ]; then
    echo "  $name already cloned at $target"
  else
    echo "  cloning $url"
    git clone --depth 1 ${ref:+--branch "$ref"} "$url" "$target"
  fi
  echo "  pinned at $(git -C "$target" rev-parse --short HEAD 2>/dev/null || echo unknown)"
}

# Record exactly what was installed, so a run can be reproduced.
record_provenance() {
  local name="$1" target="$THIRD_PARTY/$1"
  printf '%s\t%s\t%s\t%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$name" "$target" \
    "$(git -C "$target" rev-parse HEAD 2>/dev/null || echo unknown)" \
    >> "$WEIGHTS/provenance.tsv"
}

require() {
  command -v "$1" >/dev/null 2>&1 || { echo "error: $1 is required but not on PATH" >&2; exit 1; }
}

# confirm_download <size> <what> [destination]
# Large downloads never start without saying how big they are and where they
# go. Not everything lands in $WEIGHTS: Hugging Face models go to HF_HOME, so
# the caller can say where. Set VEPBENCH_ASSUME_YES=1 to skip the prompt.
confirm_download() {
  echo
  echo "About to download: $2"
  echo "Approximate size:  $1"
  echo "Destination:       ${3:-$WEIGHTS}"
  if [ "${VEPBENCH_ASSUME_YES:-0}" = "1" ]; then
    echo "VEPBENCH_ASSUME_YES is set, continuing."
    return 0
  fi
  if [ ! -t 0 ]; then
    echo "Not interactive and VEPBENCH_ASSUME_YES is not set; stopping." >&2
    return 1
  fi
  read -r -p "Continue? [y/N] " answer
  case "$answer" in [yY]|[yY][eE][sS]) return 0 ;; *) echo "Stopped."; return 1 ;; esac
}

report_free_space() {
  echo "Free space at $WEIGHTS: $(df -Ph "$WEIGHTS" | awk 'NR==2 {print $4}')"
}

# require_env <name>
# Refuse to use an environment that is absent or half-built. The interpreter
# appears in the first step of an install, before any requirement, so its
# presence alone proves nothing; the marker is written only on success.
require_env() {
  local name="$1"
  local venv="$REPO_ROOT/envs/$name/.venv"
  if [ ! -x "$venv/bin/python" ]; then
    echo "environment '$name' is not installed. Run:" >&2
    echo "  python3 -m vepbench install-env $name" >&2
    return 1
  fi
  if [ ! -f "$venv/.vepbench-install-complete" ]; then
    echo "environment '$name' exists but its install did not finish, so some" >&2
    echo "dependencies are missing. Run:" >&2
    echo "  python3 -m vepbench install-env $name" >&2
    return 1
  fi
  return 0
}

# free_space_at <path>
# df fails on a path that does not exist yet, which is normal for a cache
# directory that has never been written to, so walk up to one that does.
free_space_at() {
  local path="$1"
  while [ -n "$path" ] && [ "$path" != "/" ] && [ ! -d "$path" ]; do
    path="$(dirname "$path")"
  done
  df -Ph "${path:-/}" 2>/dev/null | awk 'NR==2 {print $4}'
}

# require_space <needed GB> <path>
# Refuse a download that cannot fit, rather than filling the filesystem and
# dying partway with ENOSPC.
require_space() {
  local needed="$1" path="$2"
  local available
  available="$(df -P --block-size=1G "$path" 2>/dev/null | awk 'NR==2 {print $4}')"
  if [ -z "$available" ]; then
    echo "could not measure free space at $path; continuing" >&2
    return 0
  fi
  if awk -v a="$available" -v n="$needed" 'BEGIN {exit !(a < n)}'; then
    cat >&2 <<MESSAGE

Not enough space at $path
  needed:    ${needed} GB
  available: ${available} GB

Point the cache somewhere larger and use the same value when you run:
  export HF_HOME=/path/with/room
MESSAGE
    return 1
  fi
  return 0
}
