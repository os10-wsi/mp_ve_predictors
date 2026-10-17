#!/usr/bin/env bash
# Build plmc, which EVcouplings calls to fit the Potts model.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
THIRD_PARTY="${VEPBENCH_THIRD_PARTY:-$ROOT/third_party}"
mkdir -p "$THIRD_PARTY"
[ -d "$THIRD_PARTY/plmc" ] || git clone --depth 1 https://github.com/debbiemarkslab/plmc.git "$THIRD_PARTY/plmc"
make -C "$THIRD_PARTY/plmc" all-openmp || make -C "$THIRD_PARTY/plmc" all
echo "plmc built at $THIRD_PARTY/plmc/bin/plmc; add it to PATH"
