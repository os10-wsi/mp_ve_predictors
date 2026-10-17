#!/usr/bin/env bash
# EVmutation / EVcouplings
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

echo "=== EVmutation / EVcouplings ==="

echo "Installed from envs/evcouplings/requirements.txt, which also builds plmc:"
echo "  python3 -m vepbench install-env evcouplings"
