#!/usr/bin/env bash
# VESPA
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

echo "=== VESPA ==="

echo "  python3 -m vepbench install-env vespa"
echo "VESPA downloads ProtT5 weights on first use (about 11 GB)."
