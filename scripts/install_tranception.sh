#!/usr/bin/env bash
# Tranception and TranceptEVE
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

echo "=== Tranception and TranceptEVE ==="
clone_repo "https://github.com/OATML-Markslab/Tranception.git" "Tranception"
record_provenance "Tranception"

echo
echo "Checkpoints: bash scripts/download_weights_tranception.sh"
echo "TranceptEVE additionally needs a trained EVE model for the same protein."
