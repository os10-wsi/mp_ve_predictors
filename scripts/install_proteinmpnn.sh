#!/usr/bin/env bash
# ProteinMPNN and ThermoMPNN
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

echo "=== ProteinMPNN and ThermoMPNN ==="
clone_repo "https://github.com/dauparas/ProteinMPNN.git" "ProteinMPNN"
record_provenance "ProteinMPNN"
clone_repo "https://github.com/Kuhlman-Lab/ThermoMPNN.git" "ThermoMPNN"
record_provenance "ThermoMPNN"

echo
echo "ProteinMPNN weights ship with the repository, under vanilla_model_weights/."
echo "ThermoMPNN checkpoints: bash scripts/download_weights_thermompnn.sh"
