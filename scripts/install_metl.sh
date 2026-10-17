#!/usr/bin/env bash
# METL
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

echo "=== METL ==="

echo "  python -m vepbench install-env metl"
echo
echo "METL: Gelman et al., Nature Methods 2025. Checkpoints download on first"
echo "use; METL-G-20M-3D is about 0.2 GB. Local fine-tuned models exist only for"
echo "proteins the authors simulated, none of them a membrane protein."
