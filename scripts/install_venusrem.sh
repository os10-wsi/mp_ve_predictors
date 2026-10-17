#!/usr/bin/env bash
# VenusREM
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

echo "=== VenusREM ==="
clone_repo "https://github.com/ai4protein/VenusREM.git" "VenusREM"
record_provenance "VenusREM"

echo
echo "VenusREM: Tan et al., Bioinformatics 2025;41(Suppl 1):i401-i409."
echo "A mirror exists at https://github.com/tyang816/VenusREM if the above moves."
