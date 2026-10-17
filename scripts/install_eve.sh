#!/usr/bin/env bash
# EVE and popEVE
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

echo "=== EVE and popEVE ==="
clone_repo "https://github.com/OATML-Markslab/EVE.git" "EVE"
record_provenance "EVE"
clone_repo "https://github.com/debbiemarkslab/popEVE.git" "popEVE"
record_provenance "popEVE"

echo
echo "EVE trains a VAE per protein: budget GPU hours per protein, not per variant."
echo "popEVE: download the proteome-wide precomputed scores from"
echo "  https://pop.evemodel.org/   into $WEIGHTS/popeve/"
