#!/usr/bin/env bash
set -euo pipefail
bash "$(cd "$(dirname "$0")/../.." && pwd)/scripts/install_stability_dg.sh"
echo "The ESM3 half needs the EvolutionaryScale licence and HF_TOKEN."
