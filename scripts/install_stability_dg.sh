#!/usr/bin/env bash
# SaProt-dG and ESM3-dG
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

echo "=== SaProt-dG and ESM3-dG ==="
clone_repo "https://github.com/yehlincho/absolute-stability-predictor.git" "absolute-stability-predictor"
record_provenance "absolute-stability-predictor"

echo
echo "These are supervised dG and ddG heads on SaProt 650M and ESM3, not"
echo "masked-marginal likelihood scores. Weights: Yehlin/absolute-stability on"
echo "Hugging Face. The ESM3 half needs the EvolutionaryScale licence and HF_TOKEN."
