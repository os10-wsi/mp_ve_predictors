#!/usr/bin/env bash
# ESM family
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

echo "=== ESM family ==="

echo "Install the environment first:"
echo "  python3 -m vepbench install-env esm"
echo
echo "ESM-1v and ESM-2 weights come from Hugging Face on first use. ESM-IF1 is"
echo "the one model still distributed only through fair-esm, so it downloads"
echo "from dl.fbaipublicfiles.com into $WEIGHTS/torch_hub."
echo
echo "Sizes: ESM-1v 2.5 GB per replicate (five of them), ESM-2 650M 2.5 GB,"
echo "ESM-2 3B 11 GB, ESM-IF1 0.5 GB."
echo "Fetch them deliberately: bash scripts/download_weights_esm.sh esm1v"
echo
echo "ESM3 and ESM C are gated: accept the licence at"
echo "  https://huggingface.co/EvolutionaryScale/esm3-sm-open-v1"
echo "and export HF_TOKEN."
