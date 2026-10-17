#!/usr/bin/env bash
set -euo pipefail
echo "ESM-IF1 is not in this environment: it needs torch-scatter and"
echo "torch-sparse, whose wheels lag several torch releases. Install it with:"
echo "  python -m vepbench install-env esm_if1"
echo
echo "ESM3 and ESM C are gated. Accept the licence at"
echo "  https://huggingface.co/EvolutionaryScale/esm3-sm-open-v1"
echo "and export HF_TOKEN before running them."
