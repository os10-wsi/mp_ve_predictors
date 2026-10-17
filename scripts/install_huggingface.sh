#!/usr/bin/env bash
# Hugging Face language models
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

echo "=== Hugging Face language models ==="

echo "Install the environment first:"
echo "  python3 -m vepbench install-env huggingface"
echo
echo "Weights download on first use into HF_HOME (default ~/.cache/huggingface):"
echo "  SaProt_650M_AF2      2.6 GB"
echo "  SaProt_1.3B_AF2      5 GB"
echo "  ProSST-2048          0.5 GB"
echo "  prot_t5_xl_uniref50  11 GB"
echo "  ankh-large           7 GB"
echo "Set HF_HOME to put them somewhere with room."
echo
echo "SaProt also needs foldseek: bash scripts/install_foldseek.sh"
