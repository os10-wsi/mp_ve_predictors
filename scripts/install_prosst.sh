#!/usr/bin/env bash
# ProSST
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

echo "=== ProSST ==="
clone_repo "https://github.com/ai4protein/ProSST.git" "ProSST"
record_provenance "ProSST"

echo
echo "The structure quantizer lives in the repository; the language model is"
echo "pulled from Hugging Face (AI4Protein/ProSST-2048) on first use."
