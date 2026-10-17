#!/usr/bin/env bash
# ProMEP
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

echo "=== ProMEP ==="
clone_repo "https://github.com/wenjiegroup/ProMEP.git" "ProMEP"
record_provenance "ProMEP"

echo
echo "ProMEP: Cheng et al., Cell Research 2024;34:630-647."
echo "It needs DGL and an SE(3)-Transformer build. If DGL fails, install the"
echo "build matching your CUDA version from https://www.dgl.ai/pages/start.html"
