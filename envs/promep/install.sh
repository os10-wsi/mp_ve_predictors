#!/usr/bin/env bash
set -euo pipefail
bash "$(cd "$(dirname "$0")/../.." && pwd)/scripts/install_promep.sh"
echo "If DGL fails, install the build matching your CUDA version from"
echo "  https://www.dgl.ai/pages/start.html"
