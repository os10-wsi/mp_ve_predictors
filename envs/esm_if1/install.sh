#!/usr/bin/env bash
# torch-scatter and torch-sparse come from the PyG wheel index, which is keyed
# by the exact torch build.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
PY="$HERE/.venv/bin/python"
TORCH="$("$PY" -c 'import torch; print(torch.__version__)')"
INDEX="https://data.pyg.org/whl/torch-${TORCH}.html"

echo "installing torch-scatter and torch-sparse for torch ${TORCH}"
if ! uv pip install --python "$PY" torch-scatter torch-sparse -f "$INDEX"; then
  cat <<MESSAGE

Could not install torch-scatter/torch-sparse from ${INDEX}.

  * If the host is unreachable, allow data.pyg.org on your network.
  * If no wheel exists for this torch build, either install a torch version
    that PyG publishes wheels for, or build from source, which needs a CUDA
    toolkit matching your driver.

Without them ESM-IF1 reports itself unavailable; everything else still runs.
MESSAGE
  exit 1
fi
echo "ESM-IF1 is ready. Fetch its weights with:"
echo "  bash scripts/download_weights_esm.sh esm-if1"
