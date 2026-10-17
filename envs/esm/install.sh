#!/usr/bin/env bash
set -euo pipefail
echo "ESM-IF1 additionally needs torch-scatter and torch-sparse built against"
echo "your torch. If they are missing, install them from the PyG index:"
echo "  uv pip install --python .venv/bin/python torch-scatter torch-sparse \\"
echo "    -f https://data.pyg.org/whl/torch-\$(.venv/bin/python -c 'import torch;print(torch.__version__)').html"
echo
echo "ESM3 and ESM C are gated. Accept the licence at"
echo "  https://huggingface.co/EvolutionaryScale/esm3-sm-open-v1"
echo "and export HF_TOKEN before running them."
