#!/usr/bin/env bash
set -euo pipefail
bash "$(cd "$(dirname "$0")/../.." && pwd)/scripts/install_rasp.sh"
if ! command -v reduce >/dev/null 2>&1 && [ ! -x "$(dirname "$0")/.venv/bin/reduce" ]; then
  echo "RaSP prepares structures with reduce, openmm and pdbfixer, which have no"
  echo "PyPI wheels. Rebuild this environment with --backend conda to get them,"
  echo "or install them yourself:"
  echo "  conda install -c conda-forge -c bioconda reduce openmm pdbfixer"
fi
