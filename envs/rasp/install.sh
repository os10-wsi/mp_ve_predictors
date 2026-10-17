#!/usr/bin/env bash
set -euo pipefail
bash "$(cd "$(dirname "$0")/../.." && pwd)/scripts/install_rasp.sh"
echo "RaSP prepares structures with reduce and pdbfixer, which upstream installs"
echo "through conda. Without them, use the Colab notebook in the repository to"
echo "reproduce their environment."
