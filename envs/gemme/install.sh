#!/usr/bin/env bash
set -euo pipefail
command -v java >/dev/null || { echo "GEMME needs a Java runtime (8+)"; exit 1; }
bash "$(cd "$(dirname "$0")/../.." && pwd)/scripts/install_gemme.sh"
