#!/usr/bin/env bash
# GEMME
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

echo "=== GEMME ==="
clone_repo "https://github.com/elolaine/GEMME.git" "GEMME"
record_provenance "GEMME"

require java
echo
echo "JET2 is bundled in the repository and needs Java 8 or newer."
echo "Java found: $(java -version 2>&1 | head -1)"
