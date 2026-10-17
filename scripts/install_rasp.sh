#!/usr/bin/env bash
# RaSP
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

echo "=== RaSP ==="
clone_repo "https://github.com/KULL-Centre/_2022_ML-ddG-Blaabjerg.git" "RaSP"
record_provenance "RaSP"

echo
echo "The cavity model and ddG head weights are inside the repository."
echo "Structure preparation needs reduce and pdbfixer, which upstream installs"
echo "through conda."
