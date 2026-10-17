#!/usr/bin/env bash
# S3F (sequence-structure-surface)
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

echo "=== S3F (sequence-structure-surface) ==="
clone_repo "https://github.com/DeepGraphLearning/S3F.git" "S3F"
record_provenance "S3F"

echo
echo "S3F: Zhang et al., NeurIPS 2024, arXiv:2412.01108."
echo "Check the repository README for the checkpoint download; its location has"
echo "moved between releases, so it is not automated here."
