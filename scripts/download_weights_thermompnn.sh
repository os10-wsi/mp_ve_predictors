#!/usr/bin/env bash
# ThermoMPNN checkpoints ship inside the repository. This locates them and
# reports what it found, rather than guessing a URL.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

REPO="${THERMOMPNN_DIR:-$THIRD_PARTY/ThermoMPNN}"
[ -d "$REPO" ] || { echo "not cloned; run bash scripts/install_proteinmpnn.sh" >&2; exit 1; }

echo "Checkpoints found in $REPO:"
FOUND=0
while IFS= read -r f; do
  printf '  %-70s %s\n' "$f" "$(du -h "$f" | cut -f1)"; FOUND=1
done < <(find "$REPO" \( -name '*.pt' -o -name '*.ckpt' \))

if [ "$FOUND" = "0" ]; then
  echo
  echo "No checkpoints in the checkout. Recent releases host them separately;"
  echo "see the repository README, then pass the file with the predictor's"
  echo "checkpoint option."
fi
