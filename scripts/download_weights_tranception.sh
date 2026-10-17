#!/usr/bin/env bash
# Tranception checkpoints.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

URL="https://marks.hms.harvard.edu/tranception/Tranception_Large_checkpoint.zip"
confirm_download "3.1 GB" "Tranception Large checkpoint" || exit 1
report_free_space

DEST="$WEIGHTS/tranception"
mkdir -p "$DEST"
cd "$DEST"
curl -fL -o Tranception_Large_checkpoint.zip "$URL"
unzip -o Tranception_Large_checkpoint.zip
rm -f Tranception_Large_checkpoint.zip
echo "Checkpoint at $DEST"
