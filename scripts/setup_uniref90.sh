#!/usr/bin/env bash
# Download and prepare UniRef90 for the local jackhmmer MSA backend.
#
# This database is far too large for git. Nothing it writes is ever committed:
# it lands under data/databases/, which .gitignore already excludes. Put it on
# a fast local disk, not a network mount, because jackhmmer streams the whole
# file on every search.
#
#   bash scripts/setup_uniref90.sh                 # default location
#   VEPBENCH_DATABASES=/scratch/db bash scripts/setup_uniref90.sh
#
# Afterwards vepbench finds it automatically, or point at it explicitly:
#   export VEPBENCH_MSA_DATABASE=/scratch/db/uniref90.fasta
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

DATABASES="${VEPBENCH_DATABASES:-$REPO_ROOT/data/databases}"
mkdir -p "$DATABASES"
TARGET="$DATABASES/uniref90.fasta"
URL="https://ftp.uniprot.org/pub/databases/uniprot/uniref/uniref90/uniref90.fasta.gz"
MIRROR="https://ftp.ebi.ac.uk/pub/databases/uniprot/uniref/uniref90/uniref90.fasta.gz"

if [ -s "$TARGET" ]; then
  echo "UniRef90 is already at $TARGET ($(du -h "$TARGET" | cut -f1))"
  exit 0
fi

echo "=== UniRef90 for jackhmmer ==="
echo "Download   : about 65 GB compressed"
echo "Unpacked   : about 180 GB, which is what jackhmmer reads"
echo "Free here  : $(df -Ph "$DATABASES" | awk 'NR==2 {print $4}')"
echo "Destination: $TARGET"
echo
echo "This is NOT committed to git. data/ is gitignored, and a file this size"
echo "cannot go in a GitHub repository in any case: the hard limit is 100 MB"
echo "per file, and 2 GB per file even with Git LFS."
echo
confirm_download "65 GB compressed, about 180 GB unpacked" "UniRef90 FASTA" || exit 1

require curl
cd "$DATABASES"
echo "downloading from $URL"
if ! curl -fL --retry 3 --retry-delay 10 -C - -o uniref90.fasta.gz "$URL"; then
  echo "primary host failed, trying the EBI mirror"
  curl -fL --retry 3 --retry-delay 10 -C - -o uniref90.fasta.gz "$MIRROR"
fi

echo "decompressing (this takes a while and needs the full 180 GB)"
gunzip -c uniref90.fasta.gz > "$TARGET"
rm -f uniref90.fasta.gz

COUNT="$(grep -c '^>' "$TARGET" || true)"
echo
echo "UniRef90 ready at $TARGET"
echo "  sequences: $COUNT"
echo "  size:      $(du -h "$TARGET" | cut -f1)"
echo
echo "Use it with:"
echo "  python -m vepbench run --uniprot P26678 --predictors msa_based \\"
echo "    --msa-backend jackhmmer"
echo
echo "It is found automatically at this path. To keep it elsewhere, set"
echo "  export VEPBENCH_MSA_DATABASE=/path/to/uniref90.fasta"
echo
command -v jackhmmer >/dev/null || cat <<'NOTE'
Note: jackhmmer is not on PATH. Install HMMER, for example:
  apt-get install hmmer        # Debian or Ubuntu
  conda install -c bioconda hmmer
NOTE
