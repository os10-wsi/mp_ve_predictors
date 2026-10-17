#!/usr/bin/env bash
# End-to-end demonstration on phospholamban (P26678): a 52-residue
# single-pass membrane protein, 988 substitutions.
#
# Uses only the two baselines, so it needs no model weights and no GPU. It
# does need network access once, to fetch the UniProt entry and the AlphaFold
# model; after that the cache makes it repeatable offline.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
ACCESSION=P26678
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

echo "=== 1. predict every substitution ==="
python -m vepbench run --uniprot "$ACCESSION" --predictors blosum62_baseline \
  --msa-backend none

echo
echo "=== 2. build a toy DMS table from the predictions ==="
# Stands in for your measured data. Two rows are deliberately broken, to show
# that unmatched rows are reported rather than dropped.
python - "$WORK/dms.csv" <<'PYTHON'
import sys
import numpy as np
import pandas as pd

wide = pd.read_csv("outputs/P26678/predictions_wide.csv")
rng = np.random.default_rng(0)
sample = wide.sample(300, random_state=0).copy()
sample["abundance"] = (
    0.35 * sample["blosum62_baseline"] + rng.normal(scale=1.0, size=len(sample))
)
broken = pd.DataFrame({
    "variant": ["Q4A", "M1A,K2A"],       # wrong wild type, and a double mutant
    "abundance": [0.0, 0.0],
})
table = pd.concat([sample[["variant", "abundance"]], broken], ignore_index=True)
table.to_csv(sys.argv[1], index=False)
print(f"wrote {sys.argv[1]} ({len(table)} rows)")
PYTHON

echo
echo "=== 3. merge onto the DMS table ==="
python -m vepbench merge --dms "$WORK/dms.csv" --uniprot "$ACCESSION" \
  --variant-col variant --out "$WORK/merged.csv"

echo
echo "=== 4. benchmark against the measured column ==="
python -m vepbench benchmark --merged "$WORK/merged.csv" --score-col abundance \
  --n-boot 500 --out "$WORK/summary.csv"

echo
echo "=== 5. what the run recorded ==="
python -m vepbench status --uniprot "$ACCESSION"
python - <<'PYTHON'
import json
metadata = json.load(open("outputs/P26678/metadata.json"))
print(f"\nsequence length : {metadata['uniprot']['sequence_length']}")
print(f"variants        : {metadata['n_variants']}")
structure = metadata.get("structure")
if structure:
    print(f"structure       : {structure['source']} {structure['version']}, "
          f"coverage {structure['uniprot_coverage']:.0%}, "
          f"{structure['mapping_note']}")
print("caveats         :")
for caveat in metadata["caveats"]:
    print(f"  - {caveat}")
PYTHON

echo
echo "Outputs are in outputs/$ACCESSION/. The merged table and benchmark"
echo "summary went to a temporary directory and are now removed; pass your own"
echo "--out paths to keep them."
