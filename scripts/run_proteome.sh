#!/usr/bin/env bash
# Score one organism's reviewed proteome, end to end.
#
#   bash scripts/run_proteome.sh --taxon 10090 --name mouse
#   bash scripts/run_proteome.sh --taxon 10090 --name mouse --submit
#
# Without --submit it fetches the data and prints the bsub it would run, so
# you can read it first. Everything is resumable: the accession list, the
# UniProt entries and the per-score caches are all reused on a second run.
#
# ClinVar and AlphaMissense are human-only resources. For any other taxon
# they are skipped, and alphamissense and foldx are dropped from the scores,
# because both would report "unavailable" on every protein and foldx has no
# ClinVar positions to score at.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TAXON=""; NAME=""; RESULTS=""; SUBMIT=0; MEMBRANE=0; MAXPAR=0; MEM=10000
EXTRA_SCORES=(); HUMAN_TAXON=9606

usage() { sed -n '2,16p' "$0"; exit "${1:-0}"; }

while [ $# -gt 0 ]; do
  case "$1" in
    --taxon)         TAXON="$2"; shift 2 ;;
    --name)          NAME="$2"; shift 2 ;;
    --results)       RESULTS="$2"; shift 2 ;;
    --scores)        shift; while [ $# -gt 0 ] && [[ "$1" != --* ]]; do EXTRA_SCORES+=("$1"); shift; done ;;
    --membrane-only) MEMBRANE=1; shift ;;
    --max-parallel)  MAXPAR="$2"; shift 2 ;;
    --mem)           MEM="$2"; shift 2 ;;
    --submit)        SUBMIT=1; shift ;;
    -h|--help)       usage 0 ;;
    *) echo "unknown option: $1" >&2; usage 2 ;;
  esac
done

[ -n "$TAXON" ] || { echo "error: --taxon is required" >&2; usage 2; }
[ -n "$NAME" ] || NAME="taxon_$TAXON"
[ -n "$RESULTS" ] || RESULTS="$ROOT/results"

PYTHON="${VEPBENCH_PYTHON:-python3}"
"$PYTHON" -c 'import vepbench.cli' >/dev/null 2>&1 || {
  echo "error: $PYTHON cannot import vepbench. Set VEPBENCH_PYTHON." >&2; exit 1; }

export VEPBENCH_OUTPUTS="$RESULTS/$NAME"
mkdir -p "$VEPBENCH_OUTPUTS"
LIST="$VEPBENCH_OUTPUTS/accessions.txt"

# --- 1. the accession list ------------------------------------------------
QUERY="(organism_id:$TAXON) AND (reviewed:true)"
if [ "$MEMBRANE" = "1" ]; then QUERY="$QUERY AND (keyword:KW-0812)"; fi

if [ -s "$LIST" ]; then
  echo "reusing $LIST ($(grep -c . "$LIST") accessions)"
else
  echo "fetching the accession list for taxon $TAXON"
  # Stream the accessions; UniProt's `list` format is one per line. Written
  # to a temp file first so an interrupted download cannot leave a short
  # list that later stages would silently treat as the whole proteome.
  ENCODED="$("$PYTHON" -c 'import urllib.parse,sys; print(urllib.parse.quote(sys.argv[1]))' "$QUERY")"
  curl -fsS --retry 4 --retry-delay 5 --max-time 1800 \
    "https://rest.uniprot.org/uniprotkb/stream?query=$ENCODED&format=list" \
    -o "$LIST.tmp"
  grep -c . "$LIST.tmp" >/dev/null || { echo "error: empty list" >&2; exit 1; }
  mv "$LIST.tmp" "$LIST"
  echo "  $(grep -c . "$LIST") accessions -> $LIST"
fi
N_IDS="$(grep -c . "$LIST")"

# --- 2. which scores apply ------------------------------------------------
if [ ${#EXTRA_SCORES[@]} -gt 0 ]; then
  SCORES=("${EXTRA_SCORES[@]}")
elif [ "$TAXON" = "$HUMAN_TAXON" ]; then
  SCORES=(alphamissense esm1v saprot thermompnn blosum62
          delta_volume delta_charge delta_hydrophobicity solvent_exposure)
else
  # No ClinVar for this organism, so alphamissense has no table and foldx
  # has no positions to score at.
  SCORES=(esm1v saprot thermompnn blosum62
          delta_volume delta_charge delta_hydrophobicity solvent_exposure)
fi

# --- 3. shared resources --------------------------------------------------
if [ "$TAXON" = "$HUMAN_TAXON" ]; then
  FETCH=(uniprot alphamissense clinvar thermompnn)
else
  FETCH=(uniprot thermompnn)
fi
echo
echo "=== fetching: ${FETCH[*]} ==="
"$PYTHON" -m vepbench fetch "${FETCH[@]}" --uniprot-list "$LIST" || {
  echo "warning: fetch reported a problem; see above" >&2; }

# --- 4. submit ------------------------------------------------------------
echo
echo "=== $NAME: $N_IDS proteins, ${#SCORES[@]} scores ==="
SUBMIT_ARGS=(--uniprot-list "$LIST" --scores "${SCORES[@]}"
             --mem "$MEM" --max-parallel "$MAXPAR" --name "vepbench-$NAME")
if [ "$SUBMIT" = "0" ]; then SUBMIT_ARGS+=(--dry-run); fi
bash "$ROOT/scripts/submit_job.sh" "${SUBMIT_ARGS[@]}"

echo
if [ "$SUBMIT" = "0" ]; then
  echo "nothing submitted. Re-run with --submit once the above looks right."
else
  echo "submitted. When it finishes:"
fi
cat <<CHECK
  export VEPBENCH_OUTPUTS=$VEPBENCH_OUTPUTS
  $PYTHON -m vepbench check  --uniprot-list $LIST
CHECK
if [ "$TAXON" = "$HUMAN_TAXON" ]; then
  echo "  $PYTHON -m vepbench report --uniprot-list $LIST --out \$VEPBENCH_OUTPUTS/report"
else
  echo "  # no ClinVar for this organism, so there is no ROC report to run"
fi
