#!/usr/bin/env bash
# One-shot setup: package, environments, weights, smoke test.
#
# Everything it does can be done by hand; see RUNBOOK.md. This exists so a
# fresh machine can be brought up in one command, and so the order is not
# something you have to remember.
#
#   bash scripts/bootstrap.sh                 # the verified core
#   bash scripts/bootstrap.sh --tier full     # every environment and weight
#   bash scripts/bootstrap.sh --backend conda
#   bash scripts/bootstrap.sh --dry-run
#
# Tiers:
#   core  esm + huggingface + proteinmpnn, and their weights. GPU-friendly,
#         covers the language models and the inverse-folding pair.
#   full  every environment, including the ones that drive upstream research
#         repositories and need the most hand-holding.
set -euo pipefail

TIER="core"
BACKEND=""
DRY_RUN=0
ASSUME_YES=0

while [ $# -gt 0 ]; do
  case "$1" in
    --tier) TIER="$2"; shift 2 ;;
    --backend) BACKEND="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    --yes|-y) ASSUME_YES=1; shift ;;
    -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

case "$TIER" in core|full) ;; *) echo "--tier must be core or full" >&2; exit 2 ;; esac

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
[ "$ASSUME_YES" = "1" ] && export VEPBENCH_ASSUME_YES=1
BACKEND_FLAG=()
[ -n "$BACKEND" ] && BACKEND_FLAG=(--backend "$BACKEND")

step() { echo; echo "=============== $* ==============="; }
run()  { echo "+ $*"; [ "$DRY_RUN" = "1" ] || "$@"; }

step "0. what we are working with"
python3 --version
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
else
  echo "no GPU detected: everything still runs, just slower"
fi
echo "free disk: $(df -Ph "$ROOT" | awk 'NR==2 {print $4}')"
if [ -z "${HF_TOKEN:-}" ]; then
  echo "HF_TOKEN is not set: ESM3 and ESM C will be skipped (they are gated)"
fi

step "1. install the orchestrator"
# Editable needs PEP 660, which means pip 21.3+ and setuptools 64+. On a
# cluster with no network, pip cannot fetch the setuptools that
# pyproject.toml asks for and falls back to whatever is installed, which may
# be too old. Try editable, then a plain install, and only then give up and
# fall back to running from the source tree, which needs no install at all.
if [ "$DRY_RUN" = "1" ]; then
  echo "+ python3 -m pip install -e ."
elif python3 -m pip install -e . ; then
  echo "installed in editable mode"
elif python3 -m pip install --upgrade pip setuptools wheel && python3 -m pip install -e . ; then
  echo "installed in editable mode after upgrading the build tools"
elif python3 -m pip install . ; then
  echo "installed normally; re-run this after changing the source"
else
  cat <<'FALLBACK'

Could not install the package. It is pure Python, so this is survivable:
run it from the source tree instead, which needs no installation.

    cd "$(pwd)"
    python3 -m vepbench list

Every command below that reads `vepbench ...` becomes `python3 -m vepbench ...`
and must be run from this directory. To fix the install properly:

    python3 -m pip install --upgrade pip setuptools wheel

FALLBACK
  echo "continuing with the source tree on PYTHONPATH"
  export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
fi

step "2. build the isolated environments ($TIER)"
if [ "$TIER" = "core" ]; then
  ENVS=(esm huggingface proteinmpnn)
else
  ENVS=(esm esm_if1 huggingface proteinmpnn evcouplings gemme vespa metl
        rasp eve tranception s3f venusrem promep stability_dg ddmut foldx)
fi
ENV_FAILED=()
for env in "${ENVS[@]}"; do
  if [ "$DRY_RUN" = "1" ]; then
    echo "+ python3 -m vepbench install-env $env ${BACKEND_FLAG[*]}"
  elif ! python3 -m vepbench install-env "$env" "${BACKEND_FLAG[@]}"; then
    ENV_FAILED+=("$env")
  fi
done
if [ "${#ENV_FAILED[@]}" -gt 0 ]; then
  echo
  echo "These environments did not install: ${ENV_FAILED[*]}"
  echo "Their predictors will report as unavailable rather than give wrong"
  echo "answers, so the rest of this is still worth doing. Fix them later with:"
  for env in "${ENV_FAILED[@]}"; do
    echo "  python3 -m vepbench install-env $env"
  done
fi

step "3. supporting binaries"
if [ -n "$BACKEND" ] && [ "$BACKEND" != "uv" ]; then
  echo "conda backend: foldseek, reduce, openmm, pdbfixer, openjdk and hmmer"
  echo "were installed with their environments"
else
  run bash scripts/install_foldseek.sh
fi

step "4. model weights"
run bash scripts/download_weights_esm.sh esm2
run bash scripts/download_weights_esm.sh esm1v
if [ "$TIER" = "full" ]; then
  run bash scripts/download_weights_esm.sh esm2-3b
  run bash scripts/download_weights_esm.sh esm-if1
  run bash scripts/download_weights_huggingface.sh all
  run bash scripts/download_weights_tranception.sh
else
  run bash scripts/download_weights_huggingface.sh saprot
  run bash scripts/download_weights_huggingface.sh prosst
fi

step "5. smoke test, no weights needed"
run bash examples/run_demo.sh

step "6. a real predictor"
run python3 -m vepbench run --uniprot P26678 --predictors esm2_650m --msa-backend none

step "done"
python3 -m vepbench status
cat <<'NEXT'

Next:
  vepbench list                      what exists and what it needs
  vepbench check-conventions         five sign conventions need confirming
  vepbench run --uniprot <ID> --predictors all
  less RUNBOOK.md                    the full procedure

NEXT
