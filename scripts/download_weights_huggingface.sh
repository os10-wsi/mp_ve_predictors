#!/usr/bin/env bash
# Pre-stage the Hugging Face model weights used by the huggingface and esm
# environments, instead of letting them download on first use.
#
# Worth doing when the machine that runs the predictions has no outbound
# network, or when you would rather pay the download cost once, deliberately.
#
#   bash scripts/download_weights_huggingface.sh saprot
#   bash scripts/download_weights_huggingface.sh all
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

TARGET="${1:-}"

# name:repo:size:environment
CATALOGUE=(
  "saprot:westlake-repl/SaProt_650M_AF2:2.6 GB:huggingface"
  "saprot-1.3b:westlake-repl/SaProt_1.3B_AF2:5 GB:huggingface"
  "prosst:AI4Protein/ProSST-2048:0.5 GB:huggingface"
  "prott5:Rostlab/prot_t5_xl_uniref50:11 GB:huggingface"
  "ankh:ElnaggarLab/ankh-large:7 GB:huggingface"
  "esm3:EvolutionaryScale/esm3-sm-open-v1:7 GB:esm"
)

usage() {
  echo "usage: $0 {all|$(printf '%s|' "${CATALOGUE[@]%%:*}" | sed 's/|$//')}" >&2
  echo >&2
  echo "available:" >&2
  for entry in "${CATALOGUE[@]}"; do
    IFS=':' read -r name repo size env <<< "$entry"
    printf '  %-12s %-38s %-8s (env: %s)\n' "$name" "$repo" "$size" "$env" >&2
  done
  exit 2
}
[ -n "$TARGET" ] || usage

# Hugging Face serves large files through its Xet CDN by default; some
# networks allow huggingface.co but block that CDN.
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"

fetch() {
  local name="$1" repo="$2" size="$3" env="$4"
  local py="$REPO_ROOT/envs/$env/.venv/bin/python"
  if [ ! -x "$py" ]; then
    echo "  skipping $name: environment '$env' is not installed" >&2
    echo "  (python -m vepbench install-env $env)" >&2
    return 0
  fi
  if [ "$repo" = "EvolutionaryScale/esm3-sm-open-v1" ] && [ -z "${HF_TOKEN:-}" ]; then
    echo "  skipping $name: gated, needs HF_TOKEN and the licence accepted" >&2
    return 0
  fi
  echo "--- $name ($repo, $size)"
  "$py" -I -c "
from huggingface_hub import snapshot_download
path = snapshot_download(
    '$repo',
    allow_patterns=['*.json', '*.bin', '*.safetensors', '*.txt', '*.model', '*.py'],
)
print('  cached at', path)
"
}

SELECTED=()
TOTAL=""
for entry in "${CATALOGUE[@]}"; do
  IFS=':' read -r name repo size env <<< "$entry"
  if [ "$TARGET" = "all" ] || [ "$TARGET" = "$name" ]; then
    SELECTED+=("$entry")
    TOTAL="$TOTAL $size,"
  fi
done
[ "${#SELECTED[@]}" -gt 0 ] || usage

confirm_download "${TOTAL% ,}" "Hugging Face weights: ${TARGET}" || exit 1
report_free_space
echo "Destination: \${HF_HOME:-~/.cache/huggingface}"

for entry in "${SELECTED[@]}"; do
  IFS=':' read -r name repo size env <<< "$entry"
  fetch "$name" "$repo" "$size" "$env"
done

echo
echo "Set HF_HOME before running if you want these somewhere other than the"
echo "default cache, and set the same value when you run the predictions."
