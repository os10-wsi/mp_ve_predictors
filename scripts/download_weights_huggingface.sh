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

# name:repo:size in GB:environment
CATALOGUE=(
  "saprot:westlake-repl/SaProt_650M_AF2:2.6:huggingface"
  "saprot-1.3b:westlake-repl/SaProt_1.3B_AF2:5:huggingface"
  "prosst:AI4Protein/ProSST-2048:0.5:huggingface"
  "prott5:Rostlab/prot_t5_xl_uniref50:11:huggingface"
  "ankh:ElnaggarLab/ankh-large:7:huggingface"
  "esm3:EvolutionaryScale/esm3-sm-open-v1:7:esm"
)

usage() {
  echo "usage: $0 {all|$(printf '%s|' "${CATALOGUE[@]%%:*}" | sed 's/|$//')}" >&2
  echo >&2
  echo "available:" >&2
  for entry in "${CATALOGUE[@]}"; do
    IFS=':' read -r name repo size env <<< "$entry"
    printf '  %-12s %-38s %5s GB (env: %s)\n' "$name" "$repo" "$size" "$env" >&2
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
  echo "--- $name ($repo, $size GB)"
  "$py" -I -c "
from huggingface_hub import snapshot_download
path = snapshot_download(
    '$repo',
    allow_patterns=['*.json', '*.bin', '*.safetensors', '*.txt', '*.model', '*.py'],
)
print('  cached at', path)
"
}

# Decide what is actually fetchable before asking anything. Prompting for a
# 33 GB download and then skipping every item wastes the answer.
MATCHED=0
SELECTED=()
SKIPPED=()
TOTAL=0
for entry in "${CATALOGUE[@]}"; do
  IFS=':' read -r name repo size env <<< "$entry"
  [ "$TARGET" = "all" ] || [ "$TARGET" = "$name" ] || continue
  MATCHED=$((MATCHED + 1))
  if ! require_env "$env" >/dev/null 2>&1; then
    SKIPPED+=("$name: environment '$env' is not installed or is incomplete -> python3 -m vepbench install-env $env")
    continue
  fi
  if [ "$repo" = "EvolutionaryScale/esm3-sm-open-v1" ] && [ -z "${HF_TOKEN:-}" ]; then
    SKIPPED+=("$name: gated -> accept the licence and export HF_TOKEN")
    continue
  fi
  SELECTED+=("$entry")
  TOTAL="$(awk -v a="$TOTAL" -v b="$size" 'BEGIN {printf "%.1f", a + b}')"
done
[ "$MATCHED" -gt 0 ] || usage

DESTINATION="${HF_HOME:-$HOME/.cache/huggingface}"

if [ "${#SKIPPED[@]}" -gt 0 ]; then
  echo "Cannot fetch these yet:"
  for line in "${SKIPPED[@]}"; do echo "  $line"; done
  echo
fi

if [ "${#SELECTED[@]}" -eq 0 ]; then
  echo "Nothing left to download. Resolve the above, then run this again." >&2
  exit 1
fi

echo "Ready to fetch ${#SELECTED[@]} of $MATCHED:"
for entry in "${SELECTED[@]}"; do
  IFS=':' read -r name repo size env <<< "$entry"
  printf '  %-12s %-38s %5s GB\n' "$name" "$repo" "$size"
done

echo "Free space for $DESTINATION: $(free_space_at "$DESTINATION")"
confirm_download "${TOTAL} GB" "Hugging Face weights: ${TARGET}" "$DESTINATION" || exit 1

for entry in "${SELECTED[@]}"; do
  IFS=':' read -r name repo size env <<< "$entry"
  fetch "$name" "$repo" "$size" "$env"
done

echo
echo "Weights are under $DESTINATION."
echo "If you set HF_HOME to put them elsewhere, set the same value when you run."
