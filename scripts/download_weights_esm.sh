#!/usr/bin/env bash
# Fetch ESM checkpoints deliberately rather than on first use. Sizes are
# stated before anything downloads.
#
#   bash scripts/download_weights_esm.sh esm1v     # 5 x 2.5 GB
#   bash scripts/download_weights_esm.sh esm2      # 2.5 GB
#   bash scripts/download_weights_esm.sh esm2-3b   # 11 GB
#   bash scripts/download_weights_esm.sh esm-if1   # 0.5 GB
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

TARGET="${1:-}"
PY="$REPO_ROOT/envs/esm/.venv/bin/python"
require_env esm || exit 1

# Hugging Face serves large files through its Xet CDN by default; some
# networks allow huggingface.co but block that CDN.
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"

case "$TARGET" in
  esm1v)
    confirm_download "12.5 GB (five replicates of 2.5 GB)" "ESM-1v ensemble" || exit 1
    MODELS="facebook/esm1v_t33_650M_UR90S_1 facebook/esm1v_t33_650M_UR90S_2 facebook/esm1v_t33_650M_UR90S_3 facebook/esm1v_t33_650M_UR90S_4 facebook/esm1v_t33_650M_UR90S_5" ;;
  esm2)     confirm_download "2.5 GB" "ESM-2 650M" || exit 1
            MODELS="facebook/esm2_t33_650M_UR50D" ;;
  esm2-3b)  confirm_download "11 GB" "ESM-2 3B" || exit 1
            MODELS="facebook/esm2_t36_3B_UR50D" ;;
  esm-if1)
    confirm_download "0.5 GB" "ESM-IF1 (from dl.fbaipublicfiles.com)" || exit 1
    report_free_space
    export TORCH_HOME="$WEIGHTS/torch_hub"; mkdir -p "$TORCH_HOME"
    "$PY" -I -c "
import torch, esm
torch.hub.set_dir('$TORCH_HOME')
esm.pretrained.esm_if1_gvp4_t16_142M_UR50()
print('  cached')
"
    echo "Weights are in $TORCH_HOME"; exit 0 ;;
  *) echo "usage: $0 {esm1v|esm2|esm2-3b|esm-if1}" >&2; exit 2 ;;
esac

report_free_space
for model in $MODELS; do
  echo "--- $model"
  "$PY" -I -c "
from huggingface_hub import snapshot_download
path = snapshot_download('$model', allow_patterns=['*.json','*.bin','*.safetensors','*.txt','*.model'])
print('  cached at', path)
"
done
