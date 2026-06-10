#!/bin/bash
# ============================================================================
# ADiT reproduction — Step 1: download checkpoints + datasets
# Scope: ONLY what protein-ligand (LBA) + protein-protein (skempi) need.
#   - ckpts/adit_S.ckpt                 (HuggingFace VectorShi/ADiT_pretrained_ckpts)
#   - ckpts/esm2_t33_650M_UR50D.pt      (Meta fair-esm public weights, ~2.5GB)
#   - dataset/LBA/   (from LBA.zip)     (HuggingFace VectorShi/ADiT_dataset)
#   - dataset/skempi/ (from skempi.zip)
# Usage:  bash reproduce/download_assets.sh
# ============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."   # repo root
mkdir -p ckpts dataset

echo ">>> [1/4] Download ADiT-S pretrained checkpoint (HuggingFace)"
python3 - <<'PY'
from huggingface_hub import hf_hub_download
p = hf_hub_download("VectorShi/ADiT_pretrained_ckpts", "adit_S.ckpt",
                    repo_type="model", local_dir="ckpts")
print("  ->", p)
PY

echo ">>> [2/4] Download ESM-2-650M (Meta fair-esm public weights)"
if [ ! -s ckpts/esm2_t33_650M_UR50D.pt ]; then
  wget -c -O ckpts/esm2_t33_650M_UR50D.pt \
    https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t33_650M_UR50D.pt
else
  echo "  already present, skip"
fi

echo ">>> [3/4] Download datasets LBA.zip + skempi.zip (HuggingFace)"
python3 - <<'PY'
from huggingface_hub import hf_hub_download
for fn in ["LBA.zip", "skempi.zip"]:
    p = hf_hub_download("VectorShi/ADiT_dataset", fn,
                        repo_type="dataset", local_dir="dataset")
    print("  ->", p)
PY

echo ">>> [4/4] Unzip datasets into dataset/"
python3 - <<'PY'
import zipfile, os
for fn in ["LBA.zip", "skempi.zip"]:
    path = os.path.join("dataset", fn)
    print("  unzip", path)
    with zipfile.ZipFile(path) as z:
        z.extractall("dataset")
PY

echo "DOWNLOAD_DONE"
echo "--- ckpts ---"; ls -la ckpts
echo "--- dataset ---"; ls -la dataset
