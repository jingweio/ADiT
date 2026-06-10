#!/bin/bash
# ============================================================================
# ADiT reproduction — Step 0: environment setup
# Target stack (per README): PyTorch 2.1.2 + CUDA 12.1 + Python 3.10
# Usage:  bash reproduce/setup_env.sh
# ============================================================================
set -euo pipefail

ENV_NAME="${ENV_NAME:-adit}"
source "$(conda info --base)/etc/profile.d/conda.sh"

echo ">>> [1/5] Create conda env '${ENV_NAME}' (python 3.10)"
conda create -y -n "${ENV_NAME}" python=3.10

echo ">>> [2/5] Install PyTorch 2.1.2 + cu121"
conda run -n "${ENV_NAME}" --no-capture-output pip install \
  torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 \
  --index-url https://download.pytorch.org/whl/cu121

echo ">>> [3/5] Install PyG companion wheels (pt21 + cu121 + cp310)"
PYG="https://data.pyg.org/whl/torch-2.1.0+cu121"
conda run -n "${ENV_NAME}" --no-capture-output pip install \
  "${PYG}/pyg_lib-0.4.0+pt21cu121-cp310-cp310-linux_x86_64.whl" \
  "${PYG}/torch_scatter-2.1.2+pt21cu121-cp310-cp310-linux_x86_64.whl" \
  "${PYG}/torch_spline_conv-1.2.2+pt21cu121-cp310-cp310-linux_x86_64.whl" \
  "${PYG}/torch_cluster-1.6.2+pt21cu121-cp310-cp310-linux_x86_64.whl" \
  "${PYG}/torch_sparse-0.6.18+pt21cu121-cp310-cp310-linux_x86_64.whl"

echo ">>> [4/5] Install project dependencies (per README)"
conda run -n "${ENV_NAME}" --no-capture-output pip install \
  rootutils tqdm biopython foldcomp lightning omegaconf hydra-core pandas dm-tree
conda run -n "${ENV_NAME}" --no-capture-output pip install \
  rich biotite atom3D torchdrug torcheval spyrmsd lifelines
# NOTE: README omits these two as well:
#  - fair-esm : adit/models/net/esm2/esm.py does `import esm` and calls
#               esm.pretrained.load_model_and_alphabet_core.
#  - loralib  : adit/utils/checkpoint_utils.py does `import loralib as lora`.
conda run -n "${ENV_NAME}" --no-capture-output pip install fair-esm loralib

echo ">>> [5/5] Install huggingface_hub (for downloading weights/datasets)"
conda run -n "${ENV_NAME}" --no-capture-output pip install huggingface_hub

# IMPORTANT version pins (README does not specify these and the defaults break):
#  - numpy<2     : torch 2.1.2 is built against NumPy 1.x; NumPy 2.x crashes the
#                  torch<->numpy bridge ("_ARRAY_API not found").
#  - setuptools<81: setuptools>=81 removed pkg_resources, which torchdrug imports.
echo ">>> [pin] numpy<2, setuptools<81"
conda run -n "${ENV_NAME}" --no-capture-output pip install "numpy<2" "setuptools<81"

echo ">>> Verifying torch + CUDA"
conda run -n "${ENV_NAME}" --no-capture-output python -c \
  "import torch; print('torch', torch.__version__, '| cuda available', torch.cuda.is_available(), '| cuda', torch.version.cuda)"

echo "ENV_SETUP_DONE"
