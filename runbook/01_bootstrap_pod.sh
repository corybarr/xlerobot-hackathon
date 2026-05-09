#!/bin/bash
# Run inside the RunPod pod after SSH.
# Idempotent — safe to re-run if a step fails partway.

set -e

cd /workspace

if [ ! -d lerobot-MakerMods ]; then
  echo "[1/3] Cloning lerobot-MakerMods fork..."
  git clone https://github.com/Maker-Mods/lerobot-MakerMods.git
else
  echo "[1/3] lerobot-MakerMods already cloned — skipping."
fi

cd lerobot-MakerMods

echo "[2/3] Installing editable lerobot fork with feetech + smolvla extras..."
pip install -e ".[feetech,smolvla]"

echo "[3/3] Verifying install..."
python -c "import lerobot; print('lerobot location:', lerobot.__file__)"
python -c "import torch; print('torch:', torch.__version__, '| cuda:', torch.cuda.is_available(), '| device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"

echo ""
echo "==============================================="
echo "Bootstrap done. Manual steps next:"
echo ""
echo "  hf auth login                        # paste WRITE token"
echo "  huggingface-cli download lerobot/smolvla_base"
echo "  huggingface-cli download <YOUR_DATASET> --repo-type dataset"
echo ""
echo "Then kick off training with:"
echo "  bash 02_train_smolvla.sh <YOUR_DATASET> <YOUR_POLICY_NAME>"
echo "==============================================="
