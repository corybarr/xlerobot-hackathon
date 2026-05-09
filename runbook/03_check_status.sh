#!/bin/bash
# Quick status checker — run inside the pod, in a second SSH session.

JOB_NAME="${1:-grab_cup_smolvla}"
LOG_DIR="/workspace/lerobot-MakerMods/outputs/train/$JOB_NAME"

echo "=== GPU ==="
nvidia-smi | head -20

echo ""
echo "=== Latest training log lines ==="
if ls "$LOG_DIR"/checkpoints/*/pretrained_model/ 2>/dev/null | head -3; then
  echo ""
  echo "Checkpoints saved so far:"
  ls "$LOG_DIR/checkpoints/" 2>/dev/null
fi

echo ""
echo "=== Latest stdout (look for step:N loss:X.X) ==="
ps aux | grep lerobot-train | grep -v grep | head -2
