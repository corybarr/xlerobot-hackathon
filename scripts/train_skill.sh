#!/usr/bin/env bash
# Train a per-skill policy. VLA-agnostic — picks the right backbone by --vla.
#
# Usage: ./scripts/train_skill.sh <skill_name> <vla_type> [extra lerobot-train args...]
#   vla_type: smolvla | act | molmoact2
#
# Outputs to outputs/train/<skill>-<vla>/ and pushes checkpoint to
# $HF_USER/xlerobot-<skill>-<vla> on completion.
#
# Run on Spark for real training; tablet CPU is a no-go for fine-tuning.

set -euo pipefail

SKILL="${1:?usage: train_skill.sh <skill_name> <vla_type>}"
VLA="${2:?usage: train_skill.sh <skill_name> <vla_type>}"
shift 2

HF_USER="${HF_USER:-Globalmysterysnailrevolution}"
DATASET_REPO="${HF_USER}/xlerobot-${SKILL}"
OUTPUT_DIR="outputs/train/${SKILL}-${VLA}"
PUSH_REPO="${HF_USER}/xlerobot-${SKILL}-${VLA}"

case "${VLA}" in
  smolvla)
    POLICY_PATH="lerobot/smolvla_base"
    ;;
  act)
    POLICY_PATH="lerobot/act_base"
    ;;
  molmoact2)
    echo "MolmoAct2 is NOT trained via lerobot-train — uses its own runner."
    echo "Use the allenai/molmoact2 repo's fine-tune script with"
    echo "  base checkpoint: allenai/MolmoAct2-SO100_101"
    echo "  dataset:         ${DATASET_REPO}"
    echo "TODO: wrap that in scripts/train_molmoact2.sh once we have it stood up."
    exit 0
    ;;
  *)
    echo "Unknown VLA: ${VLA}. Choose: smolvla | act | molmoact2"
    exit 1
    ;;
esac

echo "Training ${VLA} on dataset ${DATASET_REPO}"
echo "Output dir: ${OUTPUT_DIR}"
echo "Push to:    ${PUSH_REPO}"
echo

lerobot-train \
  --policy.path="${POLICY_PATH}" \
  --dataset.repo_id="${DATASET_REPO}" \
  --output_dir="${OUTPUT_DIR}" \
  --num_workers=2 \
  --batch_size=8 \
  --steps=20000 \
  --eval_freq=2000 \
  --save_freq=5000 \
  --push_to_hub=true \
  --hub_repo_id="${PUSH_REPO}" \
  "$@"
