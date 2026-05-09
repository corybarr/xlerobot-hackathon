#!/bin/bash
# Train SmolVLA on a HuggingFace dataset.
# Usage: bash 02_train_smolvla.sh <DATASET_REPO_ID> <POLICY_REPO_ID> [STEPS] [BATCH]
# Example: bash 02_train_smolvla.sh rafamara/grab_cup rafamara/grab_cup_smolvla 20000 32

set -e

DATASET="${1:?missing DATASET_REPO_ID — e.g. rafamara/grab_cup}"
POLICY_NAME="${2:?missing POLICY_REPO_ID — e.g. rafamara/grab_cup_smolvla}"
STEPS="${3:-20000}"
BATCH="${4:-32}"
JOB_NAME="${POLICY_NAME##*/}"

cd /workspace/lerobot-MakerMods

echo "==============================================="
echo "Training SmolVLA"
echo "  dataset:  $DATASET"
echo "  policy:   $POLICY_NAME"
echo "  steps:    $STEPS"
echo "  batch:    $BATCH"
echo "  output:   outputs/train/$JOB_NAME"
echo "==============================================="

lerobot-train \
  --policy.type=smolvla \
  --policy.path=lerobot/smolvla_base \
  --policy.repo_id="$POLICY_NAME" \
  --policy.push_to_hub=true \
  --policy.device=cuda \
  --dataset.repo_id="$DATASET" \
  --batch_size="$BATCH" \
  --steps="$STEPS" \
  --output_dir="outputs/train/$JOB_NAME" \
  --job_name="$JOB_NAME" \
  --save_freq=2000 \
  --log_freq=100 \
  --wandb.enable=true
