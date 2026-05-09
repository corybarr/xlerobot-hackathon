# RunPod training runbook — SmolVLA / ACT on grab-cup data

End-to-end recipe to get from "dataset on HF Hub" to "trained policy on HF Hub" in one paste-able sequence.

## Prerequisites (do once, before "go")

1. **HuggingFace account + token with WRITE access**
   - Settings → Access Tokens → "New token" → role: `write`
   - Copy the `hf_...` string. You'll paste it inside the pod (`hf auth login`).
2. **RunPod account with credit balance**
   - $5 covers a 12-hour exploration session on A40
   - https://www.runpod.io/console
3. **Dataset published on HF Hub**
   - `lerobot-record` pushes automatically when episodes finish
   - Note the exact `username/dataset_name` — needed below

## Cost estimate

| Run | GPU | Steps | Wall-clock | Est. cost |
|---|---|---|---|---|
| **SmolVLA 20k** | A40 48 GB | 20000 | ~4 hr | ~$1.60 |
| **ACT 20k** (fallback) | A10G 24 GB | 20000 | ~1.5 hr | ~$0.50 |
| **ACT 5k** (smoke) | A10G 24 GB | 5000 | ~25 min | ~$0.15 |

## The full sequence

### Step 1 — Spawn pod (RunPod web UI)

- Template: **RunPod PyTorch 2.4** (CUDA 12.x)
- GPU: **A40 48 GB** (or A10G 24 GB if doing ACT only)
- Disk: **50 GB**
- Region: US-CA / US-OR
- Click **Deploy**, wait ~60s for boot, copy SSH command from the pod's "Connect" tab.

### Step 2 — SSH in + bootstrap

```bash
# On your Mac:
ssh root@<POD_HOST> -p <POD_PORT> -i ~/.ssh/id_ed25519

# Inside the pod:
cd /workspace
curl -O https://raw.githubusercontent.com/corybarr/xlerobot-hackathon/rafael/runbook/01_bootstrap_pod.sh
bash 01_bootstrap_pod.sh
# When it prompts, run:
hf auth login                  # paste your write-token
huggingface-cli download lerobot/smolvla_base
huggingface-cli download <YOUR_DATASET> --repo-type dataset
```

(Or just paste the contents of `01_bootstrap_pod.sh` directly.)

### Step 3 — Kick off training

**SmolVLA (4 hr):**
```bash
bash 02_train_smolvla.sh <YOUR_DATASET> <YOUR_POLICY_NAME>
# Example: bash 02_train_smolvla.sh rafamara/grab_cup rafamara/grab_cup_smolvla
```

**ACT (1.5 hr, cheaper fallback):**
```bash
bash 02_train_act.sh <YOUR_DATASET> <YOUR_POLICY_NAME>
```

### Step 4 — Monitor

```bash
# In a second SSH session, OR via wandb if you enabled it:
tail -f /workspace/lerobot-MakerMods/outputs/train/<JOB>/logs/*.log
nvidia-smi                     # GPU utilization
```

### Step 5 — When training finishes

Trained policy is auto-pushed to `<YOUR_POLICY_NAME>` on HF Hub (because `--push_to_hub=true`). Verify in browser. Then:
- Tear down the pod (stop billing) **only after** push completes.
- For inference: see `04_inference.md`.

## If something breaks

- **OOM during training**: drop `--batch_size=32` to `16` or `8`.
- **Dataset 404**: verify `huggingface-cli download <YOUR_DATASET> --repo-type dataset` succeeded inside the pod.
- **HF auth error on push**: token is read-only. Generate a write token, `hf auth logout && hf auth login`.
- **Loss is NaN**: data has bad frames. Inspect with `lerobot-dataset-viz --repo-id <YOUR_DATASET>` locally on Mac.

## Mac-side, while training runs

- Pre-warm the local cache for the trained policy:
  ```bash
  hf download <YOUR_POLICY_NAME>   # works once push finishes
  ```
- Plan the demo flow + camera angles for Sunday 4 PM.
