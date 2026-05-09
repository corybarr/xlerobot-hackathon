# LeRobot — Physical AI Hack SF, May 9-10 2026

## Environment
- Python 3.10 venv at .venv/
- lerobot 0.4.4, torch 2.10 (MPS backend on Apple Silicon)
- Hugging Face auth: `hf auth login`

## Activation
cd <repo> && source .venv/bin/activate
# activate script sets DYLD_FALLBACK_LIBRARY_PATH so pyzbar finds zbar (macOS, brew zbar)

## Installed
- lerobot 0.4.4 (PyPI install — NOT editable)
- OpenCV 4.13, pyzbar + libzbar (QR scanning task)
- dynamixel-sdk + pyserial (SO-100/SO-101 servos)
- gym-pusht (sim, installed for pretrained policy eval)
- Reference clones (gitignored): lerobot-src/, SO-ARM100/

## Hackathon Tasks
1. Toasting bread — continuous control + timing
2. QR code scan — perception + alignment (pyzbar + cv2)
3. Setting a table — precise insertion, depth estimation

## Hardware (provided on-site)
- XLeRobot dual-arm kits (SO-100/SO-101 derivative — same dynamixel family as SO-ARM100, but URDF/calibration NOT assumed 1:1)
- ModBlocks USB-C components
- Stack: LeRobot + MakerMods App + OpenClaw + Kite ML

## First-session TODOs
- Test cv2.VideoCapture(0) — macOS will prompt for camera permission, approve for terminal app
- Clone the XLeRobot repo once it's distributed (don't assume SO-ARM100 configs transfer)
- wandb login if a sponsor demo requires it
- MakerMods / OpenClaw / Kite tooling onboarded on-site

## Known issue — pusht sim eval (deferred)
Pre-trained `lerobot/diffusion_pusht` checkpoint is incompatible with installed lerobot 0.4.4 (missing policy_preprocessor.json — pre-PolicyProcessorPipeline upload). Do NOT retry without either downgrading lerobot or finding a re-uploaded checkpoint. Sim eval is non-load-bearing for the hackathon — defer indefinitely.

## RunPod training (SmolVLA on Linux GPU)

Use this when local Mac MPS is too slow or for the real hackathon training run.

### Pod config
- GPU: A40 48GB (preferred, ~$0.40/hr) or A100 40GB (~$1.50/hr)
- Template: RunPod PyTorch 2.4+ (CUDA 12.x)
- Disk: 50GB volume minimum
- Region: US-CA or US-OR

### Bootstrap inside pod (SSH in first)
cd /workspace
git clone https://github.com/Maker-Mods/lerobot-MakerMods.git
cd lerobot-MakerMods
pip install -e ".[feetech,smolvla]"
hf auth login
huggingface-cli download lerobot/svla_so101_pickplace --repo-type dataset
huggingface-cli download lerobot/smolvla_base

## Local training on Mac (FFmpeg fix)
Required env var for any local lerobot-train command:
  export DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/opt/ffmpeg@7/lib
Reason: torchcodec 0.5 needs libavutil.56–59; brew default ffmpeg ships .60. ffmpeg@7 provides .59.
