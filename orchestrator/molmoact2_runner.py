"""MolmoAct2 runner — invokes a fine-tuned MolmoAct2 policy on the SO-101.

This is a STUB with the real interface but a placeholder control loop. Filling
it in requires:
  1. Cloning allenai/molmoact2 alongside this repo (or pip install when published)
  2. Loading the fine-tuned checkpoint (e.g. <HF_USER>/xlerobot-<skill>-molmoact2)
  3. Wiring the model's action outputs (continuous joint deltas via flow-matching
     expert) into FeetechMotorsBus.sync_write("Goal_Position", ...)

The interface mirrors `lerobot-record --policy.path=...` so the orchestrator
can call it the same way — see `build_vla_command` in orchestrator.py.

Usage:
    python -m orchestrator.molmoact2_runner --policy <hf_repo> --port COM10

Returns exit code 0 on completion, non-zero on error. The orchestrator will
still verify state via Gemma frame comparison regardless of exit code.
"""
from __future__ import annotations

import argparse
import sys
import time

# Lazy imports so this file can be inspected / tested without the heavy deps.


def _load_model(policy_repo: str):
    """Load a fine-tuned MolmoAct2 checkpoint from HF Hub.

    Default base model is `allenai/MolmoAct2-SO100_101` (single-arm SO-100/101
    fine-tune). For our skill-specific checkpoints, policy_repo points at
    `<HF_USER>/xlerobot-<skill>-molmoact2`, fine-tuned on top.
    """
    from transformers import AutoModelForImageTextToText, AutoProcessor
    print(f"  loading MolmoAct2 from {policy_repo}...")
    model = AutoModelForImageTextToText.from_pretrained(
        policy_repo, trust_remote_code=True, dtype="auto"
    )
    processor = AutoProcessor.from_pretrained(policy_repo, trust_remote_code=True)
    return model, processor


def _open_arm(port: str):
    """Open the SO-101 follower bus at the given COM port. Returns the bus."""
    from lerobot.motors import Motor, MotorNormMode
    from lerobot.motors.feetech import FeetechMotorsBus

    motors = {}
    names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
    for i, name in enumerate(names):
        norm = MotorNormMode.RANGE_0_100 if name == "gripper" else MotorNormMode.RANGE_M100_100
        motors[name] = Motor(i + 1, "sts3215", norm)
    bus = FeetechMotorsBus(port=port, motors=motors)
    bus.connect()
    return bus, names


def _capture_frames(camera_indices: list[int]) -> dict:
    """Capture one frame from each of the configured cameras."""
    import cv2
    out = {}
    for idx in camera_indices:
        cap = cv2.VideoCapture(idx)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            raise RuntimeError(f"camera {idx} capture failed")
        out[idx] = frame
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--policy", required=True, help="HF Hub repo id of fine-tuned checkpoint")
    p.add_argument("--port", required=True, help="SO-101 follower COM port (e.g. COM10)")
    p.add_argument("--max-seconds", type=float, default=15.0,
                   help="hard cap on inference loop length")
    p.add_argument("--front-cam", type=int, default=1)
    p.add_argument("--hand-cam", type=int, default=0)
    p.add_argument("--side-cam", type=int, default=2)
    args = p.parse_args()

    try:
        model, processor = _load_model(args.policy)
    except Exception as e:
        print(f"FATAL: load_model failed: {type(e).__name__}: {e}")
        return 2

    try:
        bus, motor_names = _open_arm(args.port)
    except Exception as e:
        print(f"FATAL: arm connect failed on {args.port}: {type(e).__name__}: {e}")
        return 3

    cameras = [args.hand_cam, args.front_cam, args.side_cam]
    print(f"  cameras: hand={args.hand_cam} front={args.front_cam} side={args.side_cam}")
    print(f"  running for max {args.max_seconds}s — Ctrl+C to stop")

    start = time.time()
    step = 0
    try:
        while time.time() - start < args.max_seconds:
            step += 1
            try:
                frames = _capture_frames(cameras)
                state = bus.sync_read("Present_Position", motor_names, normalize=False)
            except Exception as e:
                print(f"  step {step}: observation failed: {e}")
                break

            # ----- TODO: real MolmoAct2 inference -----
            # Build the model input (multimodal: images + state + language instruction).
            # Run model.generate(...) or the flow-matching action expert.
            # Decode action chunk -> joint position deltas.
            # Then: bus.sync_write("Goal_Position", deltas + state, normalize=False)
            #
            # Right now this is a stub: hold position so the arm doesn't drift.
            try:
                bus.sync_write("Goal_Position", state, normalize=False)
            except Exception as e:
                print(f"  step {step}: write failed: {e}")
                break

            time.sleep(0.1)  # 10 Hz placeholder rate
    except KeyboardInterrupt:
        print("\n  interrupted")
    finally:
        try:
            bus.disconnect()
        except Exception:
            pass

    elapsed = time.time() - start
    print(f"  ran {step} steps over {elapsed:.1f}s")
    print("  NOTE: real MolmoAct2 inference is stubbed. Wire allenai/molmoact2 in for production.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
