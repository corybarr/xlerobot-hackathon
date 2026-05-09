"""Gemma-based scene watcher + skill router for Set-the-Table.

Architecture:
  high-level planner   = Gemma (vision-LM, runs on Spark via Ollama tunnel)
  low-level skills     = per-skill VLAs (SmolVLA / ACT / MolmoAct2),
                         each fine-tuned on demos for ONE discrete action

Loop:
  1. Capture front-camera frame
  2. Send (frame, goal, skill list, history) to Gemma
  3. Gemma returns either {"done": true, "reason": ...}
                       or {"done": false, "skill": "<name>", "reason": ...}
  4. Invoke the chosen skill's VLA (subprocess; lerobot-record with policy.path)
  5. Append to history, repeat

Env:
  OLLAMA_HOST   default http://localhost:11434
  GEMMA_MODEL   default gemma3:27b
  CAMERA_INDEX  default 1 (front_cam per the lerobot-MakerMods convention)
  GOAL          default "set the table"
  HF_USER       default Globalmysterysnailrevolution
  VLA_BACKEND   default smolvla
  MAX_STEPS     default 20
"""
from __future__ import annotations

import base64
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import cv2
import requests
import yaml

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
GEMMA_MODEL = os.getenv("GEMMA_MODEL", "gemma3:27b")
CAMERA_INDEX = int(os.getenv("CAMERA_INDEX", "1"))
GOAL = os.getenv("GOAL", "set the table")
HF_USER = os.getenv("HF_USER", "Globalmysterysnailrevolution")
VLA_BACKEND = os.getenv("VLA_BACKEND", "smolvla")
MAX_STEPS = int(os.getenv("MAX_STEPS", "20"))

REPO_ROOT = Path(__file__).resolve().parent.parent
SKILLS_PATH = REPO_ROOT / "skills" / "skills.yaml"


def load_skills() -> dict:
    with open(SKILLS_PATH) as f:
        return yaml.safe_load(f)


def capture_frame(camera_index: int) -> bytes:
    """Capture one JPEG frame from the front camera."""
    cap = cv2.VideoCapture(camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError(f"Could not read from camera {camera_index}")
    _, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return jpeg.tobytes()


def query_gemma(image_bytes: bytes, goal: str, skills: dict, history: list[str]) -> dict:
    """Ask Gemma which skill to invoke next. Returns parsed JSON decision."""
    skills_brief = "\n".join(
        f"- {name}: {meta['description']}" for name, meta in skills.items()
    )
    history_brief = (
        "\n".join(f"  step {i+1}: {h}" for i, h in enumerate(history))
        or "  (none yet)"
    )

    prompt = f"""You are a robot task planner with access to a set of trained low-level skills (VLAs).
Your job: look at the current scene, recall what's been done, and decide the next single action.

Goal: "{goal}"

Available skills (each is a separately trained VLA you can invoke):
{skills_brief}

Steps already executed (most recent last):
{history_brief}

Respond with JSON only, no prose. Schema:
  {{"done": true,  "reason": "<one sentence why complete>"}}
or
  {{"done": false, "skill": "<exact skill name from list>", "reason": "<one sentence why this skill next>"}}
"""

    image_b64 = base64.b64encode(image_bytes).decode()
    payload = {
        "model": GEMMA_MODEL,
        "prompt": prompt,
        "images": [image_b64],
        "stream": False,
        "format": "json",
    }
    r = requests.post(f"{OLLAMA_HOST}/api/generate", json=payload, timeout=120)
    r.raise_for_status()
    raw = r.json().get("response", "{}")
    return json.loads(raw)


def invoke_skill(skill_name: str) -> bool:
    """Run lerobot inference with the per-skill VLA checkpoint.

    Expects the policy to live at HF Hub repo:
      {HF_USER}/xlerobot-{skill_name}-{VLA_BACKEND}

    Returns True if the subprocess exited 0.
    """
    policy_repo = f"{HF_USER}/xlerobot-{skill_name}-{VLA_BACKEND}"
    follower_port = os.getenv("FOLLOWER_PORT", "COM10")
    leader_port = os.getenv("LEADER_PORT", "COM7")

    cmd = [
        "lerobot-record",
        "--robot.type=so101_follower",
        f"--robot.port={follower_port}",
        "--robot.id=arm_a_follower",
        "--teleop.type=so101_leader",
        f"--teleop.port={leader_port}",
        "--teleop.id=arm_a_leader",
        f"--policy.path={policy_repo}",
        "--dataset.num_episodes=1",
        "--dataset.episode_time_s=15",
        "--dataset.repo_id=local-eval",
        "--dataset.push_to_hub=false",
    ]
    print(f"  invoking: lerobot-record --policy.path={policy_repo}")
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"  skill '{skill_name}' failed (exit {res.returncode})")
        print(f"  stderr tail: {res.stderr.splitlines()[-3:] if res.stderr else '(empty)'}")
        return False
    return True


def main() -> int:
    skills = load_skills()
    history: list[str] = []

    print(f"orchestrator: goal='{GOAL}'  vla={VLA_BACKEND}  model={GEMMA_MODEL}")
    print(f"  available skills: {list(skills.keys())}\n")

    for step in range(MAX_STEPS):
        print(f"--- step {step+1}/{MAX_STEPS} ---")
        try:
            frame = capture_frame(CAMERA_INDEX)
            decision = query_gemma(frame, GOAL, skills, history)
        except Exception as e:
            print(f"  planner error: {type(e).__name__}: {e}")
            return 1

        print(f"  gemma: {decision}")

        if decision.get("done"):
            print(f"\nGoal complete: {decision.get('reason')}")
            return 0

        skill = decision.get("skill")
        if skill not in skills:
            print(f"  ERROR: gemma chose unknown skill '{skill}'. Aborting.")
            return 1

        ok = invoke_skill(skill)
        history.append(f"{skill} ({'ok' if ok else 'FAILED'}): {decision.get('reason')}")

        if not ok:
            print("  continuing — gemma will replan from updated scene")

        time.sleep(1)

    print("\nReached max steps without goal completion.")
    return 2


if __name__ == "__main__":
    sys.exit(main())
