"""Background Gemma watcher — runs concurrent with VLA inference.

Captures from a (separate) camera at WATCH_INTERVAL_S, asks Gemma to summarize
the scene + anticipate the next skill, and writes the analysis to
runtime/watcher_state.json. The orchestrator's SELECT phase reads this file so
it can pre-plan while the current VLA skill is still executing — pipelining
the planner against the executor.

CAMERA SHARING (important):
  Windows holds USB webcams exclusively per process. While lerobot-record runs
  with the VLA's 3 cameras, this watcher CANNOT open those same indices. Use
  one of:
    (a) a 4th physical camera (set CAMERA_INDEX to its index) — easiest
    (b) a file-based frame source — point WATCH_FRAME_PATH to a JPEG that
        lerobot dumps periodically (requires patching the fork)
    (c) skip while VLA is running, only watch between skills

Env:
  OLLAMA_HOST, GEMMA_MODEL, CAMERA_INDEX (same as orchestrator)
  WATCH_INTERVAL_S        default 2.0
  WATCH_FRAME_PATH        if set, read JPEG from this path instead of the camera
  WATCH_STATE_PATH        default runtime/watcher_state.json
  WATCH_GOAL              default "set the table"
  WATCH_SKILLS_PATH       default skills/skills.yaml
"""
from __future__ import annotations

import base64
import json
import os
import sys
import time
from pathlib import Path

import requests
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import orchestrator.orchestrator as orch  # noqa: E402

WATCH_INTERVAL_S = float(os.getenv("WATCH_INTERVAL_S", "2.0"))
WATCH_FRAME_PATH = os.getenv("WATCH_FRAME_PATH")  # optional file source
WATCH_STATE_PATH = Path(os.getenv("WATCH_STATE_PATH", str(REPO_ROOT / "runtime" / "watcher_state.json")))
WATCH_GOAL = os.getenv("WATCH_GOAL", "set the table")
WATCH_SKILLS_PATH = Path(os.getenv("WATCH_SKILLS_PATH", str(REPO_ROOT / "skills" / "skills.yaml")))


def _read_frame() -> bytes:
    """Frame source: file (if WATCH_FRAME_PATH) else live camera."""
    if WATCH_FRAME_PATH:
        return Path(WATCH_FRAME_PATH).read_bytes()
    return orch.capture_frame(orch.CAMERA_INDEX)


def _load_skills() -> dict:
    with open(WATCH_SKILLS_PATH) as f:
        return yaml.safe_load(f)


def _ask_gemma(image: bytes, goal: str, skills: dict) -> dict:
    skills_brief = "\n".join(f"- {n}: {m['description']}" for n, m in skills.items())

    prompt = f"""You are a robotics co-pilot watching a scene while a robot arm
runs a skill. Your job is to summarize what you see and anticipate the next skill.

Goal: "{goal}"

Available skills (each can be invoked next):
{skills_brief}

Look at the current scene. Respond with JSON only, schema:
  {{
    "scene_description": "<one sentence of what you see now>",
    "current_skill_likely_complete": <true|false>,
    "anticipated_next_skill": "<exact skill name from list, or null if goal complete>",
    "anticipation_reason": "<one sentence why>",
    "concerns": [<list of strings — things that look wrong or risky>]
  }}
"""

    payload = {
        "model": orch.GEMMA_MODEL,
        "prompt": prompt,
        "images": [base64.b64encode(image).decode()],
        "stream": False,
        "format": "json",
    }
    headers = {"Authorization": f"Bearer {orch.GEMMA_PROXY_TOKEN}"} if orch.GEMMA_PROXY_TOKEN else {}
    r = requests.post(f"{orch.OLLAMA_HOST}/api/generate", json=payload, headers=headers, timeout=120)
    r.raise_for_status()
    return json.loads(r.json().get("response", "{}"))


def main() -> int:
    skills = _load_skills()
    WATCH_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)

    src = WATCH_FRAME_PATH if WATCH_FRAME_PATH else f"camera index {orch.CAMERA_INDEX}"
    print(f"watcher: source={src}  interval={WATCH_INTERVAL_S}s  state={WATCH_STATE_PATH}")
    print(f"  goal: {WATCH_GOAL}")
    print(f"  skills: {list(skills.keys())}\n")

    i = 0
    while True:
        i += 1
        t0 = time.time()
        try:
            frame = _read_frame()
            analysis = _ask_gemma(frame, WATCH_GOAL, skills)
            elapsed = time.time() - t0
            payload = {
                "ts": time.time(),
                "iteration": i,
                "elapsed_s": round(elapsed, 2),
                "analysis": analysis,
            }
            WATCH_STATE_PATH.write_text(json.dumps(payload, indent=2))
            print(f"[{i:03d} +{elapsed:4.1f}s] {analysis.get('scene_description', '?')}", flush=True)
            if analysis.get("concerns"):
                for c in analysis["concerns"]:
                    print(f"           concern: {c}", flush=True)
        except Exception as e:
            print(f"[{i:03d} ERROR] {type(e).__name__}: {e}", flush=True)

        sleep_for = max(0.0, WATCH_INTERVAL_S - (time.time() - t0))
        time.sleep(sleep_for)


if __name__ == "__main__":
    sys.exit(main())
