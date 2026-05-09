"""Gemma-orchestrated VLA executor for Set-the-Table.

Architecture:
  high-level planner   = Gemma (vision-LM, runs on Spark via Ollama tunnel)
  low-level executors  = per-skill VLAs (SmolVLA / ACT / MolmoAct2),
                         each fine-tuned on demos for ONE discrete action

Loop:
  1. SELECT — Gemma sees current scene + history + skills (with pre/post-conditions),
              picks next skill OR declares done
  2. INVOKE — wrap the chosen skill's VLA in a uniform subprocess call
              (build_vla_command knows how each backend launches)
  3. VERIFY — every VERIFY_INTERVAL_S seconds while the VLA runs, Gemma compares
              the pre-action frame with the current frame and classifies:
                in_progress  -> let it keep running
                completed    -> terminate VLA, replan from new scene
                problem      -> abort, replan with the failure recorded
  4. RECORD — append (skill, verdict, reason) to history; loop until done or MAX_STEPS

Adding a new VLA backend: add a branch to build_vla_command(). That's it.

Env:
  OLLAMA_HOST         default http://localhost:11434
  GEMMA_MODEL         default gemma3:27b
  CAMERA_INDEX        default 1 (front_cam per lerobot-MakerMods convention)
  GOAL                default "set the table"
  HF_USER             default Globalmysterysnailrevolution
  VLA_BACKEND         default smolvla
  MAX_STEPS           default 20  (safety cap on planner iterations)
  VERIFY_INTERVAL_S   default 3.0 (how often to ask Gemma about state mid-skill)
  SKILL_TIMEOUT_S     default 30.0
  FOLLOWER_PORT       default COM10
  LEADER_PORT         default COM7
"""
from __future__ import annotations

import base64
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
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
VERIFY_INTERVAL_S = float(os.getenv("VERIFY_INTERVAL_S", "3.0"))
SKILL_TIMEOUT_S = float(os.getenv("SKILL_TIMEOUT_S", "30.0"))
FOLLOWER_PORT = os.getenv("FOLLOWER_PORT", "COM10")
LEADER_PORT = os.getenv("LEADER_PORT", "COM7")

REPO_ROOT = Path(__file__).resolve().parent.parent
SKILLS_PATH = REPO_ROOT / "skills" / "skills.yaml"


@dataclass
class StepRecord:
    skill: str
    select_reason: str
    verdict: str   # completed | problem | timeout
    verify_reason: str


def load_skills() -> dict:
    with open(SKILLS_PATH) as f:
        return yaml.safe_load(f)


def capture_frame(camera_index: int) -> bytes:
    cap = cv2.VideoCapture(camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError(f"Could not read from camera {camera_index}")
    _, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return jpeg.tobytes()


def _gemma_call(images: list[bytes], prompt: str, timeout: int = 60) -> dict:
    """Call Gemma with one or more images + prompt. Returns parsed JSON."""
    payload = {
        "model": GEMMA_MODEL,
        "prompt": prompt,
        "images": [base64.b64encode(im).decode() for im in images],
        "stream": False,
        "format": "json",
    }
    r = requests.post(f"{OLLAMA_HOST}/api/generate", json=payload, timeout=timeout)
    r.raise_for_status()
    return json.loads(r.json().get("response", "{}"))


# ---------------------------------------------------------------------------
# Selection: Gemma picks the next skill
# ---------------------------------------------------------------------------

def select_next_skill(scene: bytes, goal: str, skills: dict, history: list[StepRecord]) -> dict:
    skills_brief = "\n".join(
        f"- {n}: {m['description']}\n"
        f"    pre:  {m.get('preconditions', '—')}\n"
        f"    post: {m.get('postconditions', '—')}"
        for n, m in skills.items()
    )
    history_brief = (
        "\n".join(
            f"  step {i+1}: {h.skill} -> {h.verdict} ({h.verify_reason})"
            for i, h in enumerate(history)
        )
        or "  (none yet)"
    )

    # Pipelined planning: if the background watcher.py is running, it will have
    # already analyzed the scene and possibly anticipated the next skill while
    # the previous VLA was executing. Pull that as a hint to confirm or override.
    watcher_hint = "  (no watcher state available)"
    watcher_path = REPO_ROOT / "runtime" / "watcher_state.json"
    if watcher_path.exists():
        try:
            ws_full = json.loads(watcher_path.read_text())
            ws = ws_full.get("analysis", {})
            age_s = round(time.time() - ws_full.get("ts", time.time()), 1)
            watcher_hint = (
                f"  scene: {ws.get('scene_description', '—')}\n"
                f"  anticipated next: {ws.get('anticipated_next_skill', '—')} "
                f"({ws.get('anticipation_reason', '—')})\n"
                f"  concerns: {ws.get('concerns', [])}\n"
                f"  state age: {age_s}s"
            )
        except Exception as e:
            watcher_hint = f"  (watcher state unreadable: {type(e).__name__})"

    prompt = f"""You are a robot task planner. Look at the current scene and pick the next single skill.

Goal: "{goal}"

Available skills (each is a separately trained VLA you can invoke):
{skills_brief}

History (most recent last):
{history_brief}

Concurrent watcher (a separate Gemma loop watching the scene; may be a few seconds stale):
{watcher_hint}

Pick the skill whose preconditions match the current scene AND whose postconditions move you toward the goal. The watcher's anticipation is a hint — confirm it if it matches the live scene, override if you see something different. Avoid repeating a skill that just succeeded.

Respond with JSON only. Schema:
  {{"done": true,  "reason": "<one sentence why complete>"}}
or
  {{"done": false, "skill": "<exact skill name from list>", "reason": "<one sentence why this skill next>"}}
"""
    return _gemma_call([scene], prompt)


# ---------------------------------------------------------------------------
# Verification: Gemma compares pre-action and current frames mid-execution
# ---------------------------------------------------------------------------

def verify_skill_state(pre: bytes, current: bytes, skill_name: str, skill_meta: dict) -> dict:
    """Two-image comparison: classify whether the in-flight skill is progressing,
    done, or in trouble. Called periodically while the VLA subprocess runs.
    """
    prompt = f"""You are watching a robot mid-skill. Two images: BEFORE the skill started, and CURRENT scene.

Skill: {skill_name}
Description: {skill_meta['description']}
Preconditions (state when skill started): {skill_meta.get('preconditions', '—')}
Postconditions (success criteria): {skill_meta.get('postconditions', '—')}

Compare BEFORE vs CURRENT and classify:
- "in_progress" — meaningfully progressing toward postconditions but not done yet
- "completed"   — postconditions are clearly satisfied
- "problem"     — collision, dropped item, wrong target, scene became inconsistent with what the skill should be doing, or stalled with no progress

Respond with JSON only. Schema:
  {{"state": "in_progress" | "completed" | "problem", "reason": "<one sentence describing what you see>"}}
"""
    return _gemma_call([pre, current], prompt)


# ---------------------------------------------------------------------------
# VLA wrappers — uniform subprocess interface across SmolVLA / ACT / MolmoAct2
# ---------------------------------------------------------------------------

def build_vla_command(skill_name: str, vla_backend: str) -> list[str]:
    """Construct subprocess command for the chosen (skill, backend) pair.
    Adding a new backend = add one branch here. The orchestrator main loop
    is backend-agnostic — it just runs whatever this returns.
    """
    policy_repo = f"{HF_USER}/xlerobot-{skill_name}-{vla_backend}"

    if vla_backend in ("smolvla", "act"):
        return [
            "lerobot-record",
            "--robot.type=so101_follower",
            f"--robot.port={FOLLOWER_PORT}",
            "--robot.id=arm_a_follower",
            "--teleop.type=so101_leader",
            f"--teleop.port={LEADER_PORT}",
            "--teleop.id=arm_a_leader",
            f"--policy.path={policy_repo}",
            "--dataset.num_episodes=1",
            "--dataset.episode_time_s=15",
            "--dataset.repo_id=local-eval",
            "--dataset.push_to_hub=false",
        ]

    if vla_backend == "molmoact2":
        # MolmoAct2 has its own runner outside lerobot — wrap your fine-tune
        # script in orchestrator/molmoact2_runner.py and invoke it here.
        return [
            sys.executable, "-m", "orchestrator.molmoact2_runner",
            "--policy", policy_repo,
            "--port", FOLLOWER_PORT,
        ]

    raise ValueError(f"Unknown VLA backend: {vla_backend}")


def execute_skill_with_verification(
    skill_name: str,
    skill_meta: dict,
    vla_backend: str,
) -> tuple[str, str]:
    """Invoke the VLA, periodically ask Gemma to classify state, terminate early
    on completed/problem. Returns (verdict, reason).
    """
    pre_frame = capture_frame(CAMERA_INDEX)
    print(f"  [{skill_name}] pre-frame captured, invoking {vla_backend}")

    cmd = build_vla_command(skill_name, vla_backend)
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

    start = time.time()
    last_verify = start

    try:
        while proc.poll() is None:
            now = time.time()

            if now - start > SKILL_TIMEOUT_S:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                return ("timeout", f"VLA exceeded {SKILL_TIMEOUT_S}s")

            if now - last_verify >= VERIFY_INTERVAL_S:
                last_verify = now
                try:
                    current = capture_frame(CAMERA_INDEX)
                    verdict = verify_skill_state(pre_frame, current, skill_name, skill_meta)
                    state = verdict.get("state", "in_progress")
                    reason = verdict.get("reason", "")
                    print(f"  [{skill_name}] {now-start:5.1f}s: {state} — {reason}")

                    if state in ("completed", "problem"):
                        proc.terminate()
                        try:
                            proc.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            proc.kill()
                        return (state, reason)
                except Exception as e:
                    print(f"  [{skill_name}] verify error: {type(e).__name__}: {e}")

            time.sleep(0.5)

        # VLA exited on its own — final state check
        post = capture_frame(CAMERA_INDEX)
        verdict = verify_skill_state(pre_frame, post, skill_name, skill_meta)
        state = verdict.get("state", "in_progress")
        return (state, verdict.get("reason", "VLA exited; final state check"))

    finally:
        if proc.poll() is None:
            proc.terminate()


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main() -> int:
    skills = load_skills()
    history: list[StepRecord] = []

    print(f"orchestrator: goal='{GOAL}'  vla={VLA_BACKEND}  model={GEMMA_MODEL}")
    print(f"  skills: {list(skills.keys())}")
    print(f"  verify every {VERIFY_INTERVAL_S}s, skill timeout {SKILL_TIMEOUT_S}s\n")

    for step in range(MAX_STEPS):
        print(f"--- step {step+1}/{MAX_STEPS} ---")
        try:
            scene = capture_frame(CAMERA_INDEX)
            decision = select_next_skill(scene, GOAL, skills, history)
        except Exception as e:
            print(f"  planner error: {type(e).__name__}: {e}")
            return 1

        print(f"  gemma decision: {decision}")

        if decision.get("done"):
            print(f"\nGoal complete: {decision.get('reason')}")
            return 0

        skill = decision.get("skill")
        if skill not in skills:
            print(f"  ERROR: gemma chose unknown skill '{skill}'. Aborting.")
            return 1

        verdict, reason = execute_skill_with_verification(skill, skills[skill], VLA_BACKEND)
        history.append(StepRecord(
            skill=skill,
            select_reason=decision.get("reason", ""),
            verdict=verdict,
            verify_reason=reason,
        ))

        if verdict == "problem":
            print(f"  problem during {skill} — replanning from updated scene")
        elif verdict == "timeout":
            print(f"  {skill} timed out — replanning")

    print("\nReached max steps without goal completion.")
    return 2


if __name__ == "__main__":
    sys.exit(main())
