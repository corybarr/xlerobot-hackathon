"""Interactive chat with Gemma about the current scene + orchestrator state.

Standalone REPL — separate from the main orchestrator loop. Use it to:
  - Ask "what do you see in the scene?"
  - Debug a stuck step ("why isn't the fork moving?")
  - Get Gemma's read on whether to re-record demos vs swap VLA vs add a new skill

Special commands:
  /refresh    — recapture the camera frame
  /history    — show the conversation so far
  /clear      — wipe conversation history (keeps current frame)
  /skills     — list known skills from skills.yaml
  /quit       — exit

Env (same names as orchestrator.py):
  OLLAMA_HOST, GEMMA_MODEL, CAMERA_INDEX, GOAL
"""
from __future__ import annotations

import base64
import json
import os
import sys
from pathlib import Path

import requests

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import orchestrator.orchestrator as orch  # noqa: E402


SYSTEM_PROMPT = """You are a robotics co-pilot for a hackathon team building a
table-setting robot using SO-101 arms. The user is asking you about what you
see in the current scene image (from the front camera) or about the
hierarchical orchestrator (Gemma planner + per-skill VLAs).

Be direct and useful. If you can't see something clearly in the image, say so.
If a question is about strategy (which VLA to pick, why a skill is failing),
reason from what you see and what you know about the available skills."""


def _gemma_chat(messages: list[dict], image: bytes | None = None, timeout: int = 120) -> str:
    """Free-form chat call (no JSON formatting). Returns Gemma's text reply."""
    payload = {
        "model": orch.GEMMA_MODEL,
        "messages": messages,
        "stream": False,
    }
    if image is not None:
        # /api/chat takes per-message images; attach to the LAST user message.
        for m in reversed(payload["messages"]):
            if m["role"] == "user":
                m["images"] = [base64.b64encode(image).decode()]
                break
    r = requests.post(f"{orch.OLLAMA_HOST}/api/chat", json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json().get("message", {}).get("content", "")


def main() -> int:
    print(f"chat with {orch.GEMMA_MODEL} via {orch.OLLAMA_HOST}")
    print(f"goal: {orch.GOAL}")
    print(f"camera index: {orch.CAMERA_INDEX}")
    print("commands: /refresh /history /clear /skills /quit\n")

    skills = orch.load_skills()
    skills_summary = "\n".join(f"- {n}: {m['description']}" for n, m in skills.items())

    print("capturing initial frame...")
    try:
        frame = orch.capture_frame(orch.CAMERA_INDEX)
        print(f"  got {len(frame)}-byte JPEG\n")
    except Exception as e:
        print(f"  WARN: camera capture failed ({type(e).__name__}: {e}). Continuing without image.\n")
        frame = None

    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT + f"\n\nKnown skills:\n{skills_summary}\n\nGoal: {orch.GOAL}"},
    ]
    image_attached_once = False

    while True:
        try:
            user = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not user:
            continue

        if user in ("/quit", "/q", "exit"):
            return 0
        if user == "/refresh":
            try:
                frame = orch.capture_frame(orch.CAMERA_INDEX)
                image_attached_once = False
                print(f"  recaptured {len(frame)}-byte JPEG")
            except Exception as e:
                print(f"  ERROR recapturing: {type(e).__name__}: {e}")
            continue
        if user == "/history":
            for i, m in enumerate(messages):
                if m["role"] == "system":
                    continue
                snippet = m["content"][:160].replace("\n", " ")
                print(f"  {i:2d} [{m['role']}] {snippet}")
            continue
        if user == "/clear":
            messages = messages[:1]  # keep system
            image_attached_once = False
            print("  history cleared")
            continue
        if user == "/skills":
            print(skills_summary)
            continue

        messages.append({"role": "user", "content": user})

        # Attach the current frame on the FIRST user message after each /refresh,
        # so we don't blow up the context with 100 copies of the same image.
        attach_image = frame if (frame is not None and not image_attached_once) else None
        try:
            reply = _gemma_chat(messages, image=attach_image)
            if attach_image is not None:
                image_attached_once = True
        except requests.HTTPError as e:
            print(f"  HTTP error: {e}. Server may be down or model not pulled.")
            messages.pop()  # don't keep the un-replied user message
            continue
        except Exception as e:
            print(f"  ERROR: {type(e).__name__}: {e}")
            messages.pop()
            continue

        messages.append({"role": "assistant", "content": reply})
        print(f"\n{reply}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
