"""Continuously watch the camera, send each frame to Gemma every N seconds.

Designed to be streamed under a monitor (e.g. Claude Code's Monitor tool, or
just `python orchestrator/watch.py | tee scene.log` from a terminal). Each
Gemma response is printed as one line so it groups naturally as one event.

Default prompt is set-the-table-aware. Pass --prompt to override.

Usage:
    python orchestrator/watch.py
    python orchestrator/watch.py --interval 3 --max-loops 10
    python orchestrator/watch.py --prompt "Where is the fork right now?"

Env: OLLAMA_HOST, GEMMA_MODEL, CAMERA_INDEX (same as orchestrator.py)
"""
from __future__ import annotations

import argparse
import base64
import sys
import time
from pathlib import Path

import requests

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import orchestrator.orchestrator as orch  # noqa: E402


DEFAULT_PROMPT = (
    "In one sentence, describe the scene. Focus on objects relevant to setting a "
    "table (fork, knife, spoon, plate, cup, napkin) and the robot arm's gripper. "
    "If anything looks wrong (dropped item, collision, missing object), call it out."
)


def ask(image: bytes, prompt: str, timeout: int = 60) -> str:
    payload = {
        "model": orch.GEMMA_MODEL,
        "prompt": prompt,
        "images": [base64.b64encode(image).decode()],
        "stream": False,
    }
    r = requests.post(f"{orch.OLLAMA_HOST}/api/generate", json=payload, timeout=timeout)
    r.raise_for_status()
    # collapse newlines so the line-per-event Monitor pattern works
    return r.json().get("response", "").strip().replace("\n", " ")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--interval", type=float, default=2.0,
                   help="seconds between captures (clamped down if Gemma is slower)")
    p.add_argument("--prompt", default=DEFAULT_PROMPT)
    p.add_argument("--max-loops", type=int, default=0, help="0 = unbounded")
    p.add_argument("--save-every", default=None,
                   help="If set, save each frame to this directory (one .jpg per cycle)")
    args = p.parse_args()

    save_dir: Path | None = None
    if args.save_every:
        save_dir = Path(args.save_every)
        save_dir.mkdir(parents=True, exist_ok=True)

    i = 0
    while True:
        i += 1
        t0 = time.time()
        try:
            frame = orch.capture_frame(orch.CAMERA_INDEX)
            if save_dir:
                (save_dir / f"frame-{i:04d}.jpg").write_bytes(frame)
            response = ask(frame, args.prompt)
            elapsed = time.time() - t0
            print(f"[{i:03d} +{elapsed:4.1f}s] {response}", flush=True)
        except Exception as e:
            print(f"[{i:03d} ERROR] {type(e).__name__}: {e}", flush=True)

        if args.max_loops and i >= args.max_loops:
            break

        sleep_for = max(0.0, args.interval - (time.time() - t0))
        time.sleep(sleep_for)

    return 0


if __name__ == "__main__":
    sys.exit(main())
