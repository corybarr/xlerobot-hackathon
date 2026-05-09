"""One-shot question to Gemma with an image attached.

By default captures the front camera; pass --image PATH to use an existing JPEG.
Useful for quick diagnostic questions ("what do you see?", "is the fork in the
cup?") without firing up the full chat REPL.

Usage:
    python orchestrator/ask.py "What objects are on the table right now?"
    python orchestrator/ask.py "Where is the fork?" --image scene.jpg
    python orchestrator/ask.py "describe this" --save-image scene.jpg
"""
from __future__ import annotations

import argparse
import base64
import sys
from pathlib import Path

import requests

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import orchestrator.orchestrator as orch  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("question", help="What to ask Gemma about the scene")
    p.add_argument("--image", help="JPEG path; if omitted, captures from camera")
    p.add_argument("--save-image", default=None, help="Save captured frame to this path")
    p.add_argument("--system", default=None, help="Optional system-style preamble")
    args = p.parse_args()

    if args.image:
        img = Path(args.image).read_bytes()
    else:
        img = orch.capture_frame(orch.CAMERA_INDEX)
        if args.save_image:
            Path(args.save_image).write_bytes(img)
            print(f"  saved frame to {args.save_image}", file=sys.stderr)

    prompt = args.question
    if args.system:
        prompt = f"{args.system}\n\n{args.question}"

    payload = {
        "model": orch.GEMMA_MODEL,
        "prompt": prompt,
        "images": [base64.b64encode(img).decode()],
        "stream": False,
    }
    r = requests.post(f"{orch.OLLAMA_HOST}/api/generate", json=payload, timeout=180)
    r.raise_for_status()
    print(r.json().get("response", "").strip())
    return 0


if __name__ == "__main__":
    sys.exit(main())
