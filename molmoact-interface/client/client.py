"""
Local client for the MolmoAct2 sponsor-GPU server.

Captures one frame per camera, sends to the server, prints the action chunk.
Wire the action chunk to the SO-100/101 follower arm where marked TODO.

Usage:
  python client.py --server http://<gpu-host>:8000 \
                   --top 0 --side 1 \
                   --task "Pick up the cup and place it on the saucer." \
                   --state 0,0,0,0,0,0
"""
import argparse
import base64
import io
import time

import cv2
import numpy as np
import requests
from PIL import Image


def pil_to_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def grab_frame(cap: cv2.VideoCapture) -> Image.Image:
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError("no frame from camera")
    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


def read_arm_state() -> list[float]:
    """
    TODO: replace with real dynamixel readout for the SO-100/101 follower.
    Returns a 6D float vector matching the training norm tag.
    """
    return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


def send_actions_to_arm(actions: list[list[float]]) -> None:
    """
    TODO: send each 6D action in the chunk to the dynamixel servos in order.
    Use dynamixel_sdk; see SO-ARM100/ for example wiring.
    """
    pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default="http://localhost:8000")
    ap.add_argument("--top", type=int, default=0, help="top camera index")
    ap.add_argument("--side", type=int, default=None, help="side camera index (optional)")
    ap.add_argument("--task", required=True, help="natural-language instruction")
    ap.add_argument(
        "--state",
        default=None,
        help="6 comma-separated floats; if omitted, reads from the arm (TODO)",
    )
    ap.add_argument("--loop", action="store_true", help="capture/predict/execute repeatedly")
    args = ap.parse_args()

    top_cap = cv2.VideoCapture(args.top)
    side_cap = cv2.VideoCapture(args.side) if args.side is not None else None
    if not top_cap.isOpened():
        raise SystemExit(f"cannot open top camera {args.top}")
    if side_cap is not None and not side_cap.isOpened():
        raise SystemExit(f"cannot open side camera {args.side}")

    def one_step():
        state = (
            [float(x) for x in args.state.split(",")]
            if args.state
            else read_arm_state()
        )
        top = grab_frame(top_cap)
        side = grab_frame(side_cap) if side_cap is not None else None
        payload = {
            "task": args.task,
            "state": state,
            "top_image_b64": pil_to_b64(top),
        }
        if side is not None:
            payload["side_image_b64"] = pil_to_b64(side)
        t0 = time.time()
        r = requests.post(f"{args.server}/predict", json=payload, timeout=120)
        r.raise_for_status()
        data = r.json()
        dt = time.time() - t0
        actions = data["actions"]
        print(f"[{dt:.2f}s] {data['n_steps']} actions; first={np.round(actions[0], 3).tolist()}")
        send_actions_to_arm(actions)

    try:
        if args.loop:
            while True:
                one_step()
        else:
            one_step()
    finally:
        top_cap.release()
        if side_cap is not None:
            side_cap.release()


if __name__ == "__main__":
    main()
