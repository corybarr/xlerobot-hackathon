"""
End-to-end sanity test for the MolmoAct2 server.
Uses the sample RealSense images shipped with the HF model — no camera, no arm.
Run this FIRST after the server boots, before plugging anything in.

Usage:
  python test_client.py http://<gpu-host>:8000
"""
import base64
import sys
import time

import requests
from huggingface_hub import hf_hub_download

REPO = "allenai/MolmoAct2-SO100_101"
SERVER = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"

print(f"checking server health at {SERVER} ...")
h = requests.get(f"{SERVER}/health", timeout=10).json()
print(f"  {h}")
if not h.get("ok"):
    raise SystemExit("server not ready")

print("downloading sample images from HF ...")
top = hf_hub_download(REPO, "assets/sample_realsense_top_rgb.png")
side = hf_hub_download(REPO, "assets/sample_realsense_side_rgb.png")


def b64(path: str) -> str:
    return base64.b64encode(open(path, "rb").read()).decode()


payload = {
    "task": "Move the arm towards the lemon, grasp it, lift it up, and drop it into the red bowl.",
    "state": [
        -0.52734375,
        189.140625,
        181.40625,
        60.64453125,
        -3.603515625,
        1.0971786975860596,
    ],
    "top_image_b64": b64(top),
    "side_image_b64": b64(side),
}

print("sending /predict ...")
t0 = time.time()
r = requests.post(f"{SERVER}/predict", json=payload, timeout=180)
r.raise_for_status()
data = r.json()
dt = time.time() - t0

print(f"  {dt:.2f}s · {data['n_steps']} actions")
print(f"  first action: {data['actions'][0]}")
print(f"  last action:  {data['actions'][-1]}")
print("OK")
