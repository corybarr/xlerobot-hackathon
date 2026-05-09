"""
MolmoAct2-SO100_101 inference server.
Run this on the NVIDIA sponsor GPU box.

Start:
  uvicorn server:app --host 0.0.0.0 --port 8000

Health check:
  curl http://<gpu-host>:8000/health

Predict:
  POST /predict  with JSON body matching PredictRequest below.
"""
import base64
import io
from typing import Optional

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from PIL import Image
from pydantic import BaseModel
from transformers import AutoModelForImageTextToText, AutoProcessor

REPO = "allenai/MolmoAct2-SO100_101"
DEVICE = "cuda"
DTYPE = torch.bfloat16

app = FastAPI(title="MolmoAct2-SO100_101")
processor = None
model = None


class PredictRequest(BaseModel):
    task: str
    state: list[float]
    top_image_b64: str
    side_image_b64: Optional[str] = None
    num_steps: int = 10
    normalize_language: bool = True


@app.on_event("startup")
def load():
    global processor, model
    print(f"loading {REPO} on {DEVICE} ({DTYPE}) ...")
    processor = AutoProcessor.from_pretrained(REPO, trust_remote_code=True)
    model = (
        AutoModelForImageTextToText.from_pretrained(
            REPO, trust_remote_code=True, dtype=DTYPE
        )
        .to(DEVICE)
        .eval()
    )
    print("model ready")


def decode_image(b64: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")


@app.post("/predict")
def predict(req: PredictRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="model not loaded yet")
    images = [decode_image(req.top_image_b64)]
    if req.side_image_b64:
        images.append(decode_image(req.side_image_b64))
    state = np.array(req.state, dtype=np.float32)

    with torch.inference_mode(), torch.autocast("cuda", dtype=DTYPE):
        out = model.predict_action(
            processor=processor,
            images=images,
            task=req.task,
            state=state,
            norm_tag="so100_so101_molmoact2",
            action_mode="continuous",
            enable_depth_reasoning=False,
            num_steps=req.num_steps,
            normalize_language=req.normalize_language,
            enable_cuda_graph=False,
        )
    actions = np.asarray(out.actions).tolist()
    return {"actions": actions, "n_steps": len(actions)}


@app.get("/health")
def health():
    return {
        "ok": model is not None,
        "model": REPO,
        "device": DEVICE,
        "dtype": str(DTYPE),
    }
