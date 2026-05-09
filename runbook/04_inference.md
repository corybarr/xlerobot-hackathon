# Inference — running the trained policy on the real arm

After training pushes to HF Hub, you have two ways to deploy.

## Option A — local Mac inference (ACT only, NOT SmolVLA)

ACT runs on M3 MPS. SmolVLA does NOT (CUDA-only).

```bash
cd ~/Projects/LeRobot && source .venv/bin/activate
export DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/opt/ffmpeg@7/lib

# This runs lerobot-record but with a policy (instead of teleop):
lerobot-record \
  --robot.type=so101_follower \
  --robot.port=/dev/cu.usbmodem<FOLLOWER_SERIAL> \
  --robot.id=test8 \
  --robot.cameras='{"front_cam":{"type":"opencv","index_or_path":0,"width":640,"height":480,"fps":30},"side_cam":{"type":"opencv","index_or_path":1,"width":640,"height":480,"fps":30}}' \
  --policy.path=<YOUR_USERNAME>/grab_cup_act \
  --dataset.repo_id=<YOUR_USERNAME>/eval_grab_cup_act \
  --dataset.single_task="<SAME TASK STRING AS TRAINING>" \
  --dataset.num_episodes=10 \
  --dataset.episode_time_s=30 \
  --display_data=true
```

Replace `<FOLLOWER_SERIAL>`, `<YOUR_USERNAME>`, and the task string. Camera names + indices must match what was used during recording.

## Option B — RunPod-hosted SmolVLA + Mac arm client (for SmolVLA only)

SmolVLA needs CUDA, so the model server stays on RunPod, the arm controller stays on your Mac. They talk over HTTP.

The scaffold for this pattern already exists in [molmoact-interface/](../molmoact-interface/) — built earlier this weekend for MolmoAct2 but the architecture is identical:

- `molmoact-interface/server/server.py` — FastAPI server, runs on RunPod, accepts `POST /predict` with images + state, returns action chunk.
- `molmoact-interface/client/client.py` — Mac client, captures cameras + arm state, posts to server, receives actions, sends to follower.

To adapt for SmolVLA:
1. Copy `molmoact-interface/` → `smolvla-interface/`
2. In `server.py`, replace the `AutoModelForImageTextToText` load with the SmolVLA HF API:
   ```python
   from transformers import AutoModel
   model = AutoModel.from_pretrained("<YOUR_USERNAME>/grab_cup_smolvla", trust_remote_code=True).to("cuda").eval()
   ```
3. Update the `predict_action` call signature to match SmolVLA's interface (see [lerobot/smolvla_base on HF](https://huggingface.co/lerobot/smolvla_base) for examples).
4. On the pod: `pip install fastapi uvicorn`, run `uvicorn server:app --host 0.0.0.0 --port 8000`.
5. On Mac: open SSH tunnel `ssh -L 8000:localhost:8000 root@<POD>` so client can hit `http://localhost:8000`.

This is ~30 min of adaptation work. Worth it if SmolVLA actually outperforms ACT on your data; otherwise stick with ACT + Option A.

## Cost during inference

- **Option A (ACT local):** $0 — runs on your Mac.
- **Option B (SmolVLA remote):** RunPod pod must stay alive during demo. ~$0.40/hr on A40.

## Demo-day timing

- Sunday 4 PM: submissions close, demos start.
- Plan to have inference WORKING by 2 PM Sunday (2-hour buffer).
- Re-deploy the RunPod pod ~2:30 PM if it was torn down after training. Boot + load takes ~3 min.
