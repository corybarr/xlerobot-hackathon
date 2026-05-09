# MolmoAct2 robot interface

Two-piece setup: server on a CUDA GPU (sponsor box), client on your Mac.

```
client (Mac, .venv)              server (NVIDIA box)
┌──────────────────┐             ┌─────────────────────────┐
│ camera capture   │  HTTP/JSON  │ MolmoAct2-SO100_101     │
│ arm state read   │ ──────────► │  predict_action(...)    │
│ task prompt      │             │                         │
│                  │ ◄────────── │ action chunk (6D × N)   │
│ send to arm      │             │                         │
└──────────────────┘             └─────────────────────────┘
```

## Server (deploy on the NVIDIA sponsor GPU)

Needs ~16 GB GPU memory in bfloat16. A10G or larger.

```bash
git clone <this-folder> molmoact-interface
cd molmoact-interface/server
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn server:app --host 0.0.0.0 --port 8000
```

First request triggers model download (~26 GB) — takes a few minutes once.

Open the firewall on port 8000 (or use an SSH tunnel from your Mac).

## Client (Mac — already set up at ~/Projects/LeRobot/.venv)

The local venv already has `opencv-python`, `pillow`, `numpy`, `huggingface_hub`. You only need to add `requests`:

```bash
cd ~/Projects/LeRobot && source .venv/bin/activate
uv pip install requests   # if not already installed
```

### Sanity-check the server with the HF sample images

```bash
python molmoact-interface/client/test_client.py http://<gpu-host>:8000
```

Should print `OK` and show 1+ predicted actions.

### Run with real cameras + prompt

```bash
python molmoact-interface/client/client.py \
  --server http://<gpu-host>:8000 \
  --top 0 --side 1 \
  --task "Pick up the cup and place it on the saucer." \
  --state 0,0,0,0,0,0
```

`--state` is a 6-float vector matching the SO-100/101 joint convention. Leave it `0,0,0,0,0,0` for the first end-to-end test; replace `read_arm_state()` in `client.py` once the dynamixel readout is wired.

## What's stubbed (mark before you ship)

- `client/client.py:read_arm_state()` — returns zeros. Replace with a `dynamixel_sdk` readout from the follower arm.
- `client/client.py:send_actions_to_arm()` — no-op. Iterate the action chunk and command each 6D pose to the SO-100/101 servos. Action space is **absolute joint pose, robot scale** (per the model card), not deltas.

## Known constraints (from the model card)

- Continuous-action mode is recommended (`action_mode="continuous"`).
- `enable_depth_reasoning=False` is required — this checkpoint will error if you set it `True`.
- `norm_tag="so100_so101_molmoact2"` is mandatory — the action denormalization depends on it.
- bfloat16 fits on a 16 GB GPU; float32 needs ~26 GB.
- First few `/predict` calls are slow (model warmup). Don't time the first one.

## When the demo runs

Loop mode (capture → predict → execute repeatedly):

```bash
python client/client.py --server ... --task "..." --loop
```

Each call returns a chunk of actions (`num_steps=10` by default). The arm should execute the whole chunk before the next prediction is requested — otherwise actions stack up and the arm thrashes.
