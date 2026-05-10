# Setup for Mattie's box (arms attached here)

The voice service has to run on the same computer the SO-101 arms are
USB-plugged into, because `lerobot-record` opens the COM ports directly.
Everything else (Gemma 4 on Spark, the Vapi assistant, the phone number)
stays where it is — you just point your local server at the same Vapi
account and the same Spark proxy.

End-state: a caller dials **`+1 (628) 251-3807`**, Vapi POSTs tool
calls to your box over a Tailscale Funnel HTTPS URL, your box spawns
`lerobot-record` against the arms.

---

## 1. Get the code

```bash
git clone https://github.com/corybarr/xlerobot-hackathon.git
cd xlerobot-hackathon/voice
```

## 2. Install the voice package

Python 3.11 or 3.12 — neither needs admin.

```bash
python3 -m venv .venv

# Linux / macOS:
.venv/bin/pip install -e .

# Windows (PowerShell):
.venv\Scripts\pip install -e .
```

## 3. Find your camera index

```bash
# Linux/macOS:
.venv/bin/python -c "import cv2;
for i in range(6):
    cap = cv2.VideoCapture(i)
    ok, frame = cap.read() if cap.isOpened() else (False, None)
    cap.release()
    print(f'index {i}:', 'available' if ok else 'no')"

# Windows: same one-liner, just .venv\Scripts\python
```

Whichever index is YOUR external workspace camera (NOT the laptop's
built-in selfie cam) — note it down. Often `0` on Linux, `1` on Windows.

## 4. Drop in `.env`

Copy `voice/.env.example` to `voice/.env` and fill in:

```bash
# ── Vapi (account-level — same keys for everyone on the team) ─────────
VAPI_PRIVATE_KEY=951fa51c-c64c-4162-83b7-406228674193
VAPI_PUBLIC_KEY=457eedeb-f937-41eb-8c94-fb22ab69c4e9
VAPI_PHONE_NUMBER_ID=02d61c15-235a-46da-bf85-4e6bed333265
VAPI_PHONE_NUMBER=+16282513807

# ── Gemma proxy on Spark (HTTPS Tailscale Funnel, public) ─────────────
OLLAMA_HOST=https://spark.tail4eec5f.ts.net
GEMMA_PROXY_TOKEN=ceb2bca782d8473f4f9345a5fb547057bc438bb3d52bda453c6903b162ba3847
GEMMA_MODEL=gemma3:27b      # swap to gemma4:26b-a4b-it-q4 after the Ollama upgrade

# ── Your camera (from step 3) ─────────────────────────────────────────
VISION_CAMERA_INDEX=1

# ── Your SO-101 ports ─────────────────────────────────────────────────
FOLLOWER_PORT=COM10         # Windows; on Linux /dev/ttyACM0 or similar
LEADER_PORT=COM7

# ── lerobot CLI on your box ───────────────────────────────────────────
# Path to the lerobot-record binary in YOUR lerobot venv / conda env.
# Linux example:  ~/miniconda3/envs/lerobot/bin/lerobot-record
# Windows example: C:/Users/<you>/miniconda3/envs/lerobot/Scripts/lerobot-record.exe
LEROBOT_RECORD_BIN=/path/to/lerobot-record

# ── Public host (filled in step 5) ────────────────────────────────────
WEBHOOK_HOST=
SERVER_PORT=8765
CONFIRM_BEFORE_EXECUTE=true
```

## 5. Public HTTPS via Tailscale Funnel

Your box has to be in the tailnet `tail4eec5f.ts.net` and have the
`funnel` node-attribute. If Ryan has shared Spark with you, you're
already a tailnet member — `tailscale status` should list your machine.

```bash
# Linux/macOS:
sudo tailscale funnel --bg 8765

# Windows (admin PowerShell):
tailscale funnel --bg 8765
```

The output prints something like
`https://<your-hostname>.tail4eec5f.ts.net`. Set that as
`WEBHOOK_HOST` in `.env`. Example: `https://mattie-laptop.tail4eec5f.ts.net`.

## 6. Start the server

```bash
# Linux/macOS:
.venv/bin/python -m voice.server

# Windows (double-click or run):
voice/run-server.cmd
```

Smoke test:

```bash
curl http://localhost:8765/health
curl https://<your-tailnet-host>/health   # public, via Funnel
```

Both should return `{"status":"ok",...}`.

## 7. Point Vapi at your box

Re-run the bootstrap; it's idempotent — PATCHes the existing
`xlerobot-demo` assistant with YOUR webhook URL.

```bash
.venv/bin/python vapi-bootstrap.py
```

You should see:

```
==> Upserting Vapi assistant 'xlerobot-demo'
  found existing assistant b3020bff-78f8-4d7b-9371-b0a33ea71da8 ...
==> Attaching phone number id 02d61c15-...
    number: +16282513807  status: active
==> Done. Dial: +16282513807
```

## 8. Dial it

`+1 (628) 251-3807` → ask "what do you see?" → "pick up the cup".

If `pick_cup` returns a `lerobot-record not found` error, double-check
`LEROBOT_RECORD_BIN` in `.env`. If the camera frame looks wrong, change
`VISION_CAMERA_INDEX` and restart the server.

## What's where right now

| Component | Where it runs | Why |
|---|---|---|
| Gemma 4 (LLM) | Spark | Has the GPU + the model; HTTPS Funnel makes it reachable |
| Vapi (voice IO + LLM router) | Vapi cloud | $0/mo trial; one number, no DIY voice stack |
| Voice tool server | **Your box** | Spawns lerobot-record; needs the arm USB |
| Cameras | **Your box** | cv2.VideoCapture on a USB cam plugged into you |
| SO-101 arms | **Your box** | USB serial on COM/ttyACM |
| Phone number | Vapi cloud | `+1 (628) 251-3807` |

Only the row marked "Your box" actually moves between machines.
Everything else is shared infrastructure.

## Sharing Spark + Vapi creds

Ryan can node-share Spark to you via Tailscale admin console
(network access only, no SSH) — that's enough for the Gemma proxy
URL to be reachable from your box. Vapi keys in `.env` are the
same for everyone; no per-user setup.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `look_at_scene` returns `gemma_failed: ConnectionRefused` | `OLLAMA_HOST` typo or you're not on the tailnet. `curl $OLLAMA_HOST` should redirect/respond. |
| `pick_*` returns `lerobot-record not found` | `LEROBOT_RECORD_BIN` wrong in `.env`. `which lerobot-record` to find it. |
| `pick_*` returns `another_skill_running` | A previous run didn't clean up. POST `/api/tools/cancel_current` (or call from the phone). |
| Vapi dashboard shows "no webhook response" | Tailscale Funnel might be off. `tailscale funnel status` should show `:8765`. |
| Server starts then exits silently | Make sure you're running with `python -m voice.server` (not `python voice/server.py` — that imports the module without calling main). On Windows use `run-server.cmd` or `start-server.py`. |
