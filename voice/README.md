# voice — phone-call frontend for the xlerobot orchestrator

You call a number. Gemma 4 picks up. You say what you want done. Gemma
looks at the camera, picks a trained per-skill VLA from `skills.yaml`,
dispatches `lerobot-record` on the arm, narrates verdicts as the
orchestrator's verifier returns them.

This package is intentionally tiny — every action is one line into
`orchestrator/orchestrator.py`. The voice layer is just Pipecat +
FastAPI on top.

```
┌─────────┐  PSTN   ┌─────────┐  WSS   ┌─────────────────────────────────────┐
│ Phone   │ ──────▶ │ Twilio  │ ─────▶ │  voice/  (Pipecat sidecar)          │
└─────────┘         └─────────┘        │                                     │
                                       │  Deepgram STT ┐                     │
                                       │               ▼                     │
                                       │       OpenAILLMService  ────────┐   │
                                       │       (Gemma 4 via gemma-proxy) │   │
                                       │               ▲                 ▼   │
                                       │    ElevenLabs TTS         VoiceTools│
                                       └───────────────┼─────────────────┼───┘
                                                       │                 │
                                              speaks back              calls
                                                       │                 │
                                                       │                 ▼
                                       ┌───────────────┴─────────────────────┐
                                       │  orchestrator/orchestrator.py        │
                                       │   - capture_frame()                 │
                                       │   - _gemma_call() [Gemma 4 proxy]   │
                                       │   - select_next_skill()             │
                                       │   - execute_skill_with_verification │
                                       │           │                          │
                                       │           ▼                          │
                                       │   subprocess: lerobot-record         │
                                       │   --policy.path=<HF repo>            │
                                       └──────────────┬──────────────────────┘
                                                      ▼
                                              [SO-101 follower on COM10]
```

## Quick start

```bash
# 1. From the repo root.
cd voice
python3.11 -m venv .venv
.venv/bin/pip install -e .            # pulls pipecat-ai, opens orchestrator/ as a peer

# 2. .env: Twilio + Deepgram + ElevenLabs + the Gemma proxy URL/token.
cat > .env <<'EOF'
DEEPGRAM_API_KEY=...
ELEVENLABS_API_KEY=...
TWILIO_ACCOUNT_SID=AC...
TWILIO_AUTH_TOKEN=...
TWILIO_PHONE_NUMBER=+1...

# Where Pipecat's OpenAILLMService POSTs /v1/chat/completions.
# Local (server runs on Spark next to ollama):  http://localhost:11434/v1
# Off-Spark (server runs on the tablet):        https://bore.pub:<port>/v1
OLLAMA_BASE_URL=http://localhost:11434/v1
OLLAMA_API_KEY=ollama              # or your gemma-proxy bearer token
OLLAMA_MODEL=gemma4:26b-a4b-it-q4

# The orchestrator uses its own variables — keep these in sync.
GEMMA_MODEL=gemma4:26b-a4b-it-q4
OLLAMA_HOST=http://localhost:11434
GEMMA_PROXY_TOKEN=                 # set if going through the proxy
CAMERA_INDEX=1
FOLLOWER_PORT=COM10
LEADER_PORT=COM7

# Public host Twilio will dial into.  Tailscale Funnel or bore.pub URL.
WEBHOOK_HOST=https://<your-public-host>

SERVER_PORT=8765
CONFIRM_BEFORE_EXECUTE=true
EOF

# 3. Start it.
.venv/bin/python -m voice.server

# 4. Expose to the public internet.
#    Option A — Tailscale Funnel (if your tailnet has it enabled):
#       sudo tailscale funnel --bg 8765
#    Option B — bore.pub (same path the gemma-proxy uses):
#       bore local 8765 --to bore.pub
#
# 5. Point Twilio at  https://<public-host>/twilio/inbound

# Smoke check
curl http://localhost:8765/health | jq
curl http://localhost:8765/api/agent.json | jq '.tools[].function.name'
curl -X POST http://localhost:8765/api/tools/list_skills -d '{}' \
     -H 'Content-Type: application/json' | jq
```

## What the agent can do

The voice agent is told, in its system prompt, exactly what each tool
does and how to chain them. A typical call:

1. Caller: _"pick up the cutlery"_.
2. Agent calls `look_at_scene()` — Gemma multimodal describes the frame.
3. Agent: _"I see a fork and a bowl on the table. I'll use `pick_cutlery`
   — that's our trained SmolVLA for forks/knives/spoons. Should I go?"_
4. Caller: _"yes"_.
5. Agent calls `execute_skill(skill="pick_cutlery")`. The orchestrator:
   - captures a pre-frame
   - spawns `lerobot-record --policy.path=Globalmysterysnailrevolution/xlerobot-pick-cutlery-smolvla …`
   - every 3 s asks Gemma to compare pre + current and emits
     `in_progress` / `completed` / `problem`
   - kills the policy on the first `completed` or `problem`
6. Agent reads the verdict aloud. If `problem`, describes the reason.
7. If more steps are queued (e.g. caller said _"do all three"_),
   continues with `queue_skills`.

## Tool surface

The same schemas are served at `GET /api/agent.json` for any non-voice
integrator (text agent, MCP host, n8n flow).

| Tool | Wraps |
|---|---|
| `look_at_scene(focus?)` | `orchestrator.capture_frame` + `_gemma_call` describe |
| `list_skills()` | `orchestrator.load_skills()` |
| `propose_skill(goal?)` | `orchestrator.select_next_skill()` |
| `execute_skill(skill)` | `orchestrator.execute_skill_with_verification()` |
| `queue_skills(steps)` | loops `execute_skill`, bails on first non-completed |
| `get_status()` | recent history + last frame age |
| `cancel_current()` | flag-armed, honored between queued steps |

## Skills

The catalog is `skills/skills.yaml` at the repo root. To add a new skill:

1. Record + train per `wiki/data-collection-plan.md` and
   `wiki/training-plan.md`.
2. Push the checkpoint to HF Hub.
3. Append a block to `skills.yaml` with `vla.uri`, `description`,
   `preconditions`, `postconditions`.
4. Restart the voice service. The new skill auto-appears in
   `list_skills()` and the system prompt's selection guidance.

## Driving from another agent (no voice)

Any LLM agent can drive this service over HTTP:

```python
import httpx, json
manifest = httpx.get("https://<public-host>/api/agent.json").json()
resp = your_llm.chat.completions.create(
    model="...",
    messages=[
        {"role": "system", "content": manifest["system_prompt"]},
        {"role": "user",   "content": "pick up the bowl"},
    ],
    tools=manifest["tools"],
)
for call in resp.choices[0].message.tool_calls or []:
    args = json.loads(call.function.arguments)
    result = httpx.post(
        f"https://<public-host>/api/tools/{call.function.name}", json=args
    ).json()
    # feed result back as a tool message…
```

The same surface works as an MCP server (one tool per
`POST /api/tools/{name}`).

## Where this runs

The arm is on COM10 (Windows tablet via `lerobot-record`). The voice
service therefore runs on the tablet too — `lerobot-record` is the
subprocess it spawns, and `cv2.VideoCapture(CAMERA_INDEX)` needs the
camera attached locally.

The Gemma proxy is on Spark; the tablet reaches it via the
bore.pub tunnel (`OLLAMA_HOST=http://bore.pub:<port>`) or Tailscale.

For demo / dry-run without the arm, set `MANIPULATION_MODE`-equivalent
behavior by pointing `FOLLOWER_PORT=COM10` at an unused COM port — the
subprocess will fail fast, the verdict comes back as `problem`, and the
voice agent narrates the failure without any motion happening.

## Tests

```bash
cd voice && .venv/bin/pytest tests/ -v
```

`tests/` mocks `orchestrator.capture_frame`, `_gemma_call`, and
`execute_skill_with_verification` — no camera, no arm, no Gemma needed
for unit tests.
