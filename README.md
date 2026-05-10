# xlerobot-hackathon

Team entry for the **Physical AI Hack World Tour — SF (May 9–10, 2026)**.
Track: **Set the Table** (bimanual SO-101, fine manipulation).

## Architecture

Hierarchical: a vision-language-model **planner** picks among trained
per-skill VLAs; each VLA is a SmolVLA fine-tune for one discrete pick action.

```
┌─────────────────────────────────────┐
│   GEMMA-3 27B  (on Spark via         │
│   Ollama, public via bore tunnel +   │
│   bearer-token auth proxy)           │
│                                      │
│   SELECT  next skill                 │
│   VERIFY  state via frame compare    │
│   WATCHER concurrent scene reads     │
└─────────────────────┬───────────────┘
                      │ skill name
                      ▼
┌─────────────────────────────────────┐
│   orchestrator/orchestrator.py       │
│   build_vla_command(skill, meta)     │
│   resolves vla.uri from skills.yaml  │
└─────────────────────┬───────────────┘
                      │ subprocess
                      ▼
            ┌─────────────────┐
            │  lerobot-record  │
            │  --policy.path=  │
            │  <HF repo>       │
            └────────┬────────┘
                     ▼
              [SO-101 arm]
```

## Trained policies

All three skills shipped to HF Hub.

| Skill | Checkpoint | Dataset (mirrored, v3.0-tagged) |
|---|---|---|
| `pick_cup` | [xlerobot-pick-cup-smolvla](https://huggingface.co/Globalmysterysnailrevolution/xlerobot-pick-cup-smolvla) | [xlerobot-pick-cup-20ep](https://huggingface.co/datasets/Globalmysterysnailrevolution/xlerobot-pick-cup-20ep) |
| `pick_cutlery` | [xlerobot-pick-cutlery-smolvla](https://huggingface.co/Globalmysterysnailrevolution/xlerobot-pick-cutlery-smolvla) | [xlerobot-pick-cutlery-20ep](https://huggingface.co/datasets/Globalmysterysnailrevolution/xlerobot-pick-cutlery-20ep) |
| `pick_bowl` | [xlerobot-pick-bowl-smolvla](https://huggingface.co/Globalmysterysnailrevolution/xlerobot-pick-bowl-smolvla) | [xlerobot-pick-bowl-20ep](https://huggingface.co/datasets/Globalmysterysnailrevolution/xlerobot-pick-bowl-20ep) |

Trained on **NVIDIA DGX Spark (GB10)** via `lerobot-train`, 5000 steps,
batch=4, full fine-tune of `lerobot/smolvla_base`. ~24 min per skill on a
single GPU. Datasets sourced from `Mattie-NT/makermods_pick_*_20ep_1` and
mirrored to our HF account with the `v3.0` git tag (lerobot strict-checks
the codebase_version tag; the originals didn't have it).

## Quick start

**Run a single trained skill on the arm directly:**

```bash
lerobot-record \
  --robot.type=so101_follower --robot.port=COM10 \
  --teleop.type=so101_leader  --teleop.port=COM7 \
  --policy.path=Globalmysterysnailrevolution/xlerobot-pick-cup-smolvla \
  --dataset.repo_id=local-eval --dataset.num_episodes=1
```

**Run the full Gemma-orchestrated loop ("set the table"):**

```bash
export OLLAMA_HOST=http://bore.pub:<port>          # see deploy_gemma_proxy.sh output
export GEMMA_PROXY_TOKEN=<token>                   # ditto
python orchestrator/orchestrator.py
```

**Hardware UI for teleop / calibration / recording** (Windows tablet):

```bash
cd MakerMods-App && python -m backend.main         # backend on :8000
cd MakerMods-App/frontend && npm run dev           # UI on :3000
```

## Where things live

```
skills/skills.yaml              — canonical skill registry. Each skill
                                  maps to a vla.uri (HF checkpoint),
                                  vla.dataset_uri (training data),
                                  preconditions, postconditions.

orchestrator/                   — Gemma planner + per-skill VLA dispatcher
  orchestrator.py               — main loop (SELECT → INVOKE → VERIFY)
  watcher.py                    — concurrent scene watcher (pipelined planning)
  ask.py                        — one-shot Gemma query w/ image
  chat.py                       — interactive Gemma REPL
  smoke_test.py                 — exercises every code path, no arm needed

scripts/
  mm.py / mm.cmd                — wrap MakerMods-App REST API from CLI
                                  (ports, cameras, calibrate auto+manual,
                                  record, train, hf, processes, locks)
  kite.py / kite.cmd            — wrap KiteML API (datasets, training,
                                  artifacts, api-keys)
  probe_ports.py                — read-only SO-101 motor scan per port
  reassign_motor_ids.py         — in-place feetech ID rewrite
  record_skill.sh               — lerobot-record wrapper, per-skill HF push
  train_skill.sh                — lerobot-train wrapper (smolvla/act/molmoact2)
  deploy_orchestrator.sh        — ssh-tunnel Gemma on Spark + start loop
  deploy_gemma_proxy.sh         — bearer-auth proxy for Gemma (multi-user)
  deploy_bore_tunnel.sh         — public URL for the proxy via bore.pub
  rotate_gemma_token.sh
  spark/gemma_proxy.py          — runs on Spark, restricts API to one model

MakerMods-App/                  — submodule (LamaSu fork w/ Windows fixes)
lerobot-MakerMods/              — submodule (upstream Maker-Mods)

wiki/                           — mkdocs/zensical site (training plan,
                                  data collection plan)
```

## Open-source contributions made along the way

- **`Maker-Mods/MakerMods-App` PR #1** — Windows port-scanner + camera-scanner
  fixes (UI returned `[]` for both on Windows because the scanners only
  globbed `/dev` paths and the API passed positional args wrong).
  Branch: [`fix/windows-port-scanner`](https://github.com/LamaSu/MakerMods-App/tree/fix/windows-port-scanner).

## Status

| Component | Status |
|---|---|
| Calibration (CLI via `mm calibrate auto/manual`) | ✓ |
| Datasets (3 skills, mirrored + tagged on HF) | ✓ |
| 3 per-skill SmolVLA policies trained + on HF | ✓ |
| `skills.yaml` wired with each `vla.uri` | ✓ |
| Gemma proxy + bore public URL (token-gated) | ✓ |
| Orchestrator (SELECT/INVOKE/VERIFY/WATCHER) | ✓ |
| End-to-end on real arms | ⏳ |
| Demo video / submission writeup | ⏳ |

## What didn't make it (and why)

- **KiteML training**: API works, datasets imported correctly after we
  mirrored+tagged them — but their training container hits a
  `torchcodec_ns::_convert_to_tensor` empty-tensor crash before any
  training step. Pivoted to Spark with our own torchcodec install.
- **Bimanual**: only one SO-101 arm pair was reliably working through
  setup; trained single-arm pick skills (sufficient for the 3 demo objects).
- **MolmoAct2**: scaffolded as a backend (`orchestrator/molmoact2_runner.py`)
  but never wired through — SmolVLA was enough for the time budget.
