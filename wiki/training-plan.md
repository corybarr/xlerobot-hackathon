# Training plan

## Model strategy: hierarchical Gemma + per-skill VLAs

**High-level planner** — Gemma 4 vision-LM (Ollama on Spark, port-forwarded to
the tablet at runtime). Watches the front camera, recalls history, picks the
next discrete skill to invoke. Replans after every action.

**Low-level skills** — one fine-tuned policy per discrete skill in
`skills/skills.yaml`. We A/B three VLA architectures by training each skill
with each VLA and shipping whichever wins per skill:

| VLA | Size | Notes |
|---|---|---|
| **SmolVLA** | ~450M | Primary. lerobot-native, bimanual built-in, fast inference. |
| **ACT** | small | Action chunking transformer. Fastest to train, often best on simple repetitive skills. |
| **MolmoAct2** | 5B | Strongest per-skill reasoning. `MolmoAct2-SO100_101` is single-arm only — bimanual would need its own fine-tune. May not fit deadline. |

## Data

- Demos collected via MakerMods-App teleop on SO-101 arm pair (single-arm v1; bimanual once a second pair is calibrated).
- Per-skill datasets at HF Hub: `<HF_USER>/xlerobot-<skill>`.
- 3 cameras: front, hand, side at 30fps 640x480 (matches lerobot-MakerMods convention).
- Canonical task list: `skills/skills.yaml` — single source of truth, read by `record_skill.sh`, `train_skill.sh`, and the orchestrator.

## Hardware

- **Recording**: Windows tablet (where the arms plug in).
- **Training**: DGX Spark via `spark-run` (113Gi free at session start). Tablet CPU is a no-go for fine-tuning anything bigger than ACT.
- **Planner inference**: Gemma on Spark via Ollama. `scripts/deploy_orchestrator.sh` pulls the model, ensures `ollama serve` is running, opens an SSH port-forward (11434), then starts the orchestrator locally.

## Parallelism

Each team member claims a **(skill, VLA)** combo. Train independently; push checkpoint to `<HF_USER>/xlerobot-<skill>-<vla>`. The orchestrator's `VLA_BACKEND` env var picks which checkpoint family to load at runtime — so one person can ship `smolvla` while another iterates on `act` for the same skill.

Coordination tip: post your claimed combo in the team chat or a `claims.md` so people don't double-up.

## Schedule (target)

| When | What |
|---|---|
| Sat afternoon | Hardware bring-up; first 50 episodes of `remove_fork_from_cup` + `place_fork` |
| Sat evening | Remaining starter skills (move_cup, pick_up_plate, place_plate) |
| Sat night | Training runs kick off in parallel on Spark |
| Sun morning | Per-skill eval → pick winners |
| Sun early afternoon | End-to-end integration test (Gemma + winning VLAs + arm) |
| **Sun 4 PM** | **Submission deadline / demo** |

## Validation

- **Per-skill**: `lerobot-record --policy.path=<HF_USER>/xlerobot-<skill>-<vla> --dataset.num_episodes=1` — visually inspect the arm executes the skill on a real scene.
- **End-to-end**: `GOAL="set the table" ./scripts/deploy_orchestrator.sh` — full pipeline. Succeeds when Gemma issues `{"done": true}` with a sensible reason and the table actually looks set.

## Fallback (if a VLA underperforms on a skill)

In order of cost:

1. **Record more demos** for that specific skill (cheap, fast)
2. **Swap VLA** for that skill — ACT often beats SmolVLA on simple repetitive motions
3. **Hand-tune a waypoint script** for that skill and have Gemma call it as if it were any other VLA — same skill name, different backend
