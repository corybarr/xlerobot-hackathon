# Data collection plan

## Tasks

Canonical skill list lives in [`skills/skills.yaml`](https://github.com/corybarr/xlerobot-hackathon/blob/main/skills/skills.yaml). Add new skills there — `record_skill.sh`, `train_skill.sh`, and the orchestrator all read from that file.

Starter set (extend as needed):

| Skill | Description | Episode time |
|---|---|---|
| `remove_fork_from_cup` | Remove the fork from the cup and lift it clear | 15s |
| `place_fork` | Place the fork at its setting position on the table | 15s |
| `move_cup` | Move the cup to the upper-right of the place setting | 15s |
| `pick_up_plate` | Pick up the plate from the stack | 20s |
| `place_plate` | Place the plate at the center of the place setting | 20s |

## Episodes per task

`record_skill.sh` defaults to **50** episodes per skill. Pass a different count as the second arg:

```bash
./scripts/record_skill.sh remove_fork_from_cup 80
```

## Operators

Whoever's teleoperating. **Don't double-record the same skill** — two people writing to the same HF dataset repo will clobber each other. Suggested split: one operator per skill until first round done, then rebalance.

## Recording rig

- **UI**: MakerMods-App at `http://localhost:3000` (the wizard) — or run `record_skill.sh` directly from CLI for repeatability.
- **Cameras** (3 total, matching lerobot-MakerMods convention):
  - `hand_cam` — wrist-mounted, index 0
  - `front_cam` — front-facing scene view, index 1
  - `side_cam` — side angle, index 2
  - 640x480 at 30fps
- **Arm**: single SO-101 pair for v1 (one leader teleop'ing one follower). Bimanual once a second pair is calibrated.
- **State schema**: 6-dim joint positions (12-dim if bimanual). SmolVLA pads to 32-dim internally.

## Storage

Datasets push to HF Hub at `<HF_USER>/xlerobot-<skill>`. Default `HF_USER` is `Globalmysterysnailrevolution`; override via env var to push under a team org instead.

Backups: HF Hub is the source of truth — no local-only datasets. If a recording session ends without a successful push, retry before recording more.

## Validation

After recording each skill, replay one episode to sanity-check:

```bash
lerobot-replay --dataset.repo_id=<HF_USER>/xlerobot-<skill> --episodes 0
```

The arm should reproduce the recorded motion. If it doesn't, the calibration is off (or the recording is corrupt) — re-record before sinking more time. Don't train on broken data.

## Schedule (target)

| When | What |
|---|---|
| Sat afternoon | Calibrate arms; first 50 episodes of `remove_fork_from_cup` + `place_fork` |
| Sat evening | Remaining 3 starter skills (`move_cup`, `pick_up_plate`, `place_plate`) |
| Sat night | Training kicks off in parallel on Spark |
| Sun morning | Extend skill list based on whatever Gemma actually needs to call to finish "set the table" |
| Sun early afternoon | Integration test |
| **Sun 4 PM** | **Submission deadline** |
