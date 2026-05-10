# Demo pitch — 5 slides + speaker script

For judging at the Physical AI Hack World Tour SF, May 10 4-6 PM. Target
length: 90 seconds spoken, max 3 min if Q&A counts as part of the slot.

---

## Slide 1 — Thesis (10 sec)

> **Long-horizon agentic task composition for physical AI.**
> Three SmolVLAs. One Gemma planner. One SO-101 arm. Set the table.

Visual: photo of the arm completing the task, paused on the moment it
places the third item. Below: `github.com/corybarr/xlerobot-hackathon`.

Spoken: "Most VLA demos pick one thing up. We built the agentic loop on top
— so the robot can decide what to do next, do it, check if it worked, and
keep going until the task is done."

---

## Slide 2 — Architecture (15 sec)

ASCII diagram (or rebuild as a clean visual):

```
   GEMMA-3 27B                ┌─ pick_cup    SmolVLA  ──┐
   on Spark         ──→       ├─ pick_cutlery SmolVLA  ─┼──→  SO-101 arm
   (planner +                  └─ pick_bowl   SmolVLA  ──┘
    verifier)         ↑___________________________________│
                          frame-compare verification
                          (same Gemma, every 3s)
```

Spoken: "Gemma sees the scene, picks one of three trained skills, invokes
the matching VLA, then watches the camera to verify. If anything looks
wrong — dropped item, missed grab — it replans. The verification loop is
the closed-loop part most demos skip."

---

## Slide 3 — Live demo (45 sec)

Switch to the live arm or pre-recorded video. Talking points to land
during the demo:

| When you see | Say |
|---|---|
| Arm picks the cup | "First skill — `pick_cup` — running on a SmolVLA fine-tuned on 30 episodes." |
| Terminal shows Gemma's pick | "Gemma chose this skill from a list it's never seen during training. It reads the YAML, sees the scene, and decides." |
| Verification fires | "Frame compare. Gemma asks itself 'did that work?' before letting the next skill start." |
| Final item placed | "Three skills, composed by the planner, with verification between each. That's the agentic loop on a robot arm." |

If the live arm fails: cut to backup video. Don't apologize — say "the
beauty of having three independent specialists is that one bad take
doesn't kill the demo. Gemma replans." Then show the backup.

---

## Slide 4 — How we built it (15 sec)

Stack list, two-column:

| Open-source we used | What we contributed back |
|---|---|
| LeRobot (HF) — SmolVLA + lerobot-train | PR #1 to MakerMods-App: Windows port-scanner + camera-scanner fixes |
| MakerMods-App — recording + calibration UI | `mm` CLI: full REST wrapper for the UI (calibrate auto+manual via WebSocket) |
| KiteML (sponsor) — tried for training | `kite` CLI: full API wrapper (datasets, training, artifacts) |
| Ollama + Gemma 3 27B on DGX Spark | `gemma-proxy` + `bore` tunnel: bearer-token shared LLM access for the team without ssh handout |
| Hugging Face Hub | 3 datasets + 3 trained checkpoints all public |

Spoken: "Most of our time was plumbing. Every fix we hit, we pushed back —
upstream PR to MakerMods, two CLIs that wrap the whole stack from terminal,
and a proxy so the team can share one Gemma instance without sharing
credentials. Next year's hackathon teams start 24 hours ahead of us."

---

## Slide 5 — What's next + ask (5 sec)

> **Trained policies, public datasets, all open-source.**
>
> Repo: github.com/corybarr/xlerobot-hackathon
> Models: huggingface.co/Globalmysterysnailrevolution
>
> Talk to us about: bimanual extension, MolmoAct2 backend, sim-to-real
> for new objects.

Spoken: "Code, models, datasets — all public. We've scaffolded a MolmoAct2
backend and a bimanual path; happy to talk to anyone working on either."

---

## Backup pocket cards

Keep these ready in case Q&A goes long.

**Q: Why three small VLAs instead of one big one?**
A: Two reasons. First, a small specialist trained on 30 episodes converges
in 24 minutes on a single GPU. A multi-task model needs all skills
proportionally represented and longer training. Second, the planner
externalizes the task identification a multi-task model does internally —
which means it's debuggable. We can see *why* Gemma picked a skill.

**Q: How does the verification work?**
A: Gemma takes the pre-action frame, the current frame, the skill's
preconditions and postconditions from skills.yaml, and outputs
`{state: in_progress | completed | problem, reason: ...}`. If `problem`,
the orchestrator aborts the current skill and asks Gemma to replan from
the new scene. Runs every 3 seconds during execution.

**Q: What if Gemma picks the wrong skill?**
A: The verifier catches it — postconditions for `pick_cup` won't be
satisfied if the arm grabbed cutlery. State flips to `problem`, planner
gets called again with the failure as context. The system self-corrects
within one or two replan cycles.

**Q: How long did training take?**
A: 24 minutes per skill on the GB10 (Grace+Blackwell) in DGX Spark. Three
skills sequential = 72 minutes. Could be ~24 minutes total in parallel
with three GPUs.

**Q: What didn't work?**
A: KiteML's training container hits a torchcodec bug — confirmed with
their team, not on our side. We pivoted to Spark with our own torchcodec
install. Bimanual was blocked by hardware: only one arm pair calibrated
cleanly in time. MolmoAct2 backend scaffolded but not wired through.

---

## Pre-demo checklist

Run through this 30 min before the demo slot:

- [ ] Gemma proxy + bore tunnel up on Spark (`scripts/deploy_gemma_proxy.sh`)
- [ ] `~/.gemma_token` present and matches deployed proxy
- [ ] `skills.yaml` `vla.uri` filled for all 3 skills (no `null` values)
- [ ] Both COM ports visible (`mm ports`)
- [ ] Arm calibration files present (`mm calibrate status`)
- [ ] Camera 1 returns a frame (`scripts/probe_ports.py` indirectly verifies)
- [ ] Smoke test passes 7/7 (`python orchestrator/smoke_test.py` with env vars set)
- [ ] Backup video on local disk in case live runs cold
- [ ] Browser tabs open: GitHub repo, HF profile, Mattie's profile
- [ ] Backup tweet drafted (in case live posting fails during demo)
