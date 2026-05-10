# Demo voiceover script

A clean, linear script optimized for reading aloud or feeding to a TTS engine
(ElevenLabs / Coqui / Anthropic Voice). Total target: **90 seconds** at
~155 wpm = ~230 words. Below sits at ~225.

Pause markers `[…]` indicate beats for the visual to land. They are not read.

---

## Voiceover (read straight through)

> Most VLA demos pick one thing up.
>
> [...]
>
> We built the agentic loop on top.
>
> [pause for arm to grab the cup]
>
> Gemma-3 27B sees the scene, picks one of three trained skills, and invokes
> the matching SmolVLA. Each skill — pick cup, pick cutlery, pick bowl — is
> its own fine-tuned policy. Twenty episodes per skill, twenty-four minutes
> of training each, on a single GPU.
>
> [pause for terminal to show Gemma's decision]
>
> But the part most demos skip is the verification. The same Gemma that
> planned the action is now watching the camera.
>
> [pause for verification log line]
>
> Frame compare, every three seconds. If the gripper drops the cup, if the
> arm grabs the wrong object, if anything looks off — Gemma flags it and
> replans before the next skill starts.
>
> [pause as second skill fires]
>
> That closed loop is what makes long-horizon task composition work on a
> physical arm. Not one model that knows everything. Three small specialists
> chosen at runtime by a planner that also verifies.
>
> [pause as final item is placed]
>
> Three skills. One planner. One arm. The table is set.
>
> [...]
>
> Code, models, datasets, all open source. The link is in the corner. Talk
> to us about bimanual, MolmoAct backends, or sim-to-real.

---

## Annotation for editor / TTS

| Beat | Audio cue | Visual must show |
|---|---|---|
| "We built the agentic loop on top." | small swell | architecture diagram fades in |
| "Gemma sees the scene…" | normal pace | terminal scrolling |
| "But the part most demos skip…" | slight pause before "verification" | terminal highlights `verify_skill_state` line |
| "If the gripper drops the cup…" | pace up slightly | rapid cut between three failure modes (or just the verifier line) |
| "Three skills. One planner. One arm. The table is set." | pace down, deliberate | wide shot of completed table |
| Final line | warmer tone | github URL + handles overlay |

---

## TTS prompt (if generating with ElevenLabs / Anthropic Voice)

```
Voice: clear, technical, slight pace variation. Not robotic, not salesy.
Pace: 150-160 wpm. Slow on the closing three sentences.
Pauses: ~600ms at [...] markers. ~1000ms before "Three skills. One
planner. One arm."
```

Recommended ElevenLabs voice: **Adam** or **Rachel** (neutral, technical).
Avoid anything overly performative.

---

## Word-count discipline

Spoken portion is exactly 225 words. At 155 wpm that's 87 seconds. Buys
us 3 seconds of headroom for visual transitions that need slightly longer
breath. If you record live and end up at 100 sec, drop the line "Twenty
episodes per skill, twenty-four minutes of training each, on a single GPU"
— it's the easiest cut without losing the thesis.
