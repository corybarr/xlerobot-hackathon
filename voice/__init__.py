"""xlerobot-voice: phone-call voice frontend for the xlerobot orchestrator.

The caller dials a Twilio number; Pipecat bridges the audio into a
text-only loop with Gemma 4 over the existing gemma-proxy on Spark.
Gemma sees the camera (via the orchestrator's ``capture_frame``), picks
a trained per-skill VLA from ``skills/skills.yaml``, and dispatches
``lerobot-record`` via the orchestrator's ``execute_skill_with_verification``
function. No new LLM, no new VLA stack — just voice IO + tool dispatch
on top of what's already wired.

Sister package to ``orchestrator/``; reuses its functions wholesale.
"""

__version__ = "0.1.0"
