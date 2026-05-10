"""xlerobot-voice: phone-call voice frontend for the xlerobot orchestrator.

The caller dials a Twilio number; Pipecat bridges the audio into a
text-only loop with Gemma 4 over the existing gemma-proxy on Spark.
Gemma sees the camera (via the orchestrator's ``capture_frame``), picks
a trained per-skill VLA from ``skills/skills.yaml``, and dispatches
``lerobot-record`` via the orchestrator's ``execute_skill_with_verification``
function. No new LLM, no new VLA stack — just voice IO + tool dispatch
on top of what's already wired.

Sister package to ``orchestrator/``; reuses its functions wholesale.

Import-time side effect: prepends the repo root to ``sys.path`` so the
sibling ``orchestrator`` package is importable regardless of where the
server was launched. The voice/ directory holds pyproject.toml + its
own __init__.py (flat layout), so the parent of __file__ IS the repo
root that contains both ``voice/`` and ``orchestrator/``.
"""

from __future__ import annotations

import pathlib
import sys

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

__version__ = "0.1.0"
