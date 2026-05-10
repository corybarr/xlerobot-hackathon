"""xlerobot-voice: phone-call voice frontend for the xlerobot orchestrator.

Loads ``voice/.env`` at import time so submodules can read os.environ
directly (tools.py and config.py both rely on this — pydantic-settings
would handle it for the latter, but tools.py reads raw env at module
load to wire up the Gemma proxy URL, ROS ports, etc).
"""

from __future__ import annotations

import pathlib
from dotenv import load_dotenv

_ENV_FILE = pathlib.Path(__file__).resolve().parent / ".env"
if _ENV_FILE.is_file():
    load_dotenv(_ENV_FILE)

"""

The caller dials a Twilio number; Pipecat bridges the audio into a
text-only loop with Gemma 4 over the existing gemma-proxy on Spark.
Gemma sees the camera (via the orchestrator's ``capture_frame``), picks
a trained per-skill VLA from ``skills/skills.yaml``, and dispatches
``lerobot-record`` via the orchestrator's ``execute_skill_with_verification``
function. No new LLM, no new VLA stack — just voice IO + tool dispatch
on top of what's already wired.

"""

__version__ = "0.1.0"
