"""Pipecat pipeline factory for the voice frontend.

Wires the call audio path::

    Twilio media stream  ->  Deepgram STT  ->  Gemma 4 (Ollama / proxy, tool-use)
                                                       |
                                                       +-> VoiceTools
                                                       |     - look_at_scene  -> orchestrator.capture_frame + Gemma describe
                                                       |     - list_skills    -> orchestrator.load_skills
                                                       |     - propose_skill  -> orchestrator.select_next_skill
                                                       |     - execute_skill  -> orchestrator.execute_skill_with_verification
                                                       |                         (runs lerobot-record on the trained VLA)
                                                       |     - queue_skills / get_status / cancel_current
                                                       |
                                              ElevenLabs TTS  ->  Twilio out

Same OpenAI-compatible LLM endpoint the orchestrator uses (Ollama on
Spark via the bearer-token proxy). We point Pipecat's
:class:`OpenAILLMService` at ``OLLAMA_BASE_URL`` and let Ollama do the
function-calling.

Pipecat is imported lazily inside :func:`build_pipeline` so unit tests
can import ``voice.agent`` without pulling the whole pipecat tree.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

from loguru import logger

from .config import VoiceSettings, get_settings
from .tools import VoiceTools

if TYPE_CHECKING:  # pragma: no cover
    from pipecat.pipeline.pipeline import Pipeline


# ── System prompt ─────────────────────────────────────────────────────────
SYSTEM_PROMPT_TEMPLATE = """You are the voice operator for an xlerobot. You speak with the caller over the phone and drive a bimanual SO-101 arm by picking the right per-skill VLA neural policy for what they want done.

# How execution works

Each "skill" in our library is a separately trained SmolVLA checkpoint. When you call `execute_skill(skill="pick_cutlery")` the orchestrator runs `lerobot-record --policy.path=<HF repo>` and watches the camera while it runs — Gemma classifies in_progress / completed / problem every few seconds and kills the policy when it's done. You do NOT control joints or motors directly. You pick the skill; the trained policy does the motion.

# Tools

1. `look_at_scene(focus?)` — capture the live camera frame and have me describe what's on the table. Always call this before naming a target or picking a skill — the scene changes.

2. `list_skills()` — full catalog from skills.yaml. Each entry has a description, preconditions, postconditions, and the VLA checkpoint URI. Cache it; it's stable for the call.

3. `propose_skill(goal?)` — let the planner (same Gemma, separate selection prompt) recommend the next skill based on the live scene + history. Use this when the caller is vague ("set the table") rather than asking for a specific object.

4. `execute_skill(skill="<name>")` — run ONE skill end-to-end. Returns `{"verdict": "completed"|"problem"|"timeout", "reason": ...}`. The orchestrator handles the subprocess + camera-based verification.

5. `queue_skills(steps=[{"skill": "..."}, ...])` — run a list in order. Bails on the first non-completed verdict. Use this when the caller says "do all three" or describes a sequence.

6. `get_status()` — last action's verdict + recent history. Cheap. Use it between steps without re-looking.

7. `cancel_current()` — best-effort interrupt for queued steps. A skill in flight runs until completed/problem/timeout (the orchestrator's SKILL_TIMEOUT_S is the upper bound).

# Conversation arc

1. Greet briefly. "Hi, I'm the robot. What would you like me to do?"
2. Hear the request. Call `look_at_scene` so you're answering from the real frame, not what you imagine.
3. If the request is vague (e.g. "set the table"), call `propose_skill` and read out what the planner picked + why.
4. If the request is specific ("pick up the bowl"), check it against `list_skills` and pick the matching skill name yourself.
5. {confirm_clause}
6. Call `execute_skill`. While the VLA runs the orchestrator narrates verdicts to you via the tool return. Read them aloud as they happen.
7. Report the outcome. Success = "Got it." Problem = describe what went wrong from the verdict reason.
8. If more steps are queued, continue. Otherwise ask if there's anything else.

# Voice style

You are SPEAKING. Short sentences. Pauses are fine. Never read URLs / HF repo paths / coordinates aloud. When narrating motion describe what you SEE in the frame, not what you assume happened. If a tool returns `{"error": ...}` acknowledge it briefly and ask the caller how to proceed.

You will NOT have a `pick_cup` skill available even if the caller asks for a cup — the cup checkpoint was trained on mislabeled data and is disabled. Substitute `pick_bowl` if the caller seems to mean a small round dish, or `pick_anything` for the generalist fallback. Tell them honestly that the cup skill isn't ready yet.
"""


def _confirm_clause(confirm: bool) -> str:
    if confirm:
        return (
            "Before calling `execute_skill`, ALWAYS say which skill you chose + why, "
            "then ask the human to confirm. Wait for yes / go / do it. If they say "
            "no, ask what to change."
        )
    return (
        "You may execute without per-step confirmation — the caller has "
        "pre-authorised this session. Still narrate clearly so they can "
        "interrupt with `cancel_current` if needed."
    )


def build_system_prompt(settings: Optional[VoiceSettings] = None) -> str:
    """Render the system prompt for the current settings.

    Uses ``str.replace`` (not ``str.format``) because the template has
    literal JSON-y braces ``{"error": ...}`` that ``.format()`` would
    parse as field placeholders.
    """
    settings = settings or get_settings()
    return SYSTEM_PROMPT_TEMPLATE.replace(
        "{confirm_clause}", _confirm_clause(settings.confirm_before_execute)
    )


# ── Tool schemas (OpenAI function-call format) ────────────────────────────


def build_openai_tool_schemas() -> list[dict[str, Any]]:
    """OpenAI-format tool list registered with the LLM service."""
    return [
        {
            "type": "function",
            "function": {
                "name": "look_at_scene",
                "description": "Capture the camera and have Gemma describe what's on the table. Call before naming a target or picking a skill.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "focus": {"type": "string", "description": "Optional hint, e.g. 'the bowl'."}
                    },
                    "required": [],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "list_skills",
                "description": "Return the catalog of trained skills (name, description, preconditions, postconditions, VLA repo).",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "propose_skill",
                "description": "Have the planner pick the next skill from the live scene + history. Use for vague requests like 'set the table'.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "goal": {"type": "string", "description": "High-level goal. Defaults to 'set the table'."}
                    },
                    "required": [],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "execute_skill",
                "description": "Run one trained skill end-to-end via lerobot-record. Returns the orchestrator's verdict (completed/problem/timeout).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "skill": {
                            "type": "string",
                            "description": "Exact skill name from list_skills(), e.g. 'pick_cutlery', 'pick_bowl', 'pick_anything'.",
                        }
                    },
                    "required": ["skill"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "queue_skills",
                "description": "Run a list of skills in order. Bails on first non-completed verdict.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "steps": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {"skill": {"type": "string"}},
                                "required": ["skill"],
                            },
                        }
                    },
                    "required": ["steps"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_status",
                "description": "Last action outcome + recent history. No side effects.",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "cancel_current",
                "description": "Best-effort interrupt for queued steps. A skill in flight runs to completion/timeout.",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        },
    ]


def _make_tool_dispatchers(tools: VoiceTools) -> dict[str, Any]:
    """Map tool names to async callables matching Pipecat's tool-callback signature."""

    async def _look(args: dict[str, Any]) -> dict[str, Any]:
        return await tools.look_at_scene(focus=args.get("focus", ""))

    async def _list(args: dict[str, Any]) -> dict[str, Any]:
        return await tools.list_skills()

    async def _propose(args: dict[str, Any]) -> dict[str, Any]:
        return await tools.propose_skill(goal=args.get("goal", ""))

    async def _exec(args: dict[str, Any]) -> dict[str, Any]:
        return await tools.execute_skill(skill=args["skill"])

    async def _queue(args: dict[str, Any]) -> dict[str, Any]:
        return await tools.queue_skills(steps=args["steps"])

    async def _status(args: dict[str, Any]) -> dict[str, Any]:
        return await tools.get_status()

    async def _cancel(args: dict[str, Any]) -> dict[str, Any]:
        return await tools.cancel_current()

    return {
        "look_at_scene": _look,
        "list_skills": _list,
        "propose_skill": _propose,
        "execute_skill": _exec,
        "queue_skills": _queue,
        "get_status": _status,
        "cancel_current": _cancel,
    }


def build_pipeline(
    *,
    websocket: Any,
    stream_sid: str,
    tools: VoiceTools,
    settings: Optional[VoiceSettings] = None,
) -> "Pipeline":
    """Build a Pipecat pipeline for one Twilio call."""
    settings = settings or get_settings()

    from pipecat.pipeline.pipeline import Pipeline
    from pipecat.processors.aggregators.openai_llm_context import (  # type: ignore[import-not-found]
        OpenAILLMContext,
    )
    from pipecat.serializers.twilio import TwilioFrameSerializer
    from pipecat.services.deepgram.stt import DeepgramSTTService
    from pipecat.services.elevenlabs.tts import ElevenLabsTTSService
    from pipecat.services.openai.llm import OpenAILLMService
    from pipecat.transports.network.fastapi_websocket import (
        FastAPIWebsocketParams,
        FastAPIWebsocketTransport,
    )

    transport = FastAPIWebsocketTransport(
        websocket=websocket,
        params=FastAPIWebsocketParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            add_wav_header=False,
            vad_enabled=True,
            serializer=TwilioFrameSerializer(stream_sid=stream_sid),
        ),
    )

    stt = DeepgramSTTService(api_key=settings.deepgram_api_key)

    llm = OpenAILLMService(
        api_key=settings.ollama_api_key,
        model=settings.ollama_model,
        base_url=settings.ollama_base_url,
    )

    dispatchers = _make_tool_dispatchers(tools)
    for name, callback in dispatchers.items():
        try:
            llm.register_function(name, callback)  # type: ignore[attr-defined]
        except AttributeError:  # pragma: no cover
            logger.error(
                "OpenAILLMService.register_function not found — Pipecat API changed; "
                "patch agent.py at the registration loop."
            )
            raise

    tts = ElevenLabsTTSService(
        api_key=settings.elevenlabs_api_key,
        voice_id=settings.elevenlabs_voice_id,
        model=settings.elevenlabs_model,
    )

    context = OpenAILLMContext(
        messages=[{"role": "system", "content": build_system_prompt(settings)}],
        tools=build_openai_tool_schemas(),
    )
    context_aggregator = llm.create_context_aggregator(context)

    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            context_aggregator.user(),
            llm,
            tts,
            transport.output(),
            context_aggregator.assistant(),
        ]
    )
    return pipeline


__all__ = ["build_pipeline", "build_system_prompt", "build_openai_tool_schemas"]
