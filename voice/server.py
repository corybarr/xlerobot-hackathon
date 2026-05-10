"""FastAPI server — webhook target for Vapi's tool calls.

Endpoints
---------

* ``GET  /health``         — Liveness + runtime snapshot.
* ``GET  /api/agent.json`` — Full agent contract: system prompt + Vapi-format
                              tool schemas + skill catalog. Used by the
                              ``vapi-bootstrap.py`` script that creates the
                              Vapi Assistant via API.
* ``GET  /api/skills``     — Skill catalog from ``skills/skills.yaml``.
* ``POST /api/tools/{name}`` — Vapi tool webhook. Vapi POSTs a tool call here;
                              we run the matching :class:`VoiceTools` method
                              and return the JSON result. This is where the
                              VLA execution starts.

Vapi tool webhooks
------------------

Vapi posts tool calls in this shape (one OR many in ``message.toolCallList``)::

    {
      "message": {
        "type": "tool-calls",
        "toolCallList": [
          {"id": "call_123", "name": "pick_cup", "arguments": {...}}
        ]
      }
    }

We accept BOTH shapes:

1. **Vapi native** — POST ``/api/tools`` with the full ``message`` payload;
   we dispatch each ``toolCallList`` item to the matching method.
2. **Direct** — POST ``/api/tools/<tool_name>`` with just the kwargs;
   used by ``curl`` smoke-tests and any non-Vapi caller.

Both return a ``results`` array Vapi unpacks into the LLM context.

Process model: ONE process, one arm, one camera. ``VoiceTools`` serialises
concurrent skill executions through a lock.
"""

from __future__ import annotations

import json
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, PlainTextResponse
from loguru import logger

from .config import VoiceSettings, get_settings
from .tools import VoiceTools


# Singleton — first call builds, subsequent return the same instance.
_tools: Optional[VoiceTools] = None


def _get_tools() -> VoiceTools:
    global _tools
    if _tools is None:
        _tools = VoiceTools()
        logger.info("VoiceTools constructed (skills={})", len(_tools._skills))  # noqa: SLF001
    return _tools


# ── System prompt for Gemma 4 (what Vapi sends as the assistant's role) ──
SYSTEM_PROMPT = """You are the voice operator for an SO-101 bimanual robot arm. You speak with the caller over the phone and drive the arm by calling the right trained Vision-Language-Action (VLA) policy for what they want done.

# How execution works

Each "skill" is a separately trained SmolVLA checkpoint. When you call `pick_cup` (or any other `pick_*` tool), the server spawns `lerobot-record` with the matching HuggingFace policy repo. The trained policy drives the arm. You don't move joints directly — you pick the skill, the policy does the motion.

The tools return a `job_id` instantly. Call `get_status(job_id)` every 3-5 seconds to read the latest progress aloud, until the state is `completed`, `error`, `timeout`, or `cancelled`.

# Tools

1. `look_at_scene(focus?)` — describe what the vision camera currently sees. Always call before naming a target — the scene changes.

2. `list_skills()` — full catalog of trained skills (cup, bowl, cutlery, plus a generalist `pick_anything`). Cache it; stable for the call.

3. `pick_cup()` / `pick_bowl()` / `pick_cutlery()` / `pick_anything()` — run one trained VLA. Returns a `job_id` instantly.

4. `run_skill(skill="...")` — same as the per-skill tools but takes a name; use when the human asks for something by name.

5. `get_status(job_id?)` — latest status of a specific job (or the most recent if no id). State is one of: starting / running / completed / error / timeout / cancelled. Read the `message` field aloud.

6. `cancel_current()` — kill any running VLA subprocess. Use when the caller says "stop".

# Conversation arc

1. Greet briefly: "Hi, this is the robot. What would you like me to pick up?"
2. Hear the request. If specific ("pick up the cup"), call `look_at_scene` to confirm it's there, then call `pick_cup()`.
3. If vague ("set the table"), call `look_at_scene`, then pick a skill yourself based on what you see — read your reasoning aloud first.
4. After calling a `pick_*` tool, you get back `{"job_id": "...", ...}`. Tell the caller "I'm running the cup pick now" and start polling `get_status(job_id=...)` every ~4 seconds.
5. For each status reply, read the `message` field aloud if it changed meaningfully (skip duplicates).
6. When state is `completed`, say so. If `error` / `timeout` / `cancelled`, read the message reason.
7. Ask if there's anything else.

# Voice style

Speak short. Never read URLs / HF repo IDs / coordinates aloud. If a tool returns `{"error": ...}`, acknowledge briefly and ask the caller how to proceed.

# Things you don't have

- No joint-level control. Only the trained skill tools.
- No bimanual coordination yet — the trained policies are single-arm.
- No `pick_fork` / `pick_spoon` specifically — use `pick_cutlery`.
"""


def create_app(settings: Optional[VoiceSettings] = None) -> FastAPI:
    settings = settings or get_settings()

    app = FastAPI(
        title="xlerobot-voice",
        version="0.1.0",
        description=(
            "Vapi tool webhook + agent discovery for the xlerobot-hackathon "
            "VLA skills. Each skill is one trained SmolVLA checkpoint run "
            "via lerobot-record."
        ),
    )

    # ── Health ────────────────────────────────────────────────────────
    @app.get("/health")
    async def health() -> dict[str, object]:
        tools = _get_tools()
        return {
            "status": "ok",
            "service": "xlerobot-voice",
            "version": "0.1.0",
            "skills": list(tools._skills.keys()),  # noqa: SLF001
            "jobs_total": len(tools._jobs),  # noqa: SLF001
            "webhook_host": settings.webhook_host,
        }

    # ── Agent discovery ──────────────────────────────────────────────
    @app.get("/api/agent.json")
    async def agent_manifest() -> JSONResponse:
        tools = _get_tools()
        skills = (await tools.list_skills())["skills"]
        return JSONResponse({
            "schema_version": "0.2",
            "name": "xlerobot-voice",
            "version": "0.1.0",
            "summary": (
                "Voice frontend for the xlerobot-hackathon Set-the-Table VLAs. "
                "Each skill = one trained SmolVLA checkpoint. Gemma 4 picks "
                "which to call from the live camera view; lerobot-record runs "
                "the chosen policy on the arm."
            ),
            "system_prompt": SYSTEM_PROMPT,
            "tools": _build_vapi_tool_schemas(skills),
            "skills": skills,
            "endpoints": {
                "tool_webhook": "POST /api/tools  (Vapi native) OR /api/tools/<name>  (direct)",
                "discovery": "GET /api/agent.json",
                "catalog": "GET /api/skills",
                "health": "GET /health",
            },
        })

    @app.get("/api/skills")
    async def skills_catalog() -> dict:
        return await _get_tools().list_skills()

    # ── Tool webhook (Vapi native shape) ─────────────────────────────
    @app.post("/api/tools")
    async def vapi_tool_webhook(request: Request) -> JSONResponse:
        """Vapi POSTs tool calls here in various shapes depending on its
        API version. We accept all of them and never return 4xx, because
        Vapi treats a non-200 as 'call broken' and ends the call."""
        try:
            body = await request.json()
        except Exception:
            body = {}

        # Vapi sends ONE of these shapes (observed across API versions):
        #   {"message": {"toolCallList": [...]}}              # current native
        #   {"message": {"toolCalls": [...]}}                 # alt camelCase
        #   {"message": {"tool_calls": [...]}}                # alt snake_case
        #   {"toolCallList": [...]}                           # un-wrapped
        #   {"toolCalls": [...]}                              # un-wrapped alt
        # OpenAI style each item: {"id", "function": {"name", "arguments"}}
        # Older flat:             {"id", "name", "arguments"}
        msg = (body or {}).get("message") or body or {}
        tool_calls = (
            msg.get("toolCallList")
            or msg.get("toolCalls")
            or msg.get("tool_calls")
            or body.get("toolCallList")
            or body.get("toolCalls")
            or body.get("tool_calls")
            or []
        )

        if not isinstance(tool_calls, list) or not tool_calls:
            # Don't 400 — log the raw payload so we can see what Vapi sent,
            # and return an empty results array so the call stays alive.
            logger.warning(
                "Vapi webhook hit /api/tools with no toolCalls extractable; raw body="
                + json.dumps(body)[:1500]
            )
            return JSONResponse({"results": []})

        tools = _get_tools()
        results = []
        for call in tool_calls:
            cid = call.get("id") or call.get("toolCallId") or "(none)"
            fn = call.get("function") or {}
            name = call.get("name") or fn.get("name")
            args = call.get("arguments")
            if args is None:
                args = fn.get("arguments")
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except Exception:
                    args = {}
            if not isinstance(args, dict):
                args = {}
            logger.info("vapi tool call: id={} name={} args={}", cid, name, args)
            res = await _dispatch_one(tools, name, args)
            results.append({"toolCallId": cid, "result": res})

        return JSONResponse({"results": results})

    # ── Direct tool invocation ───────────────────────────────────────
    @app.post("/api/tools/{name}")
    async def direct_invoke(name: str, request: Request) -> JSONResponse:
        try:
            body = await request.json()
        except Exception:
            body = {}
        if not isinstance(body, dict):
            raise HTTPException(status_code=400, detail="body must be a JSON object")
        tools = _get_tools()
        result = await _dispatch_one(tools, name, body)
        if isinstance(result, dict) and result.get("error") in (
            f"unknown_tool: {name!r}",
        ):
            raise HTTPException(status_code=404, detail=result["error"])
        return JSONResponse(result)

    @app.get("/")
    async def root() -> PlainTextResponse:
        return PlainTextResponse(
            "xlerobot-voice. Discovery: GET /api/agent.json | "
            "Vapi webhook: POST /api/tools | Health: GET /health"
        )

    return app


async def _dispatch_one(tools: VoiceTools, name: Optional[str], args: dict[str, Any]) -> Any:
    if not name:
        return {"error": "tool name missing", "where": "_dispatch_one"}
    method = getattr(tools, name, None)
    if method is None or not callable(method) or name.startswith("_"):
        return {"error": f"unknown_tool: {name!r}", "where": "_dispatch_one"}
    try:
        return await method(**args)
    except TypeError as exc:
        return {"error": f"bad_args: {exc!s}", "where": name}
    except Exception as exc:
        logger.warning("tool {} crashed: {}", name, exc)
        return {"error": f"tool_crashed: {exc!s}", "where": name}


def _build_vapi_tool_schemas(skills: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """OpenAI-function-call schemas Vapi accepts in Assistant config."""
    tools: list[dict[str, Any]] = [
        {
            "type": "function",
            "function": {
                "name": "look_at_scene",
                "description": (
                    "Capture the vision camera and have Gemma describe what's there. "
                    "Always call this before naming a target — the scene changes."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "focus": {"type": "string", "description": "Optional hint (e.g. 'the cup')."}
                    },
                    "required": [],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "list_skills",
                "description": "All trained skills from skills.yaml. Stable; cache for the call.",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        },
    ]
    # One tool per skill — Gemma can pick directly by tool name.
    for s in skills:
        tools.append({
            "type": "function",
            "function": {
                "name": s["name"],
                "description": (
                    f"{s['description']} Trained VLA: "
                    f"{(s.get('vla_uri') or 'unknown')}. "
                    f"Preconditions: {s.get('preconditions','—')}. "
                    f"Postconditions: {s.get('postconditions','—')}."
                ),
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        })
    tools.extend([
        {
            "type": "function",
            "function": {
                "name": "run_skill",
                "description": "Run any skill by name. Use when caller asks for something by name.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "skill": {"type": "string", "description": "Exact skill name from list_skills."},
                    },
                    "required": ["skill"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_status",
                "description": (
                    "Latest status of a job (or the most-recent if no id). "
                    "Read the `message` field aloud."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "job_id": {"type": "string", "description": "Job id from a pick_* tool."}
                    },
                    "required": [],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "cancel_current",
                "description": "Kill any running VLA subprocess. Use when caller says 'stop'.",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        },
    ])
    return tools


def main() -> None:
    import uvicorn

    settings = get_settings()
    logger.info(
        "xlerobot-voice starting | webhook={} | port={}",
        settings.webhook_host,
        settings.server_port,
    )
    app_instance = create_app(settings)
    uvicorn.run(
        app_instance,
        host=settings.server_host,
        port=settings.server_port,
        log_level=settings.log_level.lower(),
    )


app: Optional[FastAPI] = None

__all__ = ["create_app", "main", "app", "SYSTEM_PROMPT"]


# Enable `python -m voice.server` (was missing — server "ran" but only
# imported the module, leaving main() uncalled, which produced silent
# zero-output exits every time we tried to launch detached.)
if __name__ == "__main__":
    main()
