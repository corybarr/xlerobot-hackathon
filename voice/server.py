"""FastAPI server — Twilio bridge + agent discovery for the voice frontend.

Endpoints
---------

* ``POST /twilio/inbound`` — TwiML <Connect><Stream> directive
* ``WS   /ws``             — Twilio media stream socket (one Pipecat pipeline per call)
* ``GET  /health``         — Liveness + runtime snapshot
* ``GET  /api/agent.json`` — Full agent contract for non-voice integrations
                              (system prompt + OpenAI tool schemas + skills.yaml)
* ``POST /api/tools/{name}`` — Direct HTTP invocation of any VoiceTools method
* ``GET  /api/skills``     — Skill catalog (mirrors `voice.tools.VoiceTools.list_skills`)

Process model: ONE process per machine. The orchestrator's lerobot-record
subprocess + camera capture serialize at the OS level — concurrent calls
would fight over COM10 / the camera. The singleton ``VoiceTools`` holds
the executor lock.
"""

from __future__ import annotations

import json
from typing import Optional

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, PlainTextResponse, Response
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


def create_app(settings: Optional[VoiceSettings] = None) -> FastAPI:
    settings = settings or get_settings()

    app = FastAPI(
        title="xlerobot-voice",
        version="0.1.0",
        description=(
            "Phone-call voice frontend for the xlerobot-hackathon orchestrator. "
            "Pipecat (Twilio + Deepgram + Gemma 4 + ElevenLabs) on top of "
            "skills.yaml + lerobot-record."
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
            "ollama_model": settings.ollama_model,
            "ollama_base_url": settings.ollama_base_url,
            "skills_count": len(tools._skills),  # noqa: SLF001
            "executions_total": len(tools._history),  # noqa: SLF001
            "webhook_host": settings.webhook_host,
            "confirm_before_execute": settings.confirm_before_execute,
        }

    # ── Agent discovery ──────────────────────────────────────────────
    @app.get("/api/agent.json")
    async def agent_manifest() -> JSONResponse:
        """Full contract for other LLM agents (text, MCP, custom) to drive this service.

        Returns system prompt + OpenAI-format tool schemas + the skills
        catalog. Drop the schemas into any chat-completions call's ``tools``
        field; route tool calls to ``POST /api/tools/{name}``.
        """
        from .agent import build_openai_tool_schemas, build_system_prompt

        tools = _get_tools()
        skills_list = await tools.list_skills()
        return JSONResponse(
            {
                "schema_version": "0.1",
                "name": "xlerobot-voice",
                "version": "0.1.0",
                "summary": (
                    "Voice + text frontend for the xlerobot-hackathon "
                    "orchestrator (Set-the-Table track). Drives lerobot-record "
                    "on trained SmolVLA checkpoints via skills.yaml."
                ),
                "system_prompt": build_system_prompt(settings),
                "tools": build_openai_tool_schemas(),
                "skills": skills_list["skills"],
                "endpoints": {
                    "voice_inbound": "POST /twilio/inbound (TwiML)",
                    "voice_ws": "WS /ws (Twilio media stream)",
                    "text_invoke": "POST /api/tools/{name} body={kwargs}",
                    "discovery": "GET /api/agent.json",
                    "catalog": "GET /api/skills",
                    "health": "GET /health",
                },
                "runtime": {
                    "ollama_base_url": settings.ollama_base_url,
                    "ollama_model": settings.ollama_model,
                    "confirm_before_execute": settings.confirm_before_execute,
                },
            }
        )

    @app.get("/api/skills")
    async def skills_catalog() -> dict:
        return await _get_tools().list_skills()

    # ── Direct tool invocation (HTTP, no voice) ──────────────────────
    @app.post("/api/tools/{name}")
    async def invoke_tool(name: str, request: Request) -> JSONResponse:
        try:
            body = await request.json()
        except Exception:
            body = {}
        if not isinstance(body, dict):
            raise HTTPException(status_code=400, detail="body must be a JSON object")

        tools = _get_tools()
        method = getattr(tools, name, None)
        if method is None or not callable(method) or name.startswith("_"):
            raise HTTPException(status_code=404, detail=f"unknown tool: {name}")

        try:
            result = await method(**body)
        except TypeError as exc:
            raise HTTPException(status_code=400, detail=f"bad args: {exc!s}") from exc
        except Exception as exc:
            logger.warning("tool {} crashed: {}", name, exc)
            return JSONResponse(
                {"error": f"tool_crashed: {exc!s}", "where": name}, status_code=200
            )

        return JSONResponse(result)

    # ── Twilio inbound webhook ───────────────────────────────────────
    @app.post("/twilio/inbound")
    async def twilio_inbound(request: Request) -> Response:
        host = settings.webhook_host
        if host.startswith("https://"):
            ws_url = "wss://" + host[len("https://") :] + "/ws"
        elif host.startswith("http://"):
            ws_url = "ws://" + host[len("http://") :] + "/ws"
        else:
            ws_url = "wss://" + host.lstrip("/") + "/ws"

        twiml = (
            '<?xml version="1.0" encoding="UTF-8"?>'
            "<Response>"
            "<Connect>"
            f'<Stream url="{ws_url}" />'
            "</Connect>"
            "</Response>"
        )
        return Response(content=twiml, media_type="application/xml")

    # ── Twilio media stream WebSocket ────────────────────────────────
    @app.websocket("/ws")
    async def twilio_ws(websocket: WebSocket) -> None:
        await websocket.accept()
        stream_sid: Optional[str] = None
        try:
            await websocket.receive_text()           # "connected" frame
            start_msg = await websocket.receive_text()  # "start" frame w/ streamSid
            start_data = json.loads(start_msg)
            stream_sid = start_data.get("start", {}).get("streamSid")
            if not stream_sid:
                logger.warning("No streamSid in start frame; dropping call.")
                await websocket.close(code=1011)
                return
            logger.info("New voice call: streamSid={}", stream_sid)
        except (WebSocketDisconnect, json.JSONDecodeError, KeyError) as exc:
            logger.warning("Twilio handshake failed: {}", exc)
            return

        from pipecat.pipeline.runner import PipelineRunner
        from pipecat.pipeline.task import PipelineTask

        from .agent import build_pipeline

        tools = _get_tools()
        try:
            pipeline = build_pipeline(
                websocket=websocket,
                stream_sid=stream_sid,
                tools=tools,
                settings=settings,
            )
            task = PipelineTask(pipeline)
            runner = PipelineRunner()
            await runner.run(task)
        except Exception as exc:  # pragma: no cover - integration only
            logger.exception("Pipeline crashed: {}", exc)

    @app.get("/")
    async def root() -> PlainTextResponse:
        return PlainTextResponse(
            "xlerobot-voice. Discovery: GET /api/agent.json | "
            "Twilio inbound: POST /twilio/inbound | Health: GET /health"
        )

    return app


def main() -> None:
    """Entry point for the ``xlerobot-voice`` console script."""
    import uvicorn

    settings = get_settings()
    logger.info(
        "xlerobot-voice starting | llm={} ({}) | webhook={} | port={}",
        settings.ollama_base_url,
        settings.ollama_model,
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

__all__ = ["create_app", "main", "app"]
