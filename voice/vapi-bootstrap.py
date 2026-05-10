#!/usr/bin/env python3
"""One-shot bootstrap: create the Vapi Assistant + attach the phone number.

Reads creds from ``voice/.env`` (Vapi keys + the existing Gemma proxy URL
+ tool-webhook URL). Builds an Assistant via Vapi REST API with:

  * ``model.provider = "custom-llm"`` → Gemma 4 (or whatever the proxy is
    currently locked to) on Spark, reached over Tailscale Funnel HTTPS.
  * ``server.url`` → our voice/server.py tool webhook (one entry, handles
    every tool call shape Vapi POSTs).
  * Tool schemas from ``server.py:_build_vapi_tool_schemas`` (the same
    list returned by ``GET /api/agent.json``).

Then PATCHes the phone number to point at the new Assistant.

Run from the ``voice/`` directory:

    python vapi-bootstrap.py

Idempotent: if an Assistant named ``xlerobot-demo`` already exists, it
PATCHes that one instead of making a duplicate.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import requests
from dotenv import load_dotenv


HERE = Path(__file__).resolve().parent
load_dotenv(HERE / ".env")

VAPI_KEY = os.environ["VAPI_PRIVATE_KEY"]
VAPI_PHONE_NUMBER_ID = os.environ["VAPI_PHONE_NUMBER_ID"]

# Public LLM endpoint Vapi will call for completions.
LLM_BASE_URL = os.environ.get(
    "VAPI_CUSTOM_LLM_URL",
    "https://spark.tail4eec5f.ts.net",
)
LLM_API_KEY = os.environ.get(
    "GEMMA_PROXY_TOKEN",
    "",
)
LLM_MODEL = os.environ.get("GEMMA_MODEL", "gemma3:27b")

# Public tool-webhook for the voice server (Tailscale Funnel on the tablet).
TOOL_SERVER_URL = os.environ.get(
    "TOOL_SERVER_URL",
    "https://desktop-1r590mh.tail4eec5f.ts.net/api/tools",
)

ASSISTANT_NAME = "xlerobot-demo"

VAPI = "https://api.vapi.ai"
HEADERS = {"Authorization": f"Bearer {VAPI_KEY}", "Content-Type": "application/json"}


def fetch_agent_manifest() -> dict:
    """Pull the system prompt + tool schemas from the local voice server.

    Falls back to a static minimum if the local server isn't running.
    """
    try:
        r = requests.get("http://localhost:8765/api/agent.json", timeout=3)
        r.raise_for_status()
        return r.json()
    except Exception as exc:
        print(f"  warning: local server not reachable ({exc}); using static fallback.", file=sys.stderr)
        # Import the same constants the server uses, so the bootstrap
        # works even when the FastAPI process isn't up.
        sys.path.insert(0, str(HERE.parent))
        from voice.server import SYSTEM_PROMPT, _build_vapi_tool_schemas  # noqa: E402
        from voice.tools import VoiceTools  # noqa: E402

        skills = (VoiceTools()._skills)  # noqa: SLF001
        skills_list = [
            {
                "name": n,
                "description": m.get("description", ""),
                "preconditions": m.get("preconditions", ""),
                "postconditions": m.get("postconditions", ""),
                "vla_uri": (m.get("vla") or {}).get("uri"),
            }
            for n, m in skills.items()
        ]
        return {
            "system_prompt": SYSTEM_PROMPT,
            "tools": _build_vapi_tool_schemas(skills_list),
        }


def build_assistant_payload(system_prompt: str, tool_schemas: list[dict]) -> dict:
    """Construct the Assistant creation body for POST /assistant."""
    # Vapi wants tools shaped as {"type": "function", "function": {...}, "server": {...}}.
    # Our schemas come out as {"type": "function", "function": {...}}; attach a per-tool
    # `server.url` so Vapi knows where to POST when the LLM emits a function call.
    vapi_tools = []
    for t in tool_schemas:
        fn = t.get("function", {})
        vapi_tools.append({
            "type": "function",
            "function": fn,
            "server": {"url": f"{TOOL_SERVER_URL.rstrip('/')}/{fn.get('name')}"},
        })

    return {
        "name": ASSISTANT_NAME,
        "firstMessage": "Hi, this is the robot. What would you like me to pick up?",
        "voice": {
            # Vapi's built-in voices are free in the trial tier; swap to
            # 11labs/cartesia later for production polish.
            "provider": "vapi",
            "voiceId": "Elliot",
        },
        "transcriber": {
            "provider": "deepgram",
            "model": "nova-2",
            "language": "en-US",
        },
        "model": {
            "provider": "custom-llm",
            "url": LLM_BASE_URL.rstrip("/"),
            "model": LLM_MODEL,
            "messages": [{"role": "system", "content": system_prompt}],
            "tools": vapi_tools,
            "maxTokens": 400,
        },
        # Optional: a single fallback "server" URL Vapi POSTs to for any tool
        # call whose function-level server.url is missing.
        "server": {"url": TOOL_SERVER_URL},
    }


def find_existing_assistant(name: str) -> str | None:
    r = requests.get(f"{VAPI}/assistant", headers=HEADERS, timeout=15)
    r.raise_for_status()
    for a in r.json():
        if a.get("name") == name:
            return a.get("id")
    return None


def upsert_assistant(payload: dict) -> str:
    existing_id = find_existing_assistant(payload["name"])
    if existing_id:
        print(f"  found existing assistant {existing_id} ({payload['name']}); PATCHing.")
        r = requests.patch(
            f"{VAPI}/assistant/{existing_id}",
            headers=HEADERS,
            json=payload,
            timeout=30,
        )
    else:
        print(f"  creating new assistant {payload['name']!r}.")
        r = requests.post(f"{VAPI}/assistant", headers=HEADERS, json=payload, timeout=30)
    if not r.ok:
        print(f"  HTTP {r.status_code}: {r.text}", file=sys.stderr)
        r.raise_for_status()
    return r.json()["id"]


def attach_number_to_assistant(number_id: str, assistant_id: str) -> dict:
    r = requests.patch(
        f"{VAPI}/phone-number/{number_id}",
        headers=HEADERS,
        json={"assistantId": assistant_id},
        timeout=15,
    )
    if not r.ok:
        print(f"  HTTP {r.status_code}: {r.text}", file=sys.stderr)
        r.raise_for_status()
    return r.json()


def main() -> int:
    print(f"==> Fetching agent manifest")
    manifest = fetch_agent_manifest()
    print(f"    tools: {len(manifest['tools'])}")

    print(f"==> Upserting Vapi assistant {ASSISTANT_NAME!r}")
    payload = build_assistant_payload(manifest["system_prompt"], manifest["tools"])
    assistant_id = upsert_assistant(payload)
    print(f"    assistant id: {assistant_id}")

    print(f"==> Attaching phone number id {VAPI_PHONE_NUMBER_ID}")
    info = attach_number_to_assistant(VAPI_PHONE_NUMBER_ID, assistant_id)
    print(f"    number: {info.get('number','?')}  status: {info.get('status','?')}")

    print()
    print("==> Done.")
    print(f"    Dial:  {info.get('number','?')}")
    print(f"    Assistant: https://dashboard.vapi.ai/assistants/{assistant_id}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
