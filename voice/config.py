"""12-factor settings for the voice frontend.

Most LLM + camera + VLA env vars are inherited from the orchestrator
(GEMMA_MODEL, OLLAMA_HOST, GEMMA_PROXY_TOKEN, CAMERA_INDEX, etc.) — we
read them directly via :mod:`orchestrator.orchestrator` so the two
processes stay in lock-step.

This module owns only the voice-IO surface:

* Twilio (telephony)
* Deepgram (streaming STT)
* ElevenLabs (streaming TTS)
* Server bind host/port
* Optional confirm-before-execute gate

Pydantic-settings loads ``.env`` and real env vars; tests can call
``get_settings.cache_clear()`` between env mutations.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class VoiceSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", case_sensitive=False, extra="ignore"
    )

    # ── Telephony (Twilio) ────────────────────────────────────────────────
    twilio_account_sid: str = Field(default="", description="Twilio Account SID.")
    twilio_auth_token: str = Field(default="", description="Twilio Auth Token.")
    twilio_phone_number: str = Field(default="", description="Inbound E.164 number, e.g. +1...")

    # ── Voice stack (cloud streaming for phone-call latency) ──────────────
    deepgram_api_key: str = Field(default="", description="Deepgram streaming STT key.")
    elevenlabs_api_key: str = Field(default="", description="ElevenLabs TTS key.")
    elevenlabs_voice_id: str = Field(
        default="21m00Tcm4TlvDq8ikWAM",
        description="ElevenLabs voice id. Default = stock 'Rachel'.",
    )
    elevenlabs_model: str = Field(
        default="eleven_turbo_v2_5",
        description="Lowest-latency ElevenLabs model.",
    )

    # ── LLM (Pipecat-side handle on the Gemma proxy) ──────────────────────
    # The orchestrator uses requests against /api/generate. Pipecat's
    # OpenAILLMService speaks /v1/chat/completions, which ollama also
    # serves. We default to localhost:11434 because the proxy on Spark
    # surfaces both interfaces. Override with the proxy URL + token when
    # running off-box. The orchestrator's _gemma_call still uses its own
    # GEMMA_PROXY_TOKEN env var — they don't fight.
    ollama_base_url: str = Field(
        default="http://localhost:11434/v1",
        description="OpenAI-compatible /v1 base URL. Set to the proxy URL when off-Spark.",
    )
    ollama_api_key: str = Field(
        default="ollama",
        description="Bearer for the proxy. Pipecat's OpenAI client requires non-empty.",
    )
    ollama_model: str = Field(
        default="gemma4:26b-a4b-it-q4",
        description="LLM tag. Must be served by the configured Ollama / proxy.",
    )

    # ── Public webhook host (for the TwiML <Stream url=...> we hand Twilio) ─
    webhook_host: str = Field(
        default="http://localhost:8765",
        description="Public HTTPS host this server runs behind (Tailscale Funnel or bore.pub URL).",
    )

    # ── Server ────────────────────────────────────────────────────────────
    server_host: str = Field(default="0.0.0.0")
    server_port: int = Field(default=8765)
    log_level: str = Field(default="INFO")

    # ── UX gate ──────────────────────────────────────────────────────────
    confirm_before_execute: bool = Field(
        default=True,
        description="If True, agent asks before running a VLA. False for hands-free batch.",
    )

    @field_validator("log_level")
    @classmethod
    def _normalize_log_level(cls, v: str) -> str:
        v_up = v.upper().strip()
        return v_up if v_up in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"} else "INFO"

    @field_validator("webhook_host", "ollama_base_url")
    @classmethod
    def _strip_trailing_slash(cls, v: str) -> str:
        return v.rstrip("/")


@lru_cache(maxsize=1)
def get_settings() -> VoiceSettings:
    return VoiceSettings()  # type: ignore[call-arg]


__all__ = ["VoiceSettings", "get_settings"]
