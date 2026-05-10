"""Settings tests — env loading + validator behaviour."""

from __future__ import annotations

import pytest

from voice.config import VoiceSettings, get_settings


@pytest.fixture(autouse=True)
def _clear_cache():
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    for k in (
        "DEEPGRAM_API_KEY",
        "ELEVENLABS_API_KEY",
        "TWILIO_ACCOUNT_SID",
        "OLLAMA_BASE_URL",
        "OLLAMA_MODEL",
        "WEBHOOK_HOST",
    ):
        monkeypatch.delenv(k, raising=False)
    s = get_settings()
    assert s.ollama_model == "gemma4:26b-a4b-it-q4"
    assert s.ollama_base_url == "http://localhost:11434/v1"
    assert s.server_port == 8765
    assert s.confirm_before_execute is True
    assert s.elevenlabs_model == "eleven_turbo_v2_5"


def test_url_trailing_slash_stripped(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://localhost:11434/v1/")
    monkeypatch.setenv("WEBHOOK_HOST", "https://voice.example.com//")
    s = get_settings()
    assert s.ollama_base_url == "http://localhost:11434/v1"
    assert s.webhook_host == "https://voice.example.com"


def test_log_level_normalisation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LOG_LEVEL", "debug")
    assert get_settings().log_level == "DEBUG"
    get_settings.cache_clear()
    monkeypatch.setenv("LOG_LEVEL", "junk")
    assert get_settings().log_level == "INFO"


def test_confirm_gate_overridable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CONFIRM_BEFORE_EXECUTE", "false")
    assert get_settings().confirm_before_execute is False


def test_sponsor_keys_default_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    for k in ("DEEPGRAM_API_KEY", "ELEVENLABS_API_KEY", "TWILIO_ACCOUNT_SID",
              "TWILIO_AUTH_TOKEN", "TWILIO_PHONE_NUMBER"):
        monkeypatch.delenv(k, raising=False)
    s = get_settings()
    assert s.deepgram_api_key == ""
    assert s.elevenlabs_api_key == ""
    assert s.twilio_account_sid == ""
