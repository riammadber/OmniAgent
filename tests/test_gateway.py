"""Unit tests for the messaging gateway — OmniMessage normalisation and Telegram adapter."""

from __future__ import annotations

from datetime import datetime

import pytest

from omniagent.gateway.message import OmniMessage, parse_intent
from omniagent.gateway.telegram import TelegramGateway


# ── parse_intent tests ────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "text, expected",
    [
        ("/run_job my_job", "run_job"),
        ("/run something", "run_job"),
        ("/status", "check_status"),
        ("/list_skills", "list_skills"),
        ("/skills", "list_skills"),
        ("/help", "help"),
        ("Hello there!", "chat"),
        ("", "chat"),
        ("/UNKNOWN_COMMAND", "chat"),
    ],
)
def test_parse_intent(text: str, expected: str) -> None:
    assert parse_intent(text) == expected


# ── OmniMessage tests ─────────────────────────────────────────────────────────


def test_omni_message_defaults() -> None:
    msg = OmniMessage(
        sender="user123",
        channel="telegram",
        intent="chat",
        content="Hello",
    )
    assert msg.metadata == {}
    assert isinstance(msg.timestamp, datetime)


def test_omni_message_all_channels() -> None:
    for channel in ("telegram", "discord", "slack", "internal"):
        msg = OmniMessage(sender="u", channel=channel, intent="chat", content="x")
        assert msg.channel == channel


def test_omni_message_invalid_channel() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        OmniMessage(sender="u", channel="whatsapp", intent="chat", content="x")  # type: ignore[arg-type]


# ── TelegramGateway tests ─────────────────────────────────────────────────────


def test_telegram_gateway_requires_token() -> None:
    with pytest.raises(ValueError, match="TELEGRAM_BOT_TOKEN"):
        TelegramGateway(token="")


@pytest.mark.asyncio
async def test_telegram_handle_webhook_message() -> None:
    gateway = TelegramGateway(token="test-token")
    update = {
        "update_id": 1,
        "message": {
            "message_id": 42,
            "from": {"id": 99, "first_name": "Alice"},
            "chat": {"id": 99, "type": "private"},
            "text": "/run_job daily_report",
            "date": 1700000000,
        },
    }
    msg = await gateway.handle_webhook(update)

    assert msg is not None
    assert msg.sender == "99"
    assert msg.channel == "telegram"
    assert msg.intent == "run_job"
    assert msg.content == "/run_job daily_report"
    assert msg.metadata["chat_id"] == "99"


@pytest.mark.asyncio
async def test_telegram_handle_webhook_no_message() -> None:
    """Updates without a message (e.g. poll, inline query) should return None."""
    gateway = TelegramGateway(token="test-token")
    update = {"update_id": 2, "poll": {"id": "poll1"}}
    msg = await gateway.handle_webhook(update)
    assert msg is None


@pytest.mark.asyncio
async def test_telegram_on_job_complete_success(monkeypatch: pytest.MonkeyPatch) -> None:
    """on_job_complete should call send_message with a success summary."""
    gateway = TelegramGateway(token="test-token")
    sent: list[dict] = []

    async def mock_send(chat_id: str, text: str, **kwargs: object) -> None:
        sent.append({"chat_id": chat_id, "text": text})

    monkeypatch.setattr(gateway, "send_message", mock_send)

    run_data = {
        "id": "abc123",
        "status": "success",
        "output": "Report generated",
        "duration_ms": 1234,
    }
    await gateway.on_job_complete(run_data, user_id="user_42")

    assert len(sent) == 1
    assert "success" in sent[0]["text"]
    assert "user_42" == sent[0]["chat_id"]


@pytest.mark.asyncio
async def test_telegram_on_job_complete_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    gateway = TelegramGateway(token="test-token")
    sent: list[dict] = []

    async def mock_send(chat_id: str, text: str, **kwargs: object) -> None:
        sent.append({"text": text})

    monkeypatch.setattr(gateway, "send_message", mock_send)

    run_data = {
        "id": "xyz",
        "status": "failed",
        "output": "",
        "error": "Timeout exceeded",
        "duration_ms": 5000,
    }
    await gateway.on_job_complete(run_data, user_id="user_7")

    assert sent
    assert "failed" in sent[0]["text"]
    assert "Timeout exceeded" in sent[0]["text"]
