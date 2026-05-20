"""Normalised message format for all incoming channels.

Every messaging adapter (Telegram, Discord, Slack, …) converts its native
update format into an ``OmniMessage``.  Downstream code only ever sees
``OmniMessage``, keeping the gateway fully pluggable.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field


class OmniMessage(BaseModel):
    """Channel-agnostic representation of a user message."""

    sender: str = Field(..., description="Platform-specific user/chat ID")
    channel: Literal["telegram", "discord", "slack", "internal"]
    intent: str = Field(
        ...,
        description="Parsed intent slug, e.g. 'run_job', 'check_status', 'list_skills'",
    )
    content: str = Field(..., description="Raw message text")
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Original platform payload preserved for debugging",
    )
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


def parse_intent(text: str) -> str:
    """Derive an intent slug from raw message text.

    Recognises a small set of slash commands; everything else is 'chat'.
    """
    text = text.strip().lower()
    if text.startswith("/run_job") or text.startswith("/run"):
        return "run_job"
    if text.startswith("/status"):
        return "check_status"
    if text.startswith("/list_skills") or text.startswith("/skills"):
        return "list_skills"
    if text.startswith("/help"):
        return "help"
    return "chat"
