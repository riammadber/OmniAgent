"""Telegram webhook adapter.

Receives Telegram Bot API ``Update`` objects via a POST webhook and converts
them into ``OmniMessage`` instances.  Responses are sent back via the Bot API
using httpx so the entire adapter is non-blocking.

Setup:
1. Obtain a bot token from @BotFather.
2. Set TELEGRAM_BOT_TOKEN and TELEGRAM_WEBHOOK_URL in .env.
3. Register the webhook once::

       POST https://api.telegram.org/bot<token>/setWebhook
       {"url": "https://<your-domain>/gateway/telegram/webhook"}
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import httpx

from omniagent.gateway.message import OmniMessage, parse_intent

logger = logging.getLogger(__name__)

TELEGRAM_API = "https://api.telegram.org/bot{token}"


class TelegramGateway:
    """Telegram Bot API adapter."""

    def __init__(self, token: str) -> None:
        if not token:
            raise ValueError("TELEGRAM_BOT_TOKEN must be set to use TelegramGateway")
        self._token = token
        self._base = TELEGRAM_API.format(token=token)

    async def handle_webhook(self, update: dict[str, Any]) -> Optional[OmniMessage]:
        """Convert a Telegram Update dict into an OmniMessage.

        Returns None if the update does not contain a processable message
        (e.g. edited messages, channel posts, etc.).
        """
        message = update.get("message") or update.get("edited_message")
        if not message:
            logger.debug("Ignoring non-message update: %s", list(update.keys()))
            return None

        text: str = message.get("text", "")
        chat_id = str(message.get("chat", {}).get("id", ""))
        sender_id = str(message.get("from", {}).get("id", chat_id))

        return OmniMessage(
            sender=sender_id,
            channel="telegram",
            intent=parse_intent(text),
            content=text,
            metadata={"chat_id": chat_id, "update": update},
        )

    async def send_message(
        self,
        chat_id: str,
        text: str,
        reply_markup: Optional[dict[str, Any]] = None,
    ) -> None:
        """Send a text message to *chat_id* via the Bot API."""
        payload: dict[str, Any] = {"chat_id": chat_id, "text": text, "parse_mode": "Markdown"}
        if reply_markup:
            payload["reply_markup"] = reply_markup

        async with httpx.AsyncClient() as client:
            resp = await client.post(f"{self._base}/sendMessage", json=payload, timeout=30)
            if resp.status_code != 200:
                logger.error(
                    "Telegram sendMessage failed: %d %s", resp.status_code, resp.text[:200]
                )

    async def on_job_complete(self, run_data: dict[str, Any], user_id: str) -> None:
        """Notify a user that their job has finished.

        Sends a brief summary including status, output snippet, and duration.
        """
        status = run_data.get("status", "unknown")
        output = (run_data.get("output") or "")[:300]
        duration_ms = run_data.get("duration_ms", 0)
        run_id = run_data.get("id", "?")

        emoji = "✅" if status == "success" else "❌"
        text = (
            f"{emoji} *Run {run_id[:8]}* finished\n"
            f"Status: `{status}`\n"
            f"Duration: {duration_ms} ms\n"
        )
        if output:
            text += f"Output:\n```\n{output}\n```"
        if run_data.get("error"):
            text += f"\nError: `{run_data['error'][:200]}`"

        await self.send_message(user_id, text)
