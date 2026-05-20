"""LLM provider abstraction layer.

New providers can be added by subclassing ``LLMProvider`` and implementing
``invoke``.  The factory ``get_provider`` selects the right backend based on
the Settings object so callers never need to branch on provider names.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Optional

import httpx

logger = logging.getLogger(__name__)


# ── Abstract base ─────────────────────────────────────────────────────────────


class LLMProvider(ABC):
    """Common interface for all LLM backends."""

    @abstractmethod
    async def invoke(
        self,
        prompt: str,
        system: str = "",
        model: str = "",
        tools: Optional[list[dict]] = None,
        **kwargs: object,
    ) -> str:
        """Send *prompt* to the model and return the response text.

        Parameters
        ----------
        prompt:
            The user-facing message / task description.
        system:
            System instruction injected before the conversation.
        model:
            Model identifier string (provider-specific).  Falls back to the
            provider's default if empty.
        tools:
            Optional list of tool schemas in OpenAI/Anthropic format.
        """


# ── Claude ────────────────────────────────────────────────────────────────────


class ClaudeProvider(LLMProvider):
    """Anthropic Claude backend.

    Uses the official ``anthropic`` SDK when available, falling back to a
    raw httpx call so the project can run even without the SDK installed.
    """

    DEFAULT_MODEL = "claude-3-5-sonnet-20241022"

    def __init__(self, api_key: str) -> None:
        self._api_key = api_key

    async def invoke(
        self,
        prompt: str,
        system: str = "",
        model: str = "",
        tools: Optional[list[dict]] = None,
        **kwargs: object,
    ) -> str:
        """Invoke Claude and return the first text content block."""
        model = model or self.DEFAULT_MODEL
        try:
            import anthropic  # type: ignore[import]

            client = anthropic.AsyncAnthropic(api_key=self._api_key)
            messages = [{"role": "user", "content": prompt}]
            response = await client.messages.create(
                model=model,
                max_tokens=4096,
                system=system or "You are OmniAgent, a helpful AI assistant.",
                messages=messages,
            )
            return response.content[0].text  # type: ignore[index]
        except ImportError:
            logger.warning("anthropic SDK not installed; using httpx fallback")
            return await self._httpx_invoke(prompt, system, model)

    async def _httpx_invoke(self, prompt: str, system: str, model: str) -> str:
        """Raw HTTPS fallback for environments without the anthropic SDK."""
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": self._api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": model,
                    "max_tokens": 4096,
                    "system": system,
                    "messages": [{"role": "user", "content": prompt}],
                },
                timeout=120,
            )
            resp.raise_for_status()
            data = resp.json()
            return data["content"][0]["text"]


# ── GPT ───────────────────────────────────────────────────────────────────────


class GPTProvider(LLMProvider):
    """OpenAI GPT backend (stub — replace with full implementation as needed)."""

    DEFAULT_MODEL = "gpt-4o"

    def __init__(self, api_key: str) -> None:
        self._api_key = api_key

    async def invoke(
        self,
        prompt: str,
        system: str = "",
        model: str = "",
        tools: Optional[list[dict]] = None,
        **kwargs: object,
    ) -> str:
        """Invoke GPT via the openai SDK."""
        model = model or self.DEFAULT_MODEL
        try:
            from openai import AsyncOpenAI  # type: ignore[import]

            client = AsyncOpenAI(api_key=self._api_key)
            messages: list[dict] = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})
            response = await client.chat.completions.create(
                model=model,
                messages=messages,  # type: ignore[arg-type]
                max_tokens=4096,
            )
            return response.choices[0].message.content or ""
        except ImportError:
            logger.warning("openai SDK not installed; returning stub response")
            return f"[GPT stub] Received: {prompt[:100]}"


# ── Ollama ────────────────────────────────────────────────────────────────────


class OllamaProvider(LLMProvider):
    """Local Ollama backend — calls the REST API on localhost."""

    DEFAULT_MODEL = "llama3"

    def __init__(self, base_url: str = "http://localhost:11434") -> None:
        self._base_url = base_url.rstrip("/")

    async def invoke(
        self,
        prompt: str,
        system: str = "",
        model: str = "",
        tools: Optional[list[dict]] = None,
        **kwargs: object,
    ) -> str:
        """Send a generation request to the local Ollama server."""
        model = model or self.DEFAULT_MODEL
        full_prompt = f"{system}\n\n{prompt}" if system else prompt
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{self._base_url}/api/generate",
                json={"model": model, "prompt": full_prompt, "stream": False},
                timeout=300,
            )
            resp.raise_for_status()
            return resp.json().get("response", "")


# ── Factory ───────────────────────────────────────────────────────────────────


def get_provider(settings: object | None = None) -> LLMProvider:
    """Return the configured LLMProvider instance.

    Imports ``omniagent.config.settings`` lazily so unit tests can swap
    providers before the import happens.
    """
    if settings is None:
        from omniagent.config import settings as _settings  # type: ignore[assignment]

        settings = _settings

    provider_name: str = getattr(settings, "llm_provider", "claude")

    if provider_name == "claude":
        key = getattr(settings, "claude_api_key", None) or ""
        return ClaudeProvider(api_key=key)
    elif provider_name == "gpt":
        key = getattr(settings, "openai_api_key", None) or ""
        return GPTProvider(api_key=key)
    elif provider_name == "ollama":
        url = getattr(settings, "ollama_base_url", "http://localhost:11434")
        return OllamaProvider(base_url=url)
    else:
        raise ValueError(f"Unknown LLM provider: '{provider_name}'")
