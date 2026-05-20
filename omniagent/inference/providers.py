"""LLM provider abstraction layer.

New providers can be added by subclassing ``LLMProvider`` and implementing
``invoke``.  The factory ``get_provider`` selects the right backend based on
the Settings object so callers never need to branch on provider names.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Optional

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


# ── OpenAI-compatible providers ─────────────────────────────────────────────


class OpenAICompatibleProvider(LLMProvider):
    """Generic OpenAI-compatible chat completion backend over HTTP."""

    DEFAULT_MODEL = ""

    def __init__(
        self,
        api_key: str,
        base_url: str,
        default_model: str,
        extra_headers: Optional[dict[str, str]] = None,
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self.DEFAULT_MODEL = default_model
        self._extra_headers = extra_headers or {}

    async def invoke(
        self,
        prompt: str,
        system: str = "",
        model: str = "",
        tools: Optional[list[dict]] = None,
        **kwargs: object,
    ) -> str:
        """Invoke an OpenAI-compatible /chat/completions endpoint."""
        selected_model = model or self.DEFAULT_MODEL
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload: dict[str, Any] = {
            "model": selected_model,
            "messages": messages,
            "max_tokens": 4096,
        }
        if tools:
            payload["tools"] = tools

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            **self._extra_headers,
        }

        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{self._base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=120,
            )
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"].get("content", "") or ""


class GroqProvider(OpenAICompatibleProvider):
    """Groq backend via its OpenAI-compatible API."""

    DEFAULT_MODEL = "llama-3.1-70b-versatile"

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.groq.com/openai/v1",
    ) -> None:
        super().__init__(
            api_key=api_key,
            base_url=base_url,
            default_model=self.DEFAULT_MODEL,
        )


class OpenRouterProvider(OpenAICompatibleProvider):
    """OpenRouter backend via its OpenAI-compatible API."""

    DEFAULT_MODEL = "openai/gpt-4o-mini"

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://openrouter.ai/api/v1",
        site_url: str = "",
        app_name: str = "",
    ) -> None:
        extra_headers: dict[str, str] = {}
        if site_url:
            extra_headers["HTTP-Referer"] = site_url
        if app_name:
            extra_headers["X-Title"] = app_name
        super().__init__(
            api_key=api_key,
            base_url=base_url,
            default_model=self.DEFAULT_MODEL,
            extra_headers=extra_headers,
        )


class NvidiaNIMProvider(OpenAICompatibleProvider):
    """NVIDIA NIM backend via its OpenAI-compatible API."""

    DEFAULT_MODEL = "meta/llama-3.1-70b-instruct"

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://integrate.api.nvidia.com/v1",
    ) -> None:
        super().__init__(
            api_key=api_key,
            base_url=base_url,
            default_model=self.DEFAULT_MODEL,
        )


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
    elif provider_name == "groq":
        key = getattr(settings, "groq_api_key", None) or ""
        url = getattr(settings, "groq_base_url", "https://api.groq.com/openai/v1")
        return GroqProvider(api_key=key, base_url=url)
    elif provider_name == "openrouter":
        key = getattr(settings, "openrouter_api_key", None) or ""
        url = getattr(settings, "openrouter_base_url", "https://openrouter.ai/api/v1")
        site_url = getattr(settings, "openrouter_site_url", "") or ""
        app_name = getattr(settings, "openrouter_app_name", "") or ""
        return OpenRouterProvider(
            api_key=key,
            base_url=url,
            site_url=site_url,
            app_name=app_name,
        )
    elif provider_name == "nvidia_nim":
        key = getattr(settings, "nvidia_nim_api_key", None) or ""
        url = getattr(settings, "nvidia_nim_base_url", "https://integrate.api.nvidia.com/v1")
        return NvidiaNIMProvider(api_key=key, base_url=url)
    else:
        raise ValueError(f"Unknown LLM provider: '{provider_name}'")
