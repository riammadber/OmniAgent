"""Unit tests for provider factory selection and provider-specific wiring."""

from __future__ import annotations

from omniagent.inference.providers import (
    GroqProvider,
    NvidiaNIMProvider,
    OpenRouterProvider,
    get_provider,
)


class _Settings:
    llm_provider = ""
    groq_api_key = "groq-key"
    groq_base_url = "https://api.groq.com/openai/v1"
    openrouter_api_key = "openrouter-key"
    openrouter_base_url = "https://openrouter.ai/api/v1"
    openrouter_site_url = "https://example.com"
    openrouter_app_name = "OmniAgent"
    nvidia_nim_api_key = "nim-key"
    nvidia_nim_base_url = "https://integrate.api.nvidia.com/v1"


def test_get_provider_groq() -> None:
    s = _Settings()
    s.llm_provider = "groq"
    provider = get_provider(s)

    assert isinstance(provider, GroqProvider)


def test_get_provider_openrouter() -> None:
    s = _Settings()
    s.llm_provider = "openrouter"
    provider = get_provider(s)

    assert isinstance(provider, OpenRouterProvider)


def test_get_provider_nvidia_nim() -> None:
    s = _Settings()
    s.llm_provider = "nvidia_nim"
    provider = get_provider(s)

    assert isinstance(provider, NvidiaNIMProvider)
