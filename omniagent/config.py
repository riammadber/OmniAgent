"""Configuration management using Pydantic Settings.

All settings are read from environment variables (and optionally a .env file).
No secret is ever hard-coded here — only defaults for non-sensitive values.
"""

from __future__ import annotations

from typing import Optional

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Central configuration object for OmniAgent.

    Values are resolved in this priority order:
    1. Explicit environment variable
    2. Value in the .env file (if present)
    3. Default listed below
    """

    # ── Database ──────────────────────────────────────────────────────────────
    database_url: str = "sqlite+aiosqlite:///./omniagent.db"

    # ── LLM ───────────────────────────────────────────────────────────────────
    llm_provider: str = "claude"  # claude | gpt | ollama
    claude_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None
    ollama_base_url: str = "http://localhost:11434"
    default_model: str = "claude-3-5-sonnet-20241022"

    # ── Telegram ──────────────────────────────────────────────────────────────
    telegram_bot_token: Optional[str] = None
    telegram_webhook_url: Optional[str] = None

    # ── Server ────────────────────────────────────────────────────────────────
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    debug: bool = False

    # ── Security ──────────────────────────────────────────────────────────────
    secret_key: str = "change-me-in-production"

    model_config = {"env_file": ".env", "case_sensitive": False}


settings = Settings()
