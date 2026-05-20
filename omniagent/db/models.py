"""SQLAlchemy 2.0 async ORM models.

These classes map to database tables.  We keep them separate from the
Pydantic schemas (api/models.py) so that the persistence layer can evolve
independently of the API contract.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import DateTime, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


def _now() -> datetime:
    """Return the current UTC datetime (helper used as column default)."""
    return datetime.now(timezone.utc)


def _uuid() -> str:
    """Return a new UUID4 string (helper used as column default)."""
    return str(uuid.uuid4())


class Base(DeclarativeBase):
    """Shared declarative base for all ORM models."""


class JobORM(Base):
    """Persisted Job record."""

    __tablename__ = "jobs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    schedule: Mapped[str | None] = mapped_column(String(100), nullable=True)
    agent_id: Mapped[str] = mapped_column(String(36), nullable=False)
    context_injections: Mapped[str] = mapped_column(Text, default="[]")
    status: Mapped[str] = mapped_column(String(20), default="active")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=_now, onupdate=_now)

    # ── helpers ───────────────────────────────────────────────────────────────

    def get_context_injections(self) -> list[dict[str, Any]]:
        """Deserialise the JSON-encoded context_injections column."""
        return json.loads(self.context_injections or "[]")

    def set_context_injections(self, value: list[dict[str, Any]]) -> None:
        """Serialise context_injections to JSON for storage."""
        self.context_injections = json.dumps(value)


class RunORM(Base):
    """Persisted Run record."""

    __tablename__ = "runs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    job_id: Mapped[str] = mapped_column(String(36), nullable=False)
    agent_id: Mapped[str] = mapped_column(String(36), nullable=False)
    status: Mapped[str] = mapped_column(String(20), default="pending")
    agent_state: Mapped[str] = mapped_column(Text, default="{}")
    short_term_memory: Mapped[str] = mapped_column(Text, default="{}")
    tool_calls: Mapped[str] = mapped_column(Text, default="[]")
    output: Mapped[str] = mapped_column(Text, default="")
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    duration_ms: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    def get_tool_calls(self) -> list[dict[str, Any]]:
        """Deserialise the JSON-encoded tool_calls column."""
        return json.loads(self.tool_calls or "[]")

    def get_agent_state(self) -> dict[str, Any]:
        """Deserialise the JSON-encoded agent_state column."""
        return json.loads(self.agent_state or "{}")

    def get_short_term_memory(self) -> dict[str, Any]:
        """Deserialise the JSON-encoded short_term_memory column."""
        return json.loads(self.short_term_memory or "{}")


class AgentORM(Base):
    """Persisted Agent registration."""

    __tablename__ = "agents"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    model_provider: Mapped[str] = mapped_column(String(20), nullable=False)
    model_name: Mapped[str] = mapped_column(String(100), nullable=False)
    system_prompt: Mapped[str] = mapped_column(Text, default="")
    skills: Mapped[str] = mapped_column(Text, default="[]")
    is_healthy: Mapped[int] = mapped_column(Integer, default=1)
    last_heartbeat: Mapped[datetime] = mapped_column(DateTime, default=_now)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now)

    def get_skills(self) -> list[str]:
        """Deserialise the JSON-encoded skills list."""
        return json.loads(self.skills or "[]")


class SkillORM(Base):
    """Persisted Skill definition."""

    __tablename__ = "skills"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    name: Mapped[str] = mapped_column(String(255), nullable=False, unique=True)
    description: Mapped[str] = mapped_column(Text, default="")
    python_code: Mapped[str] = mapped_column(Text, nullable=False)
    input_schema: Mapped[str] = mapped_column(Text, default="{}")
    output_schema: Mapped[str] = mapped_column(Text, default="{}")
    usage_count: Mapped[int] = mapped_column(Integer, default=0)
    success_count: Mapped[int] = mapped_column(Integer, default=0)
    last_used: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    version: Mapped[int] = mapped_column(Integer, default=1)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now)
