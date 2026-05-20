"""Pydantic v2 schemas for all OmniAgent domain objects.

These models are the single source of truth for validation across the API,
the agent core, and the gateway.  Every field has an explicit type so that
mypy / pyright can catch misuse at development time.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


# ── OmniMessage ───────────────────────────────────────────────────────────────


class OmniMessage(BaseModel):
    """Normalised message arriving from any supported channel."""

    sender: str = Field(..., description="Platform-specific user ID of the sender")
    channel: Literal["telegram", "discord", "slack", "internal"] = Field(
        ..., description="Originating channel"
    )
    intent: str = Field(..., description="Parsed intent, e.g. 'run_job' or 'check_status'")
    content: str = Field(..., description="Raw message body")
    metadata: dict = Field(default_factory=dict, description="Original platform payload")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# ── Job ───────────────────────────────────────────────────────────────────────


class JobCreate(BaseModel):
    """Request body for creating a new Job."""

    name: str
    schedule: Optional[str] = Field(None, description="Cron expression, e.g. '0 9 * * *'")
    agent_id: UUID
    context_injections: list[dict] = Field(
        default_factory=list,
        description="Docs, env vars, or table snapshots to inject at runtime",
    )


class Job(BaseModel):
    """A recurring or one-off unit of work assigned to an Agent."""

    id: UUID = Field(default_factory=uuid4)
    name: str
    schedule: Optional[str] = None
    agent_id: UUID
    context_injections: list[dict] = Field(default_factory=list)
    status: Literal["active", "paused", "archived"] = "active"
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    model_config = {"from_attributes": True}


# ── Run ───────────────────────────────────────────────────────────────────────


class RunCreate(BaseModel):
    """Payload an agent POSTs after completing (or failing) a run."""

    run_id: UUID
    status: Literal["pending", "executing", "success", "failed", "retrying"]
    output: str = ""
    error: Optional[str] = None
    tool_calls: list[dict] = Field(default_factory=list)
    duration_ms: int = 0
    agent_state: dict = Field(default_factory=dict)
    short_term_memory: dict = Field(default_factory=dict)


class Run(BaseModel):
    """A single execution of a Job, capturing the full trajectory."""

    id: UUID = Field(default_factory=uuid4)
    job_id: UUID
    agent_id: UUID
    status: Literal["pending", "executing", "success", "failed", "retrying"] = "pending"
    agent_state: dict = Field(default_factory=dict)
    short_term_memory: dict = Field(default_factory=dict)
    tool_calls: list[dict] = Field(default_factory=list)
    output: str = ""
    error: Optional[str] = None
    duration_ms: int = 0
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: Optional[datetime] = None

    model_config = {"from_attributes": True}


# ── Agent ─────────────────────────────────────────────────────────────────────


class AgentCreate(BaseModel):
    """Request body for registering a new Agent."""

    name: str
    model_provider: Literal["claude", "gpt", "ollama", "custom"]
    model_name: str = Field(..., description="e.g. 'claude-3-5-sonnet-20241022'")
    system_prompt: str = ""
    skills: list[str] = Field(default_factory=list)


class Agent(BaseModel):
    """A registered polling worker."""

    id: UUID = Field(default_factory=uuid4)
    name: str
    model_provider: Literal["claude", "gpt", "ollama", "custom"]
    model_name: str
    system_prompt: str = ""
    skills: list[str] = Field(default_factory=list)
    is_healthy: bool = True
    last_heartbeat: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    model_config = {"from_attributes": True}


# ── Skill ─────────────────────────────────────────────────────────────────────


class Skill(BaseModel):
    """A reusable, auto-synthesised Python skill."""

    id: UUID = Field(default_factory=uuid4)
    name: str
    description: str = ""
    python_code: str = Field(..., description="Auto-generated Python function body")
    input_schema: dict = Field(default_factory=dict)
    output_schema: dict = Field(default_factory=dict)
    usage_count: int = 0
    success_count: int = 0
    last_used: Optional[datetime] = None
    version: int = 1
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    model_config = {"from_attributes": True}


# ── ToolCall ──────────────────────────────────────────────────────────────────


class ToolCall(BaseModel):
    """Record of a single tool invocation within a Run."""

    id: UUID = Field(default_factory=uuid4)
    run_id: UUID
    tool_name: str
    input_args: dict = Field(default_factory=dict)
    output: Optional[str] = None
    error: Optional[str] = None
    duration_ms: int = 0
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    model_config = {"from_attributes": True}


# ── Generic responses ─────────────────────────────────────────────────────────


class HealthResponse(BaseModel):
    """Liveness check response."""

    status: str = "ok"
    version: str = "0.1.0"


class NextJobResponse(BaseModel):
    """Response returned by the long-poll next-job endpoint."""

    job: Optional[Job] = None
    poll_interval_seconds: int = 10
