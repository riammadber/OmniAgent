"""High-level async query helpers.

Keeps SQL/ORM logic out of routers and agent code.  Every helper accepts an
``AsyncSession`` so callers can compose multiple queries in one transaction.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from omniagent.api.models import (
    Agent,
    AgentCreate,
    Job,
    JobCreate,
    Run,
    RunCreate,
)
from omniagent.db.models import AgentORM, JobORM, RunORM


# ── Job queries ───────────────────────────────────────────────────────────────


async def create_job(session: AsyncSession, payload: JobCreate) -> Job:
    """Insert a new Job row and return the Pydantic schema."""
    orm = JobORM(
        id=str(uuid.uuid4()),
        name=payload.name,
        schedule=payload.schedule,
        agent_id=str(payload.agent_id),
        context_injections=json.dumps(payload.context_injections),
        status="active",
    )
    session.add(orm)
    await session.flush()
    return _job_to_schema(orm)


async def list_jobs(
    session: AsyncSession,
    status: Optional[str] = None,
    agent_id: Optional[str] = None,
) -> list[Job]:
    """Return all jobs, optionally filtered by status and/or agent_id."""
    stmt = select(JobORM)
    if status:
        stmt = stmt.where(JobORM.status == status)
    if agent_id:
        stmt = stmt.where(JobORM.agent_id == agent_id)
    result = await session.execute(stmt)
    return [_job_to_schema(row) for row in result.scalars()]


async def get_next_job_for_agent(session: AsyncSession, agent_id: str) -> Optional[Job]:
    """Return the oldest active job assigned to *agent_id*, or None."""
    stmt = (
        select(JobORM)
        .where(JobORM.agent_id == agent_id, JobORM.status == "active")
        .order_by(JobORM.created_at)
        .limit(1)
    )
    result = await session.execute(stmt)
    orm = result.scalar_one_or_none()
    return _job_to_schema(orm) if orm else None


# ── Run queries ───────────────────────────────────────────────────────────────


async def create_run(session: AsyncSession, job_id: str, agent_id: str) -> Run:
    """Create a pending Run row for the given job."""
    orm = RunORM(
        id=str(uuid.uuid4()),
        job_id=job_id,
        agent_id=agent_id,
        status="pending",
    )
    session.add(orm)
    await session.flush()
    return _run_to_schema(orm)


async def update_run(session: AsyncSession, payload: RunCreate) -> Optional[Run]:
    """Persist agent-submitted run results; return the updated schema."""
    run_id = str(payload.run_id)
    stmt = (
        update(RunORM)
        .where(RunORM.id == run_id)
        .values(
            status=payload.status,
            output=payload.output,
            error=payload.error,
            tool_calls=json.dumps(payload.tool_calls),
            duration_ms=payload.duration_ms,
            agent_state=json.dumps(payload.agent_state),
            short_term_memory=json.dumps(payload.short_term_memory),
            completed_at=datetime.now(timezone.utc) if payload.status in ("success", "failed") else None,
        )
        .returning(RunORM)
    )
    result = await session.execute(stmt)
    orm = result.scalar_one_or_none()
    return _run_to_schema(orm) if orm else None


async def get_run(session: AsyncSession, run_id: str) -> Optional[Run]:
    """Fetch a single Run by primary key."""
    result = await session.execute(select(RunORM).where(RunORM.id == run_id))
    orm = result.scalar_one_or_none()
    return _run_to_schema(orm) if orm else None


# ── Agent queries ─────────────────────────────────────────────────────────────


async def register_agent(session: AsyncSession, payload: AgentCreate) -> Agent:
    """Insert a new Agent registration."""
    orm = AgentORM(
        id=str(uuid.uuid4()),
        name=payload.name,
        model_provider=payload.model_provider,
        model_name=payload.model_name,
        system_prompt=payload.system_prompt,
        skills=json.dumps(payload.skills),
        is_healthy=1,
        last_heartbeat=datetime.now(timezone.utc),
    )
    session.add(orm)
    await session.flush()
    return _agent_to_schema(orm)


async def list_agents(session: AsyncSession) -> list[Agent]:
    """Return all registered agents."""
    result = await session.execute(select(AgentORM))
    return [_agent_to_schema(row) for row in result.scalars()]


async def get_agent(session: AsyncSession, agent_id: str) -> Optional[Agent]:
    """Fetch a single Agent by primary key."""
    result = await session.execute(select(AgentORM).where(AgentORM.id == agent_id))
    orm = result.scalar_one_or_none()
    return _agent_to_schema(orm) if orm else None


# ── ORM → Pydantic converters ─────────────────────────────────────────────────


def _job_to_schema(orm: JobORM) -> Job:
    return Job(
        id=orm.id,
        name=orm.name,
        schedule=orm.schedule,
        agent_id=orm.agent_id,
        context_injections=orm.get_context_injections(),
        status=orm.status,  # type: ignore[arg-type]
        created_at=orm.created_at,
        updated_at=orm.updated_at,
    )


def _run_to_schema(orm: RunORM) -> Run:
    return Run(
        id=orm.id,
        job_id=orm.job_id,
        agent_id=orm.agent_id,
        status=orm.status,  # type: ignore[arg-type]
        agent_state=orm.get_agent_state(),
        short_term_memory=orm.get_short_term_memory(),
        tool_calls=orm.get_tool_calls(),
        output=orm.output,
        error=orm.error,
        duration_ms=orm.duration_ms,
        created_at=orm.created_at,
        completed_at=orm.completed_at,
    )


def _agent_to_schema(orm: AgentORM) -> Agent:
    return Agent(
        id=orm.id,
        name=orm.name,
        model_provider=orm.model_provider,  # type: ignore[arg-type]
        model_name=orm.model_name,
        system_prompt=orm.system_prompt,
        skills=orm.get_skills(),
        is_healthy=bool(orm.is_healthy),
        last_heartbeat=orm.last_heartbeat,
        created_at=orm.created_at,
    )
