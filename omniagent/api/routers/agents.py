"""Agents router — registration, heartbeat, and next-job long-poll."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from omniagent.api.models import Agent, AgentCreate, NextJobResponse
from omniagent.db import queries
from omniagent.db.engine import get_session

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/agents", tags=["agents"])


async def _session() -> AsyncSession:  # type: ignore[return]
    async with get_session() as session:
        yield session


@router.post("", response_model=Agent, status_code=201)
async def register_agent(
    payload: AgentCreate,
    session: AsyncSession = Depends(_session),
) -> Agent:
    """Register a new agent worker with the control plane."""
    agent = await queries.register_agent(session, payload)
    logger.info("Registered agent id=%s name=%s", agent.id, agent.name)
    return agent


@router.get("", response_model=list[Agent])
async def list_agents(
    session: AsyncSession = Depends(_session),
) -> list[Agent]:
    """Return all registered agents."""
    return await queries.list_agents(session)


@router.get("/{agent_id}", response_model=Agent)
async def get_agent(
    agent_id: str,
    session: AsyncSession = Depends(_session),
) -> Agent:
    """Fetch a single agent by ID."""
    agent = await queries.get_agent(session, agent_id)
    if agent is None:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_id}' not found")
    return agent


@router.get("/{agent_id}/next-job", response_model=NextJobResponse)
async def next_job(
    agent_id: str,
    session: AsyncSession = Depends(_session),
) -> NextJobResponse:
    """Long-poll endpoint for agents to pull their next pending job.

    Returns the oldest active job assigned to this agent.  If no work is
    available, returns ``{"job": null, "poll_interval_seconds": 10}`` so the
    agent knows when to poll again.

    Agents should call this endpoint in a tight loop with the suggested
    ``poll_interval_seconds`` delay between retries.
    """
    agent = await queries.get_agent(session, agent_id)
    if agent is None:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_id}' not found")

    job = await queries.get_next_job_for_agent(session, agent_id)
    if job:
        logger.info("Dispatching job id=%s to agent id=%s", job.id, agent_id)
    return NextJobResponse(job=job, poll_interval_seconds=10)
