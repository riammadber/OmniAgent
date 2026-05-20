"""Jobs router — CRUD + scheduling for Job resources."""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from omniagent.api.models import Job, JobCreate
from omniagent.db import queries
from omniagent.db.engine import get_session

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/jobs", tags=["jobs"])


async def _session() -> AsyncSession:  # type: ignore[return]
    """FastAPI dependency that yields a DB session."""
    async with get_session() as session:
        yield session


@router.post("", response_model=Job, status_code=201)
async def create_job(
    payload: JobCreate,
    session: AsyncSession = Depends(_session),
) -> Job:
    """Create and persist a new Job."""
    job = await queries.create_job(session, payload)
    logger.info("Created job id=%s name=%s", job.id, job.name)
    return job


@router.get("", response_model=list[Job])
async def list_jobs(
    status: Optional[str] = Query(None, description="Filter by status"),
    agent_id: Optional[str] = Query(None, description="Filter by agent_id"),
    session: AsyncSession = Depends(_session),
) -> list[Job]:
    """Return all jobs, optionally filtered."""
    return await queries.list_jobs(session, status=status, agent_id=agent_id)


@router.get("/{job_id}", response_model=Job)
async def get_job(
    job_id: str,
    session: AsyncSession = Depends(_session),
) -> Job:
    """Fetch a single job by ID."""
    jobs = await queries.list_jobs(session)
    for job in jobs:
        if str(job.id) == job_id:
            return job
    raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
