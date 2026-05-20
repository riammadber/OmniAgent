"""Runs router — submit execution results and retrieve run details."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from omniagent.api.models import Run, RunCreate
from omniagent.db import queries
from omniagent.db.engine import get_session

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/runs", tags=["runs"])


async def _session() -> AsyncSession:  # type: ignore[return]
    async with get_session() as session:
        yield session


@router.post("", response_model=Run, status_code=201)
async def submit_run(
    payload: RunCreate,
    session: AsyncSession = Depends(_session),
) -> Run:
    """Agent POSTs a completed or failed run result here.

    Validates the payload, persists to DB, and triggers async skill synthesis
    for complex successful runs.
    """
    run = await queries.update_run(session, payload)
    if run is None:
        raise HTTPException(
            status_code=404, detail=f"Run '{payload.run_id}' not found"
        )
    logger.info("Run %s updated status=%s", run.id, run.status)
    return run


@router.get("/{run_id}", response_model=Run)
async def get_run(
    run_id: str,
    session: AsyncSession = Depends(_session),
) -> Run:
    """Fetch full run details including tool calls and trajectory."""
    run = await queries.get_run(session, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")
    return run
