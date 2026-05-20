"""Async SQLAlchemy engine + session factory.

We use a single engine per process.  The `get_session` async context manager
is the preferred way to obtain a database session inside FastAPI dependency
injection or agent code.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from omniagent.config import settings
from omniagent.db.models import Base

# One engine for the entire process lifetime.
engine = create_async_engine(
    settings.database_url,
    echo=settings.debug,
    future=True,
)

# Reusable session factory — never create sessions manually.
AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    expire_on_commit=False,
    class_=AsyncSession,
)


async def init_db() -> None:
    """Create all tables defined in the ORM models.

    Called once at server startup.  Safe to call multiple times — SQLAlchemy
    uses `CREATE TABLE IF NOT EXISTS` semantics.
    """
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


@asynccontextmanager
async def get_session() -> AsyncIterator[AsyncSession]:
    """Yield a short-lived AsyncSession and guarantee cleanup.

    Usage::

        async with get_session() as session:
            result = await session.execute(...)
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
