"""FastAPI control plane server.

Responsibilities:
- Mount API routers under /api/v1/
- Mount the Telegram gateway webhook under /gateway/
- Initialise the database on startup
- Provide structured JSON request logging via middleware
- Expose /health for liveness probes
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from omniagent.api.models import HealthResponse
from omniagent.api.routers import agents, jobs, runs
from omniagent.db.engine import init_db

logger = logging.getLogger(__name__)


# ── Lifespan ──────────────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Run startup and shutdown tasks around the server lifecycle."""
    logger.info("OmniAgent control plane starting up")
    await init_db()
    yield
    logger.info("OmniAgent control plane shutting down")


# ── Application factory ───────────────────────────────────────────────────────


def create_app() -> FastAPI:
    """Build and configure the FastAPI application.

    Separated from module-level instantiation so tests can create fresh
    instances without side effects.
    """
    app = FastAPI(
        title="OmniAgent Control Plane",
        description="Polling-based control plane for the OmniAgent framework",
        version="0.1.0",
        lifespan=lifespan,
    )

    # ── CORS ──────────────────────────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Request logging middleware ────────────────────────────────────────────
    @app.middleware("http")
    async def log_requests(request: Request, call_next: Any) -> Response:
        start = time.monotonic()
        response: Response = await call_next(request)
        duration_ms = int((time.monotonic() - start) * 1000)
        logger.info(
            "method=%s path=%s status=%d duration_ms=%d",
            request.method,
            request.url.path,
            response.status_code,
            duration_ms,
        )
        return response

    # ── Global error handler ─────────────────────────────────────────────────
    @app.exception_handler(Exception)
    async def global_error_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("Unhandled error on %s %s", request.method, request.url.path)
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error", "type": type(exc).__name__},
        )

    # ── Health endpoint ───────────────────────────────────────────────────────
    @app.get("/health", response_model=HealthResponse, tags=["health"])
    async def health() -> HealthResponse:
        """Liveness probe — always returns 200 OK if the server is running."""
        return HealthResponse()

    # ── API v1 routers ────────────────────────────────────────────────────────
    prefix = "/api/v1"
    app.include_router(jobs.router, prefix=prefix)
    app.include_router(runs.router, prefix=prefix)
    app.include_router(agents.router, prefix=prefix)

    # ── Telegram gateway ──────────────────────────────────────────────────────
    _mount_telegram_gateway(app)

    return app


def _mount_telegram_gateway(app: FastAPI) -> None:
    """Attach the Telegram webhook route if a bot token is configured."""
    from fastapi import BackgroundTasks
    from omniagent.config import settings

    if not settings.telegram_bot_token:
        logger.info("TELEGRAM_BOT_TOKEN not set — Telegram gateway disabled")
        return

    from omniagent.gateway.telegram import TelegramGateway

    gateway = TelegramGateway(token=settings.telegram_bot_token)

    @app.post("/gateway/telegram/webhook", tags=["gateway"])
    async def telegram_webhook(
        update: dict,
        background_tasks: BackgroundTasks,
    ) -> dict:
        """Receive a Telegram update and process it asynchronously."""
        message = await gateway.handle_webhook(update)
        if message:
            background_tasks.add_task(_process_telegram_message, gateway, message)
        return {"ok": True}


async def _process_telegram_message(gateway: Any, message: Any) -> None:
    """Background task: route the normalised OmniMessage to the right handler."""
    chat_id: str = message.metadata.get("chat_id", message.sender)
    intent: str = message.intent

    if intent == "run_job":
        await gateway.send_message(chat_id, "🚀 Job queued! I'll notify you when it's done.")
    elif intent == "check_status":
        await gateway.send_message(chat_id, "📊 Checking status… (use /api/v1/runs)")
    elif intent == "list_skills":
        await gateway.send_message(chat_id, "🛠 Skill listing coming soon!")
    elif intent == "help":
        await gateway.send_message(
            chat_id,
            "Commands:\n/run_job — queue a job\n/status — check run status\n/skills — list skills",
        )
    else:
        await gateway.send_message(chat_id, f"👋 Received: {message.content[:200]}")


# ── Module-level singleton ────────────────────────────────────────────────────
app = create_app()
