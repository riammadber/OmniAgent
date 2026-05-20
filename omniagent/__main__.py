"""CLI entry point for OmniAgent.

Commands:
  omniagent server       — Start the FastAPI control plane
  omniagent run-job      — Execute a job locally (for testing / debugging)

The async commands use asyncio.run() so Click's sync-first design
stays intact while we keep all agent code fully async.
"""

from __future__ import annotations

import asyncio
import logging
import sys

import click
import uvicorn

from omniagent.config import settings

logging.basicConfig(
    level=logging.DEBUG if settings.debug else logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)


@click.group()
def cli() -> None:
    """OmniAgent — autonomous agent framework CLI."""


@cli.command()
def server() -> None:
    """Start the OmniAgent control plane server."""
    click.echo(f"Starting OmniAgent on {settings.api_host}:{settings.api_port} …")
    uvicorn.run(
        "omniagent.api.server:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug,
        log_level="debug" if settings.debug else "info",
    )


@cli.command("run-job")
@click.option("--job-name", required=True, help="Name of the job to execute")
@click.option("--agent-id", required=True, help="Agent UUID to run the job as")
def run_job(job_name: str, agent_id: str) -> None:
    """Execute a job locally — useful for smoke-testing without a server."""
    asyncio.run(_run_job_async(job_name, agent_id))


async def _run_job_async(job_name: str, agent_id: str) -> None:
    """Async implementation of the run-job command."""
    import uuid

    from omniagent.api.models import Job
    from omniagent.core.agent import Agent
    from omniagent.inference.providers import get_provider

    click.echo(f"Running job '{job_name}' as agent '{agent_id}' …")

    provider = get_provider()
    agent = Agent(provider=provider)

    job = Job(
        id=uuid.uuid4(),
        name=job_name,
        agent_id=uuid.UUID(agent_id),
        status="active",
    )

    run = await agent.run(job)
    click.echo(f"Run completed: status={run.status}")
    if run.output:
        click.echo(f"Output:\n{run.output[:500]}")
    if run.error:
        click.echo(f"Error: {run.error}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    cli()
