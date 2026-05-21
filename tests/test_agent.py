"""Unit tests for the core Agent loop and Run state machine."""

from __future__ import annotations

import uuid
from typing import Any, Optional

import pytest

from omniagent.api.models import Job, Run
from omniagent.core.agent import Agent
from omniagent.core.run import (
    can_retry,
    compute_complexity,
    is_terminal,
    record_retry,
    transition,
)
from omniagent.inference.providers import LLMProvider


# ── Fixtures ──────────────────────────────────────────────────────────────────


class StubProvider(LLMProvider):
    """Minimal LLM provider for tests — returns a configurable string."""

    def __init__(self, response: str = "Hello from stub") -> None:
        self._response = response

    async def invoke(
        self,
        prompt: str,
        system: str = "",
        model: str = "",
        tools: Optional[list[dict]] = None,
        **kwargs: Any,
    ) -> str:
        return self._response


def make_job(**kwargs: Any) -> Job:
    defaults = {
        "id": uuid.uuid4(),
        "name": "test_job",
        "agent_id": uuid.uuid4(),
        "status": "active",
    }
    defaults.update(kwargs)
    return Job(**defaults)


def make_run(**kwargs: Any) -> Run:
    defaults = {
        "id": uuid.uuid4(),
        "job_id": uuid.uuid4(),
        "agent_id": uuid.uuid4(),
        "status": "pending",
    }
    defaults.update(kwargs)
    return Run(**defaults)


# ── Run state-machine tests ────────────────────────────────────────────────────


def test_transition_valid() -> None:
    run = make_run(status="pending")
    updated = transition(run, "executing")
    assert updated.status == "executing"


def test_transition_invalid_raises() -> None:
    run = make_run(status="success")  # terminal state
    with pytest.raises(ValueError, match="Cannot transition"):
        transition(run, "executing")


def test_is_terminal() -> None:
    assert is_terminal(make_run(status="success"))
    # A failed run with no remaining retries is terminal
    assert is_terminal(make_run(status="failed", agent_state={"retry_count": 3}))
    # A failed run that can still be retried is NOT terminal
    assert not is_terminal(make_run(status="failed", agent_state={"retry_count": 0}))
    assert not is_terminal(make_run(status="executing"))


def test_can_retry_under_limit() -> None:
    run = make_run(status="failed", agent_state={"retry_count": 1})
    assert can_retry(run, max_retries=3)


def test_can_retry_at_limit() -> None:
    run = make_run(status="failed", agent_state={"retry_count": 3})
    assert not can_retry(run, max_retries=3)


def test_record_retry_increments() -> None:
    run = make_run(status="failed", agent_state={"retry_count": 0})
    retried = record_retry(run)
    assert retried.agent_state["retry_count"] == 1
    assert retried.status == "retrying"


def test_compute_complexity_zero() -> None:
    assert compute_complexity([]) == 0.0


def test_compute_complexity_scaled() -> None:
    tool_calls = [{"tool_name": f"tool_{i}"} for i in range(5)]
    score = compute_complexity(tool_calls)
    assert 0 < score <= 1.0


# ── Agent loop tests ───────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_agent_run_success_no_tools() -> None:
    """Agent should complete a run even when the LLM returns no tool calls."""
    provider = StubProvider(response="Task completed successfully.")
    agent = Agent(provider=provider)
    job = make_job(name="simple_task")

    run = await agent.run(job)

    assert run.status == "success"
    assert run.output == "Task completed successfully."
    assert run.job_id == job.id
    assert run.duration_ms >= 0


@pytest.mark.asyncio
async def test_agent_run_parses_tool_call() -> None:
    """Agent should parse a JSON tool call and record it in tool_calls."""
    response = 'Calling: {"tool": "echo", "args": {"message": "hello"}}'
    provider = StubProvider(response=response)
    agent = Agent(provider=provider)
    job = make_job(name="tool_task")

    run = await agent.run(job)

    assert run.status == "success"
    assert len(run.tool_calls) == 1
    assert run.tool_calls[0]["tool_name"] == "echo"


@pytest.mark.asyncio
async def test_agent_run_handles_provider_error() -> None:
    """Agent should catch LLM errors and mark the run as failed."""

    class ErrorProvider(LLMProvider):
        async def invoke(self, prompt: str, **kwargs: Any) -> str:
            raise RuntimeError("LLM unreachable")

    agent = Agent(provider=ErrorProvider())
    job = make_job(name="failing_task")

    run = await agent.run(job)

    assert run.status == "failed"
    assert run.error is not None
    assert "LLM unreachable" in run.error


@pytest.mark.asyncio
async def test_agent_hydrates_context_injections() -> None:
    """Context injections should be merged into the prompt."""
    captured_prompts: list[str] = []

    class CapturingProvider(LLMProvider):
        async def invoke(self, prompt: str, **kwargs: Any) -> str:
            captured_prompts.append(prompt)
            return "done"

    agent = Agent(provider=CapturingProvider())
    job = make_job(
        name="context_task",
        context_injections=[{"env_var": "PROD", "doc": "Sales report Q1"}],
    )
    run = await agent.run(job)

    assert run.status == "success"
    assert captured_prompts, "Provider was never invoked"
    assert "Sales report Q1" in captured_prompts[0]
