"""Run state machine.

A Run progresses through well-defined states:

    pending → executing → success
                       ↘ failed
                       ↘ retrying → executing (loop)

This module provides helpers to manage those transitions without embedding
state logic inside the agent loop or routers.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

from omniagent.api.models import Run

logger = logging.getLogger(__name__)

# Valid state transitions: current_state → set of allowed next states
_TRANSITIONS: dict[str, set[str]] = {
    "pending": {"executing"},
    "executing": {"success", "failed", "retrying"},
    "retrying": {"executing", "failed"},
    "failed": {"retrying"},  # a failed run may be retried
    "success": set(),        # terminal
}


def transition(run: Run, new_status: str) -> Run:
    """Return a *copy* of *run* with ``status`` set to *new_status*.

    Raises ``ValueError`` if the transition is not allowed, preventing
    the run from entering an inconsistent state.
    """
    allowed = _TRANSITIONS.get(run.status, set())
    if new_status not in allowed:
        raise ValueError(
            f"Cannot transition run {run.id} from '{run.status}' to '{new_status}'. "
            f"Allowed transitions: {allowed or 'none (terminal state)'}"
        )
    updated = run.model_copy(update={"status": new_status})
    if new_status in ("success", "failed"):
        updated = updated.model_copy(update={"completed_at": datetime.now(timezone.utc)})
    logger.debug("Run %s transitioned %s → %s", run.id, run.status, new_status)
    return updated


def is_terminal(run: Run) -> bool:
    """Return True if the run has reached a terminal state."""
    return run.status == "success" or (run.status == "failed" and not can_retry(run))


def can_retry(run: Run, max_retries: int = 3) -> bool:
    """Return True if the run can be retried.

    We track retry count via the ``agent_state`` dict to avoid adding a DB
    column.  The ``max_retries`` default of 3 follows common retry wisdom.
    """
    retries: int = run.agent_state.get("retry_count", 0)
    return run.status in ("failed", "retrying") and retries < max_retries


def record_retry(run: Run) -> Run:
    """Increment the retry counter and transition the run to 'retrying'."""
    state = dict(run.agent_state)
    state["retry_count"] = state.get("retry_count", 0) + 1
    updated = run.model_copy(update={"agent_state": state})
    return transition(updated, "retrying")


def compute_complexity(tool_calls: list[dict], error_count: Optional[int] = None) -> float:
    """Heuristic complexity score for a run's trajectory.

    Used to decide whether skill synthesis is worth the cost.
    Score is normalised to [0, 1].
    """
    n_tools = len(tool_calls)
    n_errors = error_count if error_count is not None else sum(
        1 for tc in tool_calls if tc.get("error")
    )
    # Simple linear heuristic: weight tool count and error recovery
    raw = n_tools * 0.1 + n_errors * 0.2
    return min(raw, 1.0)
