"""Human-in-the-loop (HITL) approval stub.

Sensitive tool calls (e.g. file deletion, money transfers) can be routed
through this module to require explicit human approval before execution.

The current implementation is a *stub* that always approves — integrate
with your preferred approval channel (Telegram inline keyboard, Slack
interactive message, etc.) by replacing ``request_approval``.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


async def request_approval(
    tool_name: str,
    input_args: dict[str, Any],
    user_id: str,
) -> bool:
    """Ask a human to approve a sensitive tool call.

    Parameters
    ----------
    tool_name:
        The tool that is about to be executed.
    input_args:
        Arguments that will be passed to the tool.
    user_id:
        The platform user ID to notify.

    Returns
    -------
    bool
        True if the user approved, False if rejected / timed out.
    """
    # STUB: log and auto-approve.  Replace with a real notification flow.
    logger.warning(
        "HITL (stub): auto-approving tool='%s' args=%s for user='%s'",
        tool_name,
        input_args,
        user_id,
    )
    return True
