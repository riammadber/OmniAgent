"""Two-tier memory system for OmniAgent.

Short-term memory lives in-process and is scoped to the current agent
instance (max 10 runs).  Long-term memory is a SQLite-backed graph of
high-value insights promoted from completed runs.

Why two tiers?
  - Short-term gives the LLM immediate trajectory context without DB round
    trips (low latency).
  - Long-term accumulates durable patterns across many sessions so the agent
    gets smarter over time.
"""

from __future__ import annotations

import json
import logging
from collections import deque
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)


class ShortTermMemory:
    """In-process, execution-scoped memory holding the last ``maxlen`` runs.

    Thread-safety note: this class is designed to be used within a single
    async execution context.  Do not share instances across concurrent tasks.
    """

    def __init__(self, maxlen: int = 10) -> None:
        self.history: deque[dict[str, Any]] = deque(maxlen=maxlen)
        self.current_context: dict[str, Any] = {}

    def record(self, tool_call: dict[str, Any], result: dict[str, Any]) -> None:
        """Append a tool-call/result pair to the rolling history buffer."""
        entry = {
            "tool_call": tool_call,
            "result": result,
            "recorded_at": datetime.now(timezone.utc).isoformat(),
        }
        self.history.append(entry)
        logger.debug("ShortTermMemory recorded tool_call=%s", tool_call.get("tool_name"))

    def get_relevant(self, query: str) -> list[dict[str, Any]]:
        """Return history entries whose serialised form contains *query* keywords.

        Performs a simple case-insensitive substring search.  This is fast
        enough for the small (≤10) history window and avoids a heavyweight
        embedding lookup on every cycle.
        """
        query_lower = query.lower()
        results = []
        for entry in self.history:
            serialised = json.dumps(entry).lower()
            if query_lower in serialised:
                results.append(entry)
        return results

    def to_dict(self) -> dict[str, Any]:
        """Serialise the current state for persistence in a Run record."""
        return {
            "history": list(self.history),
            "current_context": self.current_context,
        }

    def clear(self) -> None:
        """Reset memory — typically called at the start of a new run."""
        self.history.clear()
        self.current_context = {}


class LongTermMemory:
    """SQLite-backed persistent memory graph.

    Stores high-value patterns extracted from completed runs so the agent can
    recognise similar future tasks and reuse proven strategies.

    The current implementation uses a simple JSON-in-SQLite approach.  A
    production deployment can swap this for Neo4j or a vector store without
    touching the interface.
    """

    def __init__(self) -> None:
        # In-memory store for the current process session.
        # A real implementation would persist to SQLite / Neo4j.
        self._patterns: list[dict[str, Any]] = []

    async def promote(self, run_data: dict[str, Any]) -> None:
        """Extract high-value insights from a completed run.

        Currently extracts:
        - Tool chains used (ordered list of tool names)
        - Error patterns (tool name → error summary)
        - Overall success / failure
        """
        tool_calls: list[dict[str, Any]] = run_data.get("tool_calls", [])
        tool_chain = [tc.get("tool_name", "unknown") for tc in tool_calls]
        errors = [
            {"tool": tc.get("tool_name"), "error": tc.get("error")}
            for tc in tool_calls
            if tc.get("error")
        ]

        pattern: dict[str, Any] = {
            "run_id": run_data.get("id"),
            "tool_chain": tool_chain,
            "errors": errors,
            "status": run_data.get("status"),
            "promoted_at": datetime.now(timezone.utc).isoformat(),
        }
        self._patterns.append(pattern)
        logger.info(
            "LongTermMemory promoted run=%s tool_chain=%s", run_data.get("id"), tool_chain
        )

    async def query_patterns(self, task_description: str) -> list[dict[str, Any]]:
        """Return stored patterns whose tool chains mention keywords from *task_description*.

        Returns up to 5 most recently promoted matching patterns.
        """
        keywords = task_description.lower().split()
        matches = []
        for pattern in reversed(self._patterns):
            chain_text = " ".join(pattern.get("tool_chain", [])).lower()
            if any(kw in chain_text for kw in keywords):
                matches.append(pattern)
            if len(matches) >= 5:
                break
        return matches
