"""Main Agent class — Think-Act-Observe loop.

The agent is the execution heart of OmniAgent.  Each invocation of ``run``
processes a single Job through four phases:

  1. **Think** — Hydrate context (docs, env vars, memory) and build the LLM prompt.
  2. **Act**   — Invoke the LLM, parse tool calls from the response.
  3. **Observe** — Execute each tool call, record results to short-term memory.
  4. **Reflect** — If the trajectory was complex, synthesise a reusable skill.

All I/O is async so the agent can be embedded inside FastAPI or run standalone.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from datetime import datetime, timezone

from omniagent.api.models import Job, Run
from omniagent.core.memory import LongTermMemory, ShortTermMemory
from omniagent.core.run import compute_complexity, transition
from omniagent.core.skill import SkillRegistry
from omniagent.inference.providers import LLMProvider
from omniagent.skills.synthesis import SkillSynthesis

logger = logging.getLogger(__name__)


class Agent:
    """Autonomous agent that executes Jobs via the Think-Act-Observe loop.

    Parameters
    ----------
    provider:
        LLM backend to use for reasoning and code generation.
    system_prompt:
        Override the default system instruction for this agent.
    skill_registry:
        Optional pre-configured SkillRegistry.  A new instance is created
        if not provided.
    """

    def __init__(
        self,
        provider: LLMProvider,
        system_prompt: str = "",
        skill_registry: Optional[SkillRegistry] = None,
    ) -> None:
        self._provider = provider
        self._system = system_prompt or (
            "You are OmniAgent, a reliable autonomous AI assistant. "
            "When you need to call a tool, output a JSON block like:\n"
            '{"tool": "<tool_name>", "args": {...}}'
        )
        self._registry = skill_registry or SkillRegistry()
        self._short_term = ShortTermMemory()
        self._long_term = LongTermMemory()
        self._synthesis = SkillSynthesis(llm_provider=provider)

    async def run(self, job: Job, context_injections: dict[str, Any] | None = None) -> Run:
        """Execute *job* and return a fully realised Run.

        Progresses through Think → Act → Observe → Reflect and captures the
        entire trajectory so the control plane can persist it.
        """
        run = Run(
            id=uuid.uuid4(),
            job_id=job.id,
            agent_id=job.agent_id,
            status="pending",
            created_at=datetime.now(timezone.utc),
        )

        start_ms = time.monotonic()
        try:
            run = transition(run, "executing")

            # ── 1. Think ──────────────────────────────────────────────────────
            context = await self._hydrate_context(job, context_injections or {})
            prompt = self._synthesize_prompt(context, job.name)
            logger.info("Agent thinking for job '%s'", job.name)

            # ── 2. Act ────────────────────────────────────────────────────────
            llm_response = await self._provider.invoke(
                prompt=prompt,
                system=self._system,
            )
            tool_calls_raw = self._parse_tool_calls(llm_response)
            logger.info("Agent parsed %d tool call(s) for job '%s'", len(tool_calls_raw), job.name)

            # ── 3. Observe ────────────────────────────────────────────────────
            executed_tool_calls: list[dict[str, Any]] = []
            for tc in tool_calls_raw:
                result = await self._execute_tool(tc, context)
                self._short_term.record(tc, result)
                executed_tool_calls.append({**tc, **result})

            # ── 4. Reflect ────────────────────────────────────────────────────
            complexity = compute_complexity(executed_tool_calls)
            output = llm_response

            run = run.model_copy(
                update={
                    "tool_calls": executed_tool_calls,
                    "short_term_memory": self._short_term.to_dict(),
                    "output": output,
                    "duration_ms": int((time.monotonic() - start_ms) * 1000),
                }
            )
            run = transition(run, "success")

            if complexity >= 0.4:
                skill = await self._synthesis.synthesize(run)
                if skill:
                    logger.info("New skill synthesised: '%s'", skill.name)

            await self._long_term.promote(run.model_dump())

        except Exception as exc:  # noqa: BLE001
            logger.exception("Agent run failed for job '%s': %s", job.name, exc)
            run = run.model_copy(
                update={
                    "error": str(exc),
                    "duration_ms": int((time.monotonic() - start_ms) * 1000),
                    "status": "failed",
                    "completed_at": datetime.now(timezone.utc),
                }
            )

        return run

    # ── Think helpers ─────────────────────────────────────────────────────────

    async def _hydrate_context(
        self, job: Job, context_injections: dict[str, Any]
    ) -> dict[str, Any]:
        """Merge job-level and call-level context injections.

        In production this would fetch Markdown docs from URLs, decrypt env
        vars, or snapshot SQLite tables.  The current implementation merges
        the dicts and appends relevant short-term memory.
        """
        context: dict[str, Any] = {}
        for injection in job.context_injections:
            context.update(injection)
        context.update(context_injections)

        # Append relevant short-term memory for the job name
        relevant = self._short_term.get_relevant(job.name)
        if relevant:
            context["memory_hints"] = relevant

        return context

    def _synthesize_prompt(self, context: dict[str, Any], job_name: str) -> str:
        """Build the full LLM prompt by combining context and job description.

        Keeps the prompt readable so the LLM can reason clearly.
        """
        parts: list[str] = [f"# Task: {job_name}"]

        if context:
            parts.append("\n## Context")
            for key, value in context.items():
                parts.append(f"**{key}**: {json.dumps(value, default=str)[:500]}")

        parts.append(
            "\n## Instructions\n"
            "Complete the task above. If you need to call tools, output one JSON block "
            'per tool call: {"tool": "<name>", "args": {...}}'
        )
        return "\n".join(parts)

    # ── Act helpers ───────────────────────────────────────────────────────────

    def _parse_tool_calls(self, response: str) -> list[dict[str, Any]]:
        """Extract tool call JSON objects from the LLM response.

        Uses a brace-depth scan so nested JSON values (e.g. ``"args": {...}``)
        are handled correctly.  Supports both fenced code blocks and bare JSON.
        """
        tool_calls: list[dict[str, Any]] = []

        for raw in self._extract_json_objects(response):
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, dict) and "tool" in parsed:
                    tool_calls.append(
                        {
                            "tool_name": parsed["tool"],
                            "input_args": parsed.get("args", {}),
                        }
                    )
            except json.JSONDecodeError:
                continue

        return tool_calls

    @staticmethod
    def _extract_json_objects(text: str) -> list[str]:
        """Return all top-level JSON object strings found in *text*.

        Tracks brace depth so nested ``{...}`` inside values are included
        in the same object rather than treated as separate candidates.
        """
        results: list[str] = []
        depth = 0
        start = -1
        for i, ch in enumerate(text):
            if ch == "{":
                if depth == 0:
                    start = i
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0 and start != -1:
                    results.append(text[start : i + 1])
                    start = -1
        return results

    # ── Observe helpers ───────────────────────────────────────────────────────

    async def _execute_tool(
        self, tool_call: dict[str, Any], context: dict[str, Any]
    ) -> dict[str, Any]:
        """Dispatch a single tool call and return its result dict.

        First checks the SkillRegistry for a matching skill; falls back to
        a no-op so the agent never hard-crashes on an unknown tool.
        """
        tool_name: str = tool_call.get("tool_name", "")
        input_args: dict[str, Any] = tool_call.get("input_args", {})

        start = time.monotonic()
        skill = await self._registry.load(tool_name)

        if skill:
            result = await self._registry.execute(skill, input_args)
        else:
            logger.warning("Tool '%s' not found in registry; returning stub", tool_name)
            result = {
                "output": f"[stub] Tool '{tool_name}' not implemented",
                "error": None,
            }

        result["duration_ms"] = int((time.monotonic() - start) * 1000)
        return result
