"""Skill auto-synthesis from execution trajectories.

After a sufficiently complex and successful run, OmniAgent can *learn* by
generating a reusable Python skill.  The synthesis pipeline:

1. Summarise the tool chain into a function signature.
2. Ask the LLM to write the body.
3. Validate that the code is syntactically correct.
4. Persist via the SkillRegistry.

The LLM call is intentionally cheap (small prompt) — we only synthesise
skills that are likely to recur, so we accept slightly lower quality in
exchange for fast turnaround.
"""

from __future__ import annotations

import ast
import logging
import textwrap
import uuid
from datetime import datetime, timezone
from typing import Optional

from omniagent.api.models import Run, Skill

logger = logging.getLogger(__name__)

# Minimum number of tool calls before we consider synthesis worthwhile.
MIN_TOOLS_FOR_SYNTHESIS = 3


class SkillSynthesis:
    """Generate Python skills from completed, successful Run trajectories."""

    def __init__(self, llm_provider: object | None = None) -> None:
        """Accept an optional LLMProvider for integration; defaults to None (stub mode)."""
        self._provider = llm_provider

    async def synthesize(self, run: Run) -> Optional[Skill]:
        """Attempt to synthesise a Skill from *run*.

        Returns a Skill object if synthesis succeeded, or None if the run
        does not warrant a skill (too simple, failed, etc.).
        """
        if run.status != "success":
            logger.debug("Skipping synthesis: run %s status=%s", run.id, run.status)
            return None

        if len(run.tool_calls) < MIN_TOOLS_FOR_SYNTHESIS:
            logger.debug(
                "Skipping synthesis: run %s has only %d tool calls (min %d)",
                run.id,
                len(run.tool_calls),
                MIN_TOOLS_FOR_SYNTHESIS,
            )
            return None

        tool_chain = [tc.get("tool_name", "unknown") for tc in run.tool_calls]
        skill_name = self._derive_skill_name(tool_chain)
        python_code = await self._generate_code(run, skill_name)

        if not self._is_valid_python(python_code):
            logger.warning("Synthesised code for '%s' is not valid Python; discarding", skill_name)
            return None

        skill = Skill(
            id=uuid.uuid4(),
            name=skill_name,
            description=f"Auto-synthesised from run {run.id}. Tool chain: {' → '.join(tool_chain)}",
            python_code=python_code,
            input_schema={},
            output_schema={},
            version=1,
            created_at=datetime.now(timezone.utc),
        )
        logger.info("Synthesised skill '%s' from run %s", skill_name, run.id)
        return skill

    # ── private helpers ───────────────────────────────────────────────────────

    def _derive_skill_name(self, tool_chain: list[str]) -> str:
        """Build a readable snake_case name from the tool chain."""
        unique_tools = list(dict.fromkeys(tool_chain))[:3]  # first 3 unique
        return "_then_".join(t.lower().replace("-", "_") for t in unique_tools)

    async def _generate_code(self, run: Run, skill_name: str) -> str:
        """Ask the LLM to generate Python code, or fall back to a template."""
        tool_chain = [tc.get("tool_name", "unknown") for tc in run.tool_calls]

        if self._provider is not None:
            prompt = textwrap.dedent(f"""
                Write a Python async function named `{skill_name}` that reproduces
                the following tool chain: {" → ".join(tool_chain)}.

                The function should:
                - Accept **kwargs as inputs
                - Return a dict with key "output"
                - Include a docstring explaining what it does
                - Handle exceptions and return {{"output": None, "error": str(e)}} on failure

                Only output the function body — no imports, no main block.
            """)
            try:
                code = await self._provider.invoke(  # type: ignore[union-attr]
                    prompt=prompt,
                    system="You are a Python code generator. Output only valid Python code.",
                )
                return code.strip()
            except Exception as exc:  # noqa: BLE001
                logger.warning("LLM code generation failed: %s — using template", exc)

        # Template fallback
        steps = "\n        ".join(
            f"# Step {i+1}: {tool}" for i, tool in enumerate(tool_chain)
        )
        return textwrap.dedent(f"""
            async def {skill_name}(**kwargs):
                \"\"\"Auto-synthesised skill. Tool chain: {" → ".join(tool_chain)}\"\"\"
                try:
                    {steps}
                    return {{"output": None}}
                except Exception as e:
                    return {{"output": None, "error": str(e)}}
        """).strip()

    @staticmethod
    def _is_valid_python(code: str) -> bool:
        """Return True if *code* parses as valid Python."""
        try:
            ast.parse(code)
            return True
        except SyntaxError:
            return False
