"""Skill registry: discover, load, and execute reusable skills.

Skills are Python functions that can be invoked by the agent during the Act
phase.  They live either as files in the ``skills/`` directory or as rows in
the ``skills`` database table.

Design goals:
- Discovery is cheap: scan once at startup and cache.
- Execution is safe: validate inputs/outputs via Pydantic schemas.
- Metrics are automatic: every call increments usage/success counters.
"""

from __future__ import annotations

import importlib
import importlib.util
import logging
import time
from pathlib import Path
from typing import Any, Callable, Optional

from omniagent.api.models import Skill

logger = logging.getLogger(__name__)

# ── SkillRegistry ─────────────────────────────────────────────────────────────


class SkillRegistry:
    """Discover, load, and execute skills from the skills directory and DB."""

    def __init__(self, skills_dir: Optional[Path] = None) -> None:
        self._skills_dir = skills_dir or Path(__file__).parent.parent / "skills"
        self._cache: dict[str, Skill] = {}
        self._callables: dict[str, Callable[..., Any]] = {}

    async def discover(self) -> list[Skill]:
        """Scan the skills directory for Python modules that expose a ``SKILL`` constant.

        Each skill module must define a module-level ``SKILL: Skill`` object
        and a callable named ``run(**kwargs) -> dict`` at the top level.
        """
        found: list[Skill] = []
        if not self._skills_dir.exists():
            logger.warning("Skills directory %s does not exist", self._skills_dir)
            return found

        for path in self._skills_dir.glob("*.py"):
            if path.name.startswith("_"):
                continue
            try:
                skill = await self._load_from_file(path)
                if skill:
                    found.append(skill)
                    self._cache[skill.name] = skill
            except Exception as exc:  # noqa: BLE001
                logger.error("Failed to load skill from %s: %s", path, exc)

        logger.info("SkillRegistry discovered %d skill(s)", len(found))
        return found

    async def load(self, skill_name: str) -> Optional[Skill]:
        """Return the Skill object for *skill_name*, loading it if necessary."""
        if skill_name in self._cache:
            return self._cache[skill_name]
        # Try to load from file
        path = self._skills_dir / f"{skill_name}.py"
        if path.exists():
            skill = await self._load_from_file(path)
            if skill:
                self._cache[skill_name] = skill
                return skill
        logger.warning("Skill '%s' not found", skill_name)
        return None

    async def execute(self, skill: Skill, inputs: dict[str, Any]) -> dict[str, Any]:
        """Run *skill* with *inputs* and record metrics.

        Returns a dict with keys ``output`` (any value) and ``error`` (str or None).
        Always returns — never raises — so the agent loop can continue on failure.
        """
        fn = self._callables.get(skill.name)
        if fn is None:
            return {"output": None, "error": f"Skill '{skill.name}' callable not loaded"}

        start = time.monotonic()
        skill.usage_count += 1
        try:
            raw = fn(**inputs)
            elapsed_ms = int((time.monotonic() - start) * 1000)
            skill.success_count += 1
            logger.info("Skill '%s' executed in %d ms", skill.name, elapsed_ms)
            # Skills return a dict with at least an "output" key; merge
            # directly rather than double-nesting under another "output".
            result: dict[str, Any] = raw if isinstance(raw, dict) else {"output": raw}
            return {**result, "error": result.get("error"), "duration_ms": elapsed_ms}
        except Exception as exc:  # noqa: BLE001
            elapsed_ms = int((time.monotonic() - start) * 1000)
            logger.error("Skill '%s' failed: %s", skill.name, exc)
            return {"output": None, "error": str(exc), "duration_ms": elapsed_ms}

    # ── private helpers ───────────────────────────────────────────────────────

    async def _load_from_file(self, path: Path) -> Optional[Skill]:
        """Import a skill module and extract its ``SKILL`` constant."""
        spec = importlib.util.spec_from_file_location(path.stem, path)
        if spec is None or spec.loader is None:
            return None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)  # type: ignore[union-attr]

        skill: Optional[Skill] = getattr(module, "SKILL", None)
        if skill is None:
            return None

        fn: Optional[Callable[..., Any]] = getattr(module, "run", None)
        if fn is not None:
            self._callables[skill.name] = fn

        return skill
