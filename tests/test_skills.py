"""Unit tests for the skill system — registry, synthesis, and memory."""

from __future__ import annotations

import uuid
from pathlib import Path
from textwrap import dedent
from typing import Any

import pytest

from omniagent.api.models import Run
from omniagent.core.memory import LongTermMemory, ShortTermMemory
from omniagent.core.skill import SkillRegistry
from omniagent.skills.synthesis import SkillSynthesis


# ── ShortTermMemory tests ─────────────────────────────────────────────────────


def test_short_term_memory_record_and_retrieve() -> None:
    mem = ShortTermMemory(maxlen=5)
    mem.record({"tool_name": "web_search"}, {"output": "results"})
    mem.record({"tool_name": "csv_writer"}, {"output": "file.csv"})

    hits = mem.get_relevant("web_search")
    assert len(hits) == 1
    assert hits[0]["tool_call"]["tool_name"] == "web_search"


def test_short_term_memory_maxlen() -> None:
    mem = ShortTermMemory(maxlen=3)
    for i in range(5):
        mem.record({"tool_name": f"tool_{i}"}, {"output": i})

    # Only last 3 entries should remain
    assert len(mem.history) == 3
    # Oldest entries evicted — last entry should be tool_4
    last = list(mem.history)[-1]
    assert last["tool_call"]["tool_name"] == "tool_4"


def test_short_term_memory_to_dict() -> None:
    mem = ShortTermMemory()
    mem.current_context = {"job": "test"}
    mem.record({"tool_name": "ping"}, {"output": "pong"})

    d = mem.to_dict()
    assert "history" in d
    assert "current_context" in d
    assert d["current_context"]["job"] == "test"


def test_short_term_memory_clear() -> None:
    mem = ShortTermMemory()
    mem.record({"tool_name": "t"}, {"output": "x"})
    mem.clear()
    assert len(mem.history) == 0
    assert mem.current_context == {}


# ── LongTermMemory tests ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_long_term_memory_promote_and_query() -> None:
    ltm = LongTermMemory()

    run_data = {
        "id": str(uuid.uuid4()),
        "status": "success",
        "tool_calls": [
            {"tool_name": "selenium"},
            {"tool_name": "csv_writer"},
        ],
    }
    await ltm.promote(run_data)

    results = await ltm.query_patterns("selenium")
    assert len(results) == 1
    assert "selenium" in results[0]["tool_chain"]


@pytest.mark.asyncio
async def test_long_term_memory_no_match() -> None:
    ltm = LongTermMemory()
    await ltm.promote({
        "id": "abc",
        "status": "success",
        "tool_calls": [{"tool_name": "echo"}],
    })

    results = await ltm.query_patterns("completely_unrelated")
    assert results == []


# ── SkillRegistry tests ───────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_skill_registry_discover_empty(tmp_path: Path) -> None:
    """An empty skills directory should yield zero skills."""
    registry = SkillRegistry(skills_dir=tmp_path)
    skills = await registry.discover()
    assert skills == []


@pytest.mark.asyncio
async def test_skill_registry_loads_skill(tmp_path: Path) -> None:
    """A properly structured skill module should be discovered and executable."""
    skill_code = dedent("""
        from omniagent.api.models import Skill
        import uuid

        SKILL = Skill(
            id=uuid.uuid4(),
            name="test_skill",
            description="A test skill",
            python_code="def run(**kwargs): return {'output': 'ok'}",
        )

        def run(**kwargs):
            return {"output": "ok"}
    """)
    (tmp_path / "test_skill.py").write_text(skill_code)

    registry = SkillRegistry(skills_dir=tmp_path)
    skills = await registry.discover()

    assert len(skills) == 1
    assert skills[0].name == "test_skill"


@pytest.mark.asyncio
async def test_skill_registry_execute(tmp_path: Path) -> None:
    skill_code = dedent("""
        from omniagent.api.models import Skill
        import uuid

        SKILL = Skill(
            id=uuid.uuid4(),
            name="adder_skill",
            description="Adds two numbers",
            python_code="",
        )

        def run(**kwargs):
            a = kwargs.get("a", 0)
            b = kwargs.get("b", 0)
            return {"output": a + b}
    """)
    (tmp_path / "adder_skill.py").write_text(skill_code)

    registry = SkillRegistry(skills_dir=tmp_path)
    await registry.discover()

    skill = await registry.load("adder_skill")
    assert skill is not None

    result = await registry.execute(skill, {"a": 3, "b": 4})
    assert result["output"] == 7
    assert result["error"] is None


# ── SkillSynthesis tests ──────────────────────────────────────────────────────


def make_run(status: str = "success", n_tools: int = 4) -> Run:
    tool_calls = [{"tool_name": f"tool_{i}", "input_args": {}} for i in range(n_tools)]
    return Run(
        id=uuid.uuid4(),
        job_id=uuid.uuid4(),
        agent_id=uuid.uuid4(),
        status=status,
        tool_calls=tool_calls,
    )


@pytest.mark.asyncio
async def test_synthesis_skips_failed_run() -> None:
    synth = SkillSynthesis()
    run = make_run(status="failed", n_tools=5)
    skill = await synth.synthesize(run)
    assert skill is None


@pytest.mark.asyncio
async def test_synthesis_skips_simple_run() -> None:
    """Runs with fewer tool calls than MIN_TOOLS_FOR_SYNTHESIS are skipped."""
    synth = SkillSynthesis()
    run = make_run(status="success", n_tools=1)
    skill = await synth.synthesize(run)
    assert skill is None


@pytest.mark.asyncio
async def test_synthesis_generates_skill() -> None:
    synth = SkillSynthesis()
    run = make_run(status="success", n_tools=4)
    skill = await synth.synthesize(run)

    assert skill is not None
    assert skill.name  # name was derived from tool chain
    assert skill.python_code  # code was generated
    assert skill.description  # description was set


@pytest.mark.asyncio
async def test_synthesis_with_llm_provider() -> None:
    """When an LLM provider is available, it should be used for code generation."""
    from omniagent.inference.providers import LLMProvider

    class FakeProvider(LLMProvider):
        async def invoke(self, prompt: str, **kwargs: Any) -> str:
            return "async def generated_skill(**kwargs):\n    return {'output': 'llm'}\n"

    synth = SkillSynthesis(llm_provider=FakeProvider())
    run = make_run(status="success", n_tools=4)
    skill = await synth.synthesize(run)

    assert skill is not None
    assert "generated_skill" in skill.python_code
