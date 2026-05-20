"""Web search skill powered by Tavily."""

from __future__ import annotations

from typing import Any
from uuid import uuid4

import httpx

from omniagent.api.models import Skill
from omniagent.config import settings

SKILL = Skill(
    id=uuid4(),
    name="web_search",
    description="Search the web using Tavily and return concise, structured results.",
    python_code=(
        "def run(query: str, max_results: int = 5, include_answer: bool = True, "
        "search_depth: str = 'basic') -> dict: ..."
    ),
)


def run(
    query: str,
    max_results: int = 5,
    include_answer: bool = True,
    search_depth: str = "basic",
) -> dict[str, Any]:
    """Execute a web search through Tavily.

    Parameters
    ----------
    query:
        Search query string.
    max_results:
        Maximum number of result items to return.
    include_answer:
        Whether Tavily should include a synthesized answer.
    search_depth:
        Tavily search depth; commonly "basic" or "advanced".
    """
    if not query.strip():
        return {"output": None, "error": "query cannot be empty"}

    api_key = settings.tavily_api_key
    if not api_key:
        return {
            "output": None,
            "error": "TAVILY_API_KEY is not configured",
        }

    payload = {
        "api_key": api_key,
        "query": query,
        "max_results": max(1, min(max_results, 10)),
        "include_answer": include_answer,
        "search_depth": search_depth,
    }

    try:
        response = httpx.post(settings.tavily_base_url, json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()

        normalized_results = [
            {
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "content": item.get("content", ""),
                "score": item.get("score", 0),
            }
            for item in data.get("results", [])
        ]

        return {
            "output": {
                "query": query,
                "answer": data.get("answer", ""),
                "results": normalized_results,
            },
            "error": None,
        }
    except httpx.HTTPError as exc:
        return {
            "output": None,
            "error": f"Tavily request failed: {exc}",
        }
