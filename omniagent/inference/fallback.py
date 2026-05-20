"""Provider failover logic.

When the primary LLM provider fails (rate-limit, network error, etc.),
``FallbackProvider`` tries each configured backup in order.  This makes
the agent resilient to transient outages without any changes to calling code.
"""

from __future__ import annotations

import logging
from typing import Optional

from omniagent.inference.providers import LLMProvider

logger = logging.getLogger(__name__)


class FallbackProvider(LLMProvider):
    """Try providers in order, returning the first successful response.

    Parameters
    ----------
    providers:
        Ordered list of ``LLMProvider`` instances.  At least one must be
        supplied.  The first provider is the primary; the rest are fallbacks.
    """

    def __init__(self, providers: list[LLMProvider]) -> None:
        if not providers:
            raise ValueError("FallbackProvider requires at least one provider")
        self._providers = providers

    async def invoke(
        self,
        prompt: str,
        system: str = "",
        model: str = "",
        tools: Optional[list[dict]] = None,
        **kwargs: object,
    ) -> str:
        """Invoke providers in order; return the first successful response."""
        last_error: Exception | None = None
        for idx, provider in enumerate(self._providers):
            try:
                result = await provider.invoke(
                    prompt, system=system, model=model, tools=tools, **kwargs
                )
                if idx > 0:
                    logger.info(
                        "FallbackProvider succeeded on provider #%d (%s)",
                        idx,
                        type(provider).__name__,
                    )
                return result
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Provider #%d (%s) failed: %s — trying next",
                    idx,
                    type(provider).__name__,
                    exc,
                )
                last_error = exc

        raise RuntimeError(
            f"All {len(self._providers)} LLM providers failed. "
            f"Last error: {last_error}"
        ) from last_error
