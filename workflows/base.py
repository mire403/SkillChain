from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict

from skillchain.agent.runtime import AgentRuntime


class Workflow(ABC):
    """
    High-level multi-step strategy.

    Workflows orchestrate agents and skills; they do not replace them.
    """

    @abstractmethod
    async def run(self, *, agent: AgentRuntime, user_intent: str) -> Dict[str, Any]:
        raise NotImplementedError

