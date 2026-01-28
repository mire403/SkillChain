from __future__ import annotations

from typing import Any, Dict

from skillchain.agent.runtime import AgentRuntime

from .base import Workflow


class SingleStepWorkflow(Workflow):
    """Reference workflow: run exactly one router->skill step."""

    async def run(self, *, agent: AgentRuntime, user_intent: str) -> Dict[str, Any]:
        result = await agent.step(user_intent=user_intent)
        return {
            "result": result,
            "short_term": {
                "context": agent.short_term.context,
                "decisions": agent.short_term.decisions,
                "results": agent.short_term.results,
            },
        }

