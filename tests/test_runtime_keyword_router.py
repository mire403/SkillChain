from __future__ import annotations

import asyncio

from skillchain.agent.runtime import AgentRuntime
from skillchain.router.rule_based import KeywordRouter
from skillchain.skills.echo import EchoSkill


def test_agent_runtime_runs_echo_via_keyword_router() -> None:
    async def run() -> dict:
        agent = AgentRuntime(router=KeywordRouter(default_skill="echo"), skills=[EchoSkill()])
        result = await agent.step(user_intent="please echo this")
        assert result.success is True
        assert result.skill_name == "echo"
        assert result.output["echo"]["text"] == "please echo this"
        return result.output

    asyncio.run(run())

