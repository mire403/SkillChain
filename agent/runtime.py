from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional

from skillchain.memory.long_term import LongTermMemory
from skillchain.memory.short_term import ShortTermMemory
from skillchain.router.base import RouterContext, SkillRouter
from skillchain.skills.base import Skill, SkillContext
from skillchain.types import SkillCall, SkillResult


@dataclass(frozen=True)
class AgentConfig:
    """
    Agent runtime config.

    Keep the agent lightweight; put capability in skills and selection in routers.
    """

    agent_name: str = "skillchain-agent"


class AgentRuntime:
    """
    Agent runtime: router -> execute selected skill -> update explicit memory.
    """

    def __init__(
        self,
        *,
        router: SkillRouter,
        skills: Iterable[Skill],
        short_term: Optional[ShortTermMemory] = None,
        long_term: Optional[LongTermMemory] = None,
        config: Optional[AgentConfig] = None,
    ) -> None:
        self._router = router
        self._skills: Dict[str, Skill] = {s.name: s for s in skills}
        self.short_term = short_term or ShortTermMemory()
        self.long_term = long_term
        self.config = config or AgentConfig()

    def list_skills(self) -> Dict[str, str]:
        return {name: getattr(skill, "description", "") for name, skill in self._skills.items()}

    async def step(self, *, user_intent: str) -> SkillResult:
        available = [{"name": n, "description": d} for n, d in self.list_skills().items()]
        router_ctx = RouterContext(
            short_term_context=self.short_term.context,
            long_term_summary=self.long_term.get_summary() if self.long_term else None,
        )
        call: SkillCall = await self._router.select_skill(
            user_intent=user_intent,
            available_skills=available,
            ctx=router_ctx,
        )
        self.short_term.record_decision(call)

        skill = self._skills.get(call.skill_name)
        if not skill:
            result = SkillResult(
                skill_name=call.skill_name,
                output={},
                success=False,
                error=f"Unknown skill selected by router: {call.skill_name}",
            )
            self.short_term.record_result(result)
            return result

        ctx = SkillContext(request_id=str(uuid.uuid4()), short_term_context=self.short_term.context)
        try:
            output = skill.run(skill_input=call.skill_input, ctx=ctx)
            result = SkillResult(skill_name=skill.name, output=output, success=True)
        except Exception as e:  # keep framework ergonomic; callers can harden this
            result = SkillResult(skill_name=skill.name, output={}, success=False, error=str(e))

        self.short_term.record_result(result)
        return result

