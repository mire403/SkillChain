from __future__ import annotations

import json
from typing import Any, Dict, List

from skillchain.adapters.base import ChatMessage, LLMAdapter
from skillchain.types import SkillCall

from .base import RouterContext, SkillRouter


class LLMSemanticRouter(SkillRouter):
    """
    Semantic router that uses an LLM to choose a skill.

    This is selection-only. The returned SkillCall is an explicit decision artifact.
    """

    def __init__(self, *, llm: LLMAdapter, model: str) -> None:
        self._llm = llm
        self._model = model

    async def select_skill(
        self,
        *,
        user_intent: str,
        available_skills: List[Dict[str, str]],
        ctx: RouterContext,
    ) -> SkillCall:
        system = (
            "You are a skill router. Select exactly one skill.\n"
            "Return STRICT JSON with keys: skill_name, skill_input, rationale.\n"
            "Rules: choose only from available skills; do not execute tasks; no prose outside JSON."
        )
        skills_json = json.dumps(available_skills, ensure_ascii=False)
        user = (
            f"User intent:\n{user_intent}\n\n"
            f"Available skills (JSON):\n{skills_json}\n\n"
            f"Short-term context keys:\n{sorted(list(ctx.short_term_context.keys()))}\n"
        )
        if ctx.long_term_summary:
            user += f"\nLong-term memory summary:\n{ctx.long_term_summary}\n"

        resp = await self._llm.chat(
            model=self._model,
            messages=[ChatMessage(role="system", content=system), ChatMessage(role="user", content=user)],
            temperature=0.0,
            max_tokens=300,
        )
        data = json.loads(resp.content)
        return SkillCall(
            skill_name=str(data["skill_name"]),
            skill_input=dict(data.get("skill_input") or {}),
            rationale=data.get("rationale"),
        )

