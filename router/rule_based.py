from __future__ import annotations

from typing import Dict, List

from skillchain.types import SkillCall

from .base import RouterContext, SkillRouter


class KeywordRouter(SkillRouter):
    """
    Minimal local router (no LLM): picks a skill by keyword match.

    This is mainly for quick local runs/tests. Production routing should use semantic routing.
    """

    def __init__(self, *, default_skill: str) -> None:
        self._default = default_skill

    async def select_skill(
        self,
        *,
        user_intent: str,
        available_skills: List[Dict[str, str]],
        ctx: RouterContext,
    ) -> SkillCall:
        _ = ctx  # selection-only; keep signature consistent
        names = {s["name"] for s in available_skills}

        intent_lower = user_intent.lower()
        for s in available_skills:
            name = s["name"]
            if name.lower() in intent_lower:
                return SkillCall(skill_name=name, skill_input={"text": user_intent}, rationale="keyword")

        if self._default not in names:
            # fallback to first available skill
            chosen = available_skills[0]["name"] if available_skills else self._default
            return SkillCall(skill_name=chosen, skill_input={"text": user_intent}, rationale="fallback-first")

        return SkillCall(skill_name=self._default, skill_input={"text": user_intent}, rationale="default")

