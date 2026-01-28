from __future__ import annotations

from typing import Any, Dict

from .base import Skill, SkillContext


class EchoSkill(Skill):
    """
    Reference skill: returns the provided input and a small ctx snapshot.

    Useful for testing the runtime pipeline without embedding agent behavior in prompts.
    """

    name = "echo"
    description = "Echoes input payload for debugging and pipeline verification."

    def run(self, *, skill_input: Dict[str, Any], ctx: SkillContext) -> Dict[str, Any]:
        return {
            "echo": skill_input,
            "ctx": {
                "request_id": ctx.request_id,
                "short_term_context_keys": sorted(list(ctx.short_term_context.keys())),
            },
        }

