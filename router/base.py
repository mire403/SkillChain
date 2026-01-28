from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from skillchain.types import SkillCall


@dataclass(frozen=True)
class RouterContext:
    """
    Context passed into a router to make a selection.

    Routers may use LLMs/prompts, but they must not execute tasks (no side effects beyond selection).
    """

    short_term_context: Dict[str, Any]
    long_term_summary: Optional[str] = None


class SkillRouter(ABC):
    """Selects which skill should run next, given a user goal and available skills."""

    @abstractmethod
    async def select_skill(
        self,
        *,
        user_intent: str,
        available_skills: List[Dict[str, str]],
        ctx: RouterContext,
    ) -> SkillCall:
        raise NotImplementedError

