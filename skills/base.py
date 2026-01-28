from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class SkillContext:
    """
    Execution context passed into a skill.

    Note: skills must not manage memory; this is read-only context from the agent runtime.
    """

    request_id: str
    short_term_context: Dict[str, Any]


class Skill(ABC):
    """
    Capability primitive.

    Rules:
      - Single responsibility
      - Explicit input/output
      - No routing
      - No memory management
      - No calling other skills directly
    """

    name: str
    description: str

    def __init__(self, *, name: Optional[str] = None, description: Optional[str] = None) -> None:
        if name is not None:
            self.name = name
        if description is not None:
            self.description = description

    @abstractmethod
    def run(self, *, skill_input: Dict[str, Any], ctx: SkillContext) -> Dict[str, Any]:
        raise NotImplementedError

