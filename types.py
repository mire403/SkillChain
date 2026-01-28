from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


JSONDict = Dict[str, Any]


@dataclass(frozen=True)
class SkillCall:
    """A decision artifact: which skill to run and with what input payload."""

    skill_name: str
    skill_input: JSONDict
    rationale: Optional[str] = None


@dataclass(frozen=True)
class SkillResult:
    """An execution artifact: structured output produced by a skill."""

    skill_name: str
    output: JSONDict
    success: bool = True
    error: Optional[str] = None

