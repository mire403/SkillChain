from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from skillchain.types import SkillCall, SkillResult


@dataclass
class ShortTermMemory:
    """
    Execution-context memory for a single run/session.

    Explicit structure (not prompt concatenation).
    """

    context: Dict[str, Any] = field(default_factory=dict)
    decisions: List[SkillCall] = field(default_factory=list)
    results: List[SkillResult] = field(default_factory=list)

    def record_decision(self, call: SkillCall) -> None:
        self.decisions.append(call)

    def record_result(self, result: SkillResult) -> None:
        self.results.append(result)

