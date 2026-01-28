from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class LongTermMemory:
    """
    Persistent knowledge store.

    Intentionally simple JSON file backend to keep the framework inspectable.
    """

    path: Path
    data: Dict[str, Any] = field(default_factory=dict)

    def load(self) -> None:
        if not self.path.exists():
            self.data = {}
            return
        self.data = json.loads(self.path.read_text(encoding="utf-8") or "{}")

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, indent=2, ensure_ascii=False), encoding="utf-8")

    def get_summary(self) -> Optional[str]:
        """
        Optional: small human/LLM-readable summary for routers.
        Keep it short and explicit.
        """

        if not self.data:
            return None
        keys = sorted(list(self.data.keys()))
        return f"LongTermMemory keys: {keys}"

