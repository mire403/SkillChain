from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class ChatMessage:
    role: str  # "system" | "user" | "assistant" | "tool"
    content: str


@dataclass(frozen=True)
class ChatCompletion:
    content: str
    raw: Optional[Dict[str, Any]] = None


class LLMAdapter(ABC):
    """
    Unified LLM interface.

    The rest of the codebase must depend only on this abstraction (not vendor SDKs).
    """

    @abstractmethod
    async def chat(
        self,
        *,
        model: str,
        messages: List[ChatMessage],
        temperature: float = 0.2,
        max_tokens: Optional[int] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> ChatCompletion:
        raise NotImplementedError

