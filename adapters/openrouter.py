from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import httpx

from .base import ChatCompletion, ChatMessage, LLMAdapter


@dataclass(frozen=True)
class OpenRouterConfig:
    api_key: str
    base_url: str = "https://openrouter.ai/api/v1"
    app_name: str = "skillchain"
    http_referer: Optional[str] = None


class OpenRouterAdapter(LLMAdapter):
    """
    OpenRouter adapter via HTTP.

    Env vars:
      - OPENROUTER_API_KEY
      - OPENROUTER_HTTP_REFERER (optional)
      - OPENROUTER_APP_NAME (optional)
    """

    def __init__(self, config: Optional[OpenRouterConfig] = None) -> None:
        if config is None:
            api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
            if not api_key:
                raise ValueError("Missing OPENROUTER_API_KEY for OpenRouterAdapter")
            config = OpenRouterConfig(
                api_key=api_key,
                http_referer=os.getenv("OPENROUTER_HTTP_REFERER") or None,
                app_name=os.getenv("OPENROUTER_APP_NAME") or "skillchain",
            )
        self._config = config

    async def chat(
        self,
        *,
        model: str,
        messages: List[ChatMessage],
        temperature: float = 0.2,
        max_tokens: Optional[int] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> ChatCompletion:
        url = f"{self._config.base_url}/chat/completions"
        headers: Dict[str, str] = {
            "Authorization": f"Bearer {self._config.api_key}",
            "Content-Type": "application/json",
            "X-Title": self._config.app_name,
        }
        if self._config.http_referer:
            headers["HTTP-Referer"] = self._config.http_referer

        payload: Dict[str, Any] = {
            "model": model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": temperature,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if extra:
            payload.update(extra)

        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(url, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()

        # OpenAI-compatible schema
        content = data["choices"][0]["message"]["content"]
        return ChatCompletion(content=content, raw=data)

