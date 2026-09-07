from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from diskcache import Cache


@dataclass
class ResponseCache:
    path: str = "cache/llm"

    def __post_init__(self) -> None:
        self._cache = Cache(self.path)

    @staticmethod
    def _make_key(payload: dict[str, Any]) -> str:
        serialized = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def get(self, payload: dict[str, Any]) -> str | None:
        return self._cache.get(self._make_key(payload))

    def set(self, payload: dict[str, Any], value: str) -> None:
        self._cache.set(self._make_key(payload), value)


LLMCache = ResponseCache
