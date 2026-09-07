from __future__ import annotations

from typing import Protocol


class ChatBackend(Protocol):
    def complete(self, prompt: str, *, model: str | None = None, system_prompt: str | None = None) -> str:
        ...
