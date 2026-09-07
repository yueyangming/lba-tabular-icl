from __future__ import annotations

import time
from pathlib import Path

from anthropic import Anthropic
from tenacity import retry, stop_after_attempt, wait_exponential

from rail.config import OpenAICompatibleConfig
from rail.evaluation.io import append_jsonl
from rail.llm.cache import ResponseCache


class AnthropicCompatibleBackend:
    def __init__(
        self,
        config: OpenAICompatibleConfig | None = None,
        *,
        cache: ResponseCache | None = None,
        timeout_seconds: float = 60.0,
        request_log_path: str | Path | None = None,
    ) -> None:
        self.config = config or OpenAICompatibleConfig.from_env()
        self.timeout_seconds = timeout_seconds
        self.request_log_path = request_log_path
        self.client = Anthropic(
            api_key=self.config.api_key,
            base_url=self.config.base_url,
            timeout=timeout_seconds,
        )
        self.cache = cache or ResponseCache()

    @retry(wait=wait_exponential(min=1, max=8), stop=stop_after_attempt(3), reraise=True)
    def complete(
        self,
        prompt: str,
        *,
        model: str | None = None,
        system_prompt: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 256,
        timeout_seconds: float | None = None,
        thinking_budget: int | None = None,
    ) -> str:
        timeout_seconds = timeout_seconds or self.timeout_seconds
        effective_temperature = max(float(temperature), float(self.config.min_temperature))
        payload = {
            "provider": self.config.provider,
            "model": model or self.config.default_model,
            "system_prompt": system_prompt,
            "prompt": prompt,
            "temperature": effective_temperature,
            "max_tokens": max_tokens,
            "thinking_budget": thinking_budget,
            "disable_thinking": self.config.disable_thinking,
        }
        cached = self.cache.get(payload)
        if cached is not None:
            self._log_request(
                provider=self.config.provider,
                model=model or self.config.default_model,
                temperature=effective_temperature,
                max_tokens=max_tokens,
                prompt=prompt,
                status="cache_hit",
                elapsed_seconds=0.0,
                timeout_seconds=timeout_seconds,
                thinking_budget=thinking_budget,
            )
            return cached

        start = time.perf_counter()
        model_name = model or self.config.default_model
        request_kwargs: dict[str, object] = {
            "model": model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": effective_temperature,
            "max_tokens": max_tokens,
        }
        if system_prompt:
            request_kwargs["system"] = system_prompt
        if not self.config.disable_thinking and thinking_budget is not None and thinking_budget > 0:
            request_kwargs["thinking"] = {
                "type": "enabled",
                "budget_tokens": max(1024, int(thinking_budget)),
            }
        try:
            response = self.client.messages.create(**request_kwargs)
            content_parts: list[str] = []
            reasoning_parts: list[str] = []
            for block in response.content:
                block_type = getattr(block, "type", None)
                if block_type == "text":
                    text_value = getattr(block, "text", None)
                    if text_value:
                        content_parts.append(text_value)
                elif block_type == "thinking":
                    thinking_value = getattr(block, "thinking", None)
                    if thinking_value:
                        reasoning_parts.append(thinking_value)
            text = "".join(content_parts).strip()
            reasoning_text = "".join(reasoning_parts).strip()
            self.cache.set(payload, text)
            self._log_request(
                provider=self.config.provider,
                model=model_name,
                temperature=effective_temperature,
                max_tokens=max_tokens,
                prompt=prompt,
                status="ok" if text else "empty_content",
                elapsed_seconds=time.perf_counter() - start,
                timeout_seconds=timeout_seconds,
                response_chars=len(text),
                reasoning_chars=(len(reasoning_text) if reasoning_text else None),
                thinking_budget=thinking_budget,
            )
            return text
        except Exception as exc:
            self._log_request(
                provider=self.config.provider,
                model=model_name,
                temperature=effective_temperature,
                max_tokens=max_tokens,
                prompt=prompt,
                status="error",
                elapsed_seconds=time.perf_counter() - start,
                timeout_seconds=timeout_seconds,
                thinking_budget=thinking_budget,
                error_type=type(exc).__name__,
                error_message=str(exc),
            )
            raise

    def _log_request(
        self,
        *,
        provider: str,
        model: str,
        temperature: float,
        max_tokens: int,
        prompt: str,
        status: str,
        elapsed_seconds: float,
        timeout_seconds: float,
        thinking_budget: int | None,
        response_chars: int | None = None,
        reasoning_chars: int | None = None,
        error_type: str | None = None,
        error_message: str | None = None,
    ) -> None:
        if self.request_log_path is None:
            return
        append_jsonl(
            self.request_log_path,
            {
                "provider": provider,
                "model": model,
                "status": status,
                "elapsed_seconds": round(elapsed_seconds, 4),
                "timeout_seconds": timeout_seconds,
                "thinking_budget": thinking_budget,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "prompt_chars": len(prompt),
                "response_chars": response_chars,
                "reasoning_chars": reasoning_chars,
                "error_type": error_type,
                "error_message": error_message,
            },
        )
