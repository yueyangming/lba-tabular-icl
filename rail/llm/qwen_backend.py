from __future__ import annotations

import os
import time
from pathlib import Path

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from rail.config import OpenAICompatibleConfig
from rail.evaluation.io import append_jsonl
from rail.llm.cache import ResponseCache


class QwenBackend:
    def __init__(
        self,
        config: OpenAICompatibleConfig | None = None,
        *,
        cache: ResponseCache | None = None,
        timeout_seconds: float = 60.0,
        request_log_path: str | Path | None = None,
        use_stream: bool = True,
    ) -> None:
        self.config = config or OpenAICompatibleConfig.from_env()
        self.timeout_seconds = timeout_seconds
        self.request_log_path = request_log_path
        self.use_stream = use_stream
        default_headers: dict[str, str] | None = None
        if self.config.provider == "openrouter":
            default_headers = {}
            http_referer = os.getenv("OPENROUTER_HTTP_REFERER", "").strip()
            x_title = os.getenv("OPENROUTER_X_TITLE", "").strip()
            if http_referer:
                default_headers["HTTP-Referer"] = http_referer
            if x_title:
                default_headers["X-Title"] = x_title
        self.client = OpenAI(
            api_key=self.config.api_key,
            base_url=self.config.base_url,
            timeout=timeout_seconds,
            default_headers=default_headers,
        )
        # Backend cache is the lowest-level response cache. It deduplicates
        # provider/model/prompt-complete calls regardless of which runner
        # generated them. Runner-level caches above this can still memoize
        # method-specific payloads such as selected indices or strategy names.
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

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        start = time.perf_counter()
        model_name = model or self.config.default_model
        extra_body: dict[str, object] = {}
        if self.config.disable_thinking:
            extra_body["enable_thinking"] = False
        if self.config.provider == "minimax":
            extra_body["reasoning_split"] = True
        if thinking_budget is not None:
            extra_body["thinking_budget"] = thinking_budget
        try:
            response = self.client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=effective_temperature,
                max_tokens=max_tokens,
                timeout=timeout_seconds,
                stream=self.use_stream,
                extra_body=extra_body or None,
            )
            if self.use_stream:
                content_parts: list[str] = []
                reasoning_parts: list[str] = []
                for chunk in response:
                    if not chunk.choices:
                        continue
                    delta = chunk.choices[0].delta
                    content = getattr(delta, "content", None)
                    reasoning = getattr(delta, "reasoning_content", None)
                    reasoning_details = getattr(delta, "reasoning_details", None)
                    if content:
                        content_parts.append(content)
                    if reasoning:
                        reasoning_parts.append(reasoning)
                    if reasoning_details:
                        for detail in reasoning_details:
                            if isinstance(detail, dict) and detail.get("text"):
                                reasoning_parts.append(str(detail["text"]))
                text = "".join(content_parts).strip()
                reasoning_text = "".join(reasoning_parts).strip()
            else:
                content = response.choices[0].message.content
                text = content.strip() if content else ""
                reasoning_text = None
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
                reasoning_chars=(len(reasoning_text) if reasoning_text is not None else None),
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
