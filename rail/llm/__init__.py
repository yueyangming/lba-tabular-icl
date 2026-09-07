"""LLM backends for RAIL."""

from rail.config import OpenAICompatibleConfig
from rail.llm.anthropic_backend import AnthropicCompatibleBackend
from rail.llm.qwen_backend import QwenBackend


def build_backend(
    config: OpenAICompatibleConfig,
    *,
    timeout_seconds: float = 60.0,
    request_log_path=None,
):
    if config.provider == "minimax_anthropic":
        return AnthropicCompatibleBackend(
            config,
            timeout_seconds=timeout_seconds,
            request_log_path=request_log_path,
        )
    return QwenBackend(
        config,
        timeout_seconds=timeout_seconds,
        request_log_path=request_log_path,
    )
