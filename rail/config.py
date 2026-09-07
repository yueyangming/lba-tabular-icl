from __future__ import annotations

import os
from pathlib import Path
from dataclasses import dataclass

from dotenv import load_dotenv


load_dotenv(Path(__file__).resolve().parents[1] / ".env")


def _parse_bool(value: str | None, *, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class OpenAICompatibleConfig:
    provider: str
    api_key: str
    base_url: str
    default_model: str
    latest_model: str
    debug_model: str
    disable_thinking: bool
    min_temperature: float

    @classmethod
    def from_env(cls) -> "OpenAICompatibleConfig":
        provider = cls._resolve_provider()
        defaults = cls._provider_defaults(provider)
        provider_api_key = os.getenv(defaults["api_key_env"], "")
        provider_base_url = os.getenv(defaults["base_url_env"], "")
        if not provider_api_key and defaults.get("api_key_env_fallback"):
            provider_api_key = os.getenv(str(defaults["api_key_env_fallback"]), "")
        if not provider_base_url and defaults.get("base_url_env_fallback"):
            provider_base_url = os.getenv(str(defaults["base_url_env_fallback"]), "")
        provider_default_model = os.getenv(defaults["default_model_env"], "")
        provider_latest_model = os.getenv(defaults["latest_model_env"], "")
        provider_debug_model = os.getenv(defaults["debug_model_env"], "")
        # If an explicit provider is selected, prefer that provider's own env
        # variables over the generic compatibility aliases. This lets one .env
        # file keep multiple provider credentials without routing requests to the
        # wrong base URL.
        api_key = provider_api_key or os.getenv("OPENAI_COMPAT_API_KEY") or os.getenv("QWEN_API_KEY")
        base_url = provider_base_url or os.getenv("OPENAI_COMPAT_BASE_URL") or os.getenv("QWEN_BASE_URL") or defaults["base_url"]
        default_model = provider_default_model or os.getenv("QWEN_DEFAULT_MODEL", defaults["default_model"])
        latest_model = provider_latest_model or os.getenv("QWEN_LATEST_MODEL", defaults["latest_model"])
        debug_model = provider_debug_model or os.getenv("QWEN_DEBUG_MODEL", defaults["debug_model"])
        disable_thinking = _parse_bool(
            os.getenv("QWEN_DISABLE_THINKING"),
            default=bool(defaults["disable_thinking"]),
        )
        min_temperature = float(os.getenv("QWEN_MIN_TEMPERATURE", defaults["min_temperature"]))
        if not api_key:
            api_env = defaults["api_key_env"]
            raise ValueError(
                f"Missing API key in environment. Set OPENAI_COMPAT_API_KEY, QWEN_API_KEY, or {api_env}."
            )
        return cls(
            provider=provider,
            api_key=api_key,
            base_url=base_url,
            default_model=default_model,
            latest_model=latest_model,
            debug_model=debug_model,
            disable_thinking=disable_thinking,
            min_temperature=min_temperature,
        )

    @staticmethod
    def _resolve_provider() -> str:
        raw_provider = os.getenv("QWEN_PROVIDER", "").strip().lower()
        if raw_provider in {"aliyun", "dashscope", "alibaba"}:
            return "aliyun"
        if raw_provider in {"openrouter", "open_router"}:
            return "openrouter"
        if raw_provider in {"minimax_anthropic", "minimax-anthropic"}:
            return "minimax_anthropic"
        if raw_provider in {"minimax", "mini_max"}:
            return "minimax"
        if raw_provider in {"poe"}:
            return "poe"
        if raw_provider in {"siliconflow", "silicon_flow"}:
            return "siliconflow"
        if os.getenv("QWEN_API_KEY"):
            return raw_provider or "siliconflow"
        if os.getenv("DASHSCOPE_API_KEY"):
            return "aliyun"
        if os.getenv("OPENROUTER_API_KEY"):
            return raw_provider or "openrouter"
        if os.getenv("MINIMAX_ANTHROPIC_API_KEY"):
            return raw_provider or "minimax_anthropic"
        if os.getenv("MINIMAX_API_KEY"):
            return raw_provider or "minimax"
        if os.getenv("POE_API_KEY"):
            return raw_provider or "poe"
        return "siliconflow"

    @staticmethod
    def _provider_defaults(provider: str) -> dict[str, str | bool]:
        if provider == "aliyun":
            return {
                "api_key_env": "DASHSCOPE_API_KEY",
                "base_url_env": "DASHSCOPE_BASE_URL",
                "default_model_env": "DASHSCOPE_DEFAULT_MODEL",
                "latest_model_env": "DASHSCOPE_LATEST_MODEL",
                "debug_model_env": "DASHSCOPE_DEBUG_MODEL",
                "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
                "default_model": "qwen3.5-35b-a3b",
                "latest_model": "qwen3.5-397b-a17b",
                "debug_model": "qwen3.5-27b",
                "disable_thinking": True,
                "min_temperature": 0.0,
            }
        if provider == "openrouter":
            return {
                "api_key_env": "OPENROUTER_API_KEY",
                "base_url_env": "OPENROUTER_BASE_URL",
                "default_model_env": "OPENROUTER_DEFAULT_MODEL",
                "latest_model_env": "OPENROUTER_LATEST_MODEL",
                "debug_model_env": "OPENROUTER_DEBUG_MODEL",
                "base_url": "https://openrouter.ai/api/v1",
                "default_model": "meta-llama/llama-3.3-70b-instruct:free",
                "latest_model": "meta-llama/llama-3.3-70b-instruct:free",
                "debug_model": "meta-llama/llama-3.3-70b-instruct:free",
                "disable_thinking": True,
                "min_temperature": 0.0,
            }
        if provider == "minimax":
            return {
                "api_key_env": "MINIMAX_API_KEY",
                "base_url_env": "MINIMAX_BASE_URL",
                "default_model_env": "MINIMAX_DEFAULT_MODEL",
                "latest_model_env": "MINIMAX_LATEST_MODEL",
                "debug_model_env": "MINIMAX_DEBUG_MODEL",
                "base_url": "https://api.minimaxi.com/v1",
                "default_model": "MiniMax-M2.5",
                "latest_model": "MiniMax-M2.5",
                "debug_model": "MiniMax-M2.5",
                "disable_thinking": False,
                "min_temperature": 0.01,
            }
        if provider == "minimax_anthropic":
            return {
                "api_key_env": "MINIMAX_ANTHROPIC_API_KEY",
                "api_key_env_fallback": "MINIMAX_API_KEY",
                "base_url_env": "MINIMAX_ANTHROPIC_BASE_URL",
                "default_model_env": "MINIMAX_ANTHROPIC_DEFAULT_MODEL",
                "latest_model_env": "MINIMAX_ANTHROPIC_LATEST_MODEL",
                "debug_model_env": "MINIMAX_ANTHROPIC_DEBUG_MODEL",
                "base_url": "https://api.minimaxi.com/anthropic",
                "default_model": "MiniMax-M2.5",
                "latest_model": "MiniMax-M2.5",
                "debug_model": "MiniMax-M2.5",
                "disable_thinking": True,
                "min_temperature": 0.01,
            }
        if provider == "poe":
            return {
                "api_key_env": "POE_API_KEY",
                "base_url_env": "POE_BASE_URL",
                "default_model_env": "POE_DEFAULT_MODEL",
                "latest_model_env": "POE_LATEST_MODEL",
                "debug_model_env": "POE_DEBUG_MODEL",
                "base_url": "https://api.poe.com/v1",
                "default_model": "Claude-Sonnet-4.5",
                "latest_model": "Claude-Sonnet-4.5",
                "debug_model": "Claude-Sonnet-4.5",
                "disable_thinking": True,
                "min_temperature": 0.0,
            }
        return {
            "api_key_env": "SILICONFLOW_API_KEY",
            "base_url_env": "SILICONFLOW_BASE_URL",
            "default_model_env": "SILICONFLOW_DEFAULT_MODEL",
            "latest_model_env": "SILICONFLOW_LATEST_MODEL",
            "debug_model_env": "SILICONFLOW_DEBUG_MODEL",
            "base_url": "https://api.siliconflow.cn/v1",
            "default_model": "Qwen/Qwen3-32B",
            "latest_model": "Qwen/Qwen3.5-397B-A17B",
            "debug_model": "Qwen/Qwen3-8B",
            "disable_thinking": False,
            "min_temperature": 0.0,
        }


SiliconFlowConfig = OpenAICompatibleConfig
