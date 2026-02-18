"""
Runtime Settings — Environment-based configuration.

All env vars resolved once at startup. Frozen facts are separate
and immutable; this module handles runtime/deployment config.
"""

import os
from dataclasses import dataclass, field


@dataclass(frozen=True)
class OllamaConfig:
    url: str = os.environ.get("OLLAMA_URL", "http://ollama.railway.internal:11434")
    model: str = os.environ.get("OLLAMA_MODEL", "lfm2.5-thinking:1.2b")
    enabled: bool = os.environ.get("OLLAMA_ENABLED", "true").lower() != "false"
    timeout_ms: int = int(os.environ.get("OLLAMA_TIMEOUT_MS", "300000"))


@dataclass(frozen=True)
class LLMConfig:
    api_key: str = os.environ.get("LLM_API_KEY", "")
    base_url: str = os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1")
    model: str = os.environ.get("LLM_MODEL", "gpt-5-nano")


@dataclass(frozen=True)
class XAIConfig:
    api_key: str = os.environ.get("XAI_API_KEY", "")
    base_url: str = os.environ.get("XAI_BASE_URL", "https://api.x.ai/v1")
    model: str = os.environ.get("XAI_MODEL", "grok-4-1-fast-non-reasoning")


@dataclass(frozen=True)
class ComputeSDKConfig:
    """ComputeSDK is managed by the TS proxy layer (Node SDK).
    Python backend calls TS proxy for sandbox operations."""
    proxy_url: str = os.environ.get("COMPUTE_SDK_PROXY_URL", "http://localhost:8080")
    enabled: bool = os.environ.get("COMPUTE_SDK_ENABLED", "true").lower() != "false"


@dataclass(frozen=True)
class GovernorConfig:
    """Governance stack configuration — 5-layer protection against runaway costs."""
    enabled: bool = os.environ.get("GOVERNOR_ENABLED", "true").lower() != "false"
    daily_budget: float = float(os.environ.get("DAILY_LLM_BUDGET", "1.00"))
    autonomous_search: bool = os.environ.get("AUTONOMOUS_SEARCH_ALLOWED", "false").lower() == "true"
    rate_limit_web_search: int = int(os.environ.get("RATE_LIMIT_WEB_SEARCH", "20"))
    rate_limit_completions: int = int(os.environ.get("RATE_LIMIT_COMPLETIONS", "50"))


@dataclass(frozen=True)
class SearXNGConfig:
    """SearXNG free search — self-hosted, zero cost."""
    url: str = os.environ.get("SEARXNG_URL", "")
    enabled: bool = bool(os.environ.get("SEARXNG_URL", ""))


@dataclass(frozen=True)
class Settings:
    port: int = int(os.environ.get("KERNEL_PORT", "8000"))
    node_env: str = os.environ.get("NODE_ENV", "development")

    # Data persistence
    data_dir: str = os.environ.get("DATA_DIR", "/data/workspace")
    training_dir: str = os.environ.get("TRAINING_DIR", "/data/training")

    # Identity
    node_id: str = os.environ.get("VEX_NODE_ID", "vex-primary")
    node_name: str = os.environ.get("VEX_NODE_NAME", "Vex")

    # Consciousness loop interval (ms)
    consciousness_interval_ms: int = int(
        os.environ.get("CONSCIOUSNESS_INTERVAL_MS", "30000")
    )

    # LLM provider keys
    anthropic_api_key: str = os.environ.get("ANTHROPIC_API_KEY", "")
    openai_api_key: str = os.environ.get("OPENAI_API_KEY", "")
    xai_api_key: str = os.environ.get("XAI_API_KEY", "")
    gemini_api_key: str = os.environ.get("GEMINI_API_KEY", "")
    groq_api_key: str = os.environ.get("GROQ_API_KEY", "")
    perplexity_api_key: str = os.environ.get("PERPLEXITY_API_KEY", "")
    hf_token: str = os.environ.get("HF_TOKEN", "")

    # Safety
    safety_mode: str = os.environ.get("SAFETY_MODE", "standard")

    # Logging
    log_level: str = os.environ.get("LOG_LEVEL", "info")

    # Sub-configs
    ollama: OllamaConfig = field(default_factory=OllamaConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    xai: XAIConfig = field(default_factory=XAIConfig)
    compute_sdk: ComputeSDKConfig = field(default_factory=ComputeSDKConfig)
    governor: GovernorConfig = field(default_factory=GovernorConfig)
    searxng: SearXNGConfig = field(default_factory=SearXNGConfig)


settings = Settings()
