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
    """OpenAI Responses API configuration.

    Note: reads OPENAI_API_KEY (matching Railway env var naming),
    not LLM_API_KEY which was never set.
    """

    api_key: str = os.environ.get("OPENAI_API_KEY", "")
    base_url: str = os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1")
    model: str = os.environ.get("LLM_MODEL", "gpt-5-nano")


@dataclass(frozen=True)
class XAIConfig:
    """xAI Responses API configuration.

    Model: grok-4-1-fast-reasoning (reasoning model, not non-reasoning).
    Endpoint: base_url + /responses (appended by client.py).
    """

    api_key: str = os.environ.get("XAI_API_KEY", "")
    base_url: str = os.environ.get("XAI_BASE_URL", "https://api.x.ai/v1")
    model: str = os.environ.get("XAI_MODEL", "grok-4-1-fast-reasoning")


@dataclass(frozen=True)
class ComputeSDKConfig:
    """ComputeSDK is managed by the TS proxy layer (Node SDK).
    Python backend calls TS proxy for sandbox operations."""

    proxy_url: str = os.environ.get("COMPUTE_SDK_PROXY_URL", "http://localhost:8080")
    enabled: bool = os.environ.get("COMPUTE_SDK_ENABLED", "true").lower() != "false"


@dataclass(frozen=True)
class GPUHarvestConfig:
    """GPU-accelerated coordizer harvest via ComputeSDK/Railway.

    When enabled, the harvest pipeline runs on GPU instances through
    the ComputeSDK Railway provider for full probability distribution
    capture from Transformers/vLLM backends.

    v6.0 §19: CoordizerV2 three-phase scoring (256→2K→10K→32K)
    with four vocabulary tiers.
    """

    enabled: bool = os.environ.get("GPU_HARVEST_ENABLED", "false").lower() == "true"
    model_id: str = os.environ.get("GPU_HARVEST_MODEL", "meta-llama/Llama-3.2-3B")
    batch_size: int = int(os.environ.get("GPU_HARVEST_BATCH_SIZE", "32"))
    vocab_target: int = int(os.environ.get("GPU_HARVEST_VOCAB_TARGET", "32768"))
    artifact_dir: str = os.environ.get("GPU_HARVEST_ARTIFACT_DIR", "/data/resonance-bank")
    # Three-phase scoring thresholds (v6.0 §19.1)
    phase1_cutoff: int = int(os.environ.get("GPU_HARVEST_PHASE1_CUTOFF", "2000"))
    phase2_cutoff: int = int(os.environ.get("GPU_HARVEST_PHASE2_CUTOFF", "10000"))
    phase3_cutoff: int = int(os.environ.get("GPU_HARVEST_PHASE3_CUTOFF", "32768"))


@dataclass(frozen=True)
class ModalConfig:
    """Modal GPU integration — inference and coordizer harvesting.

    Inference:
        When inference_enabled=true and inference_url is set, the LLM
        client routes Ollama API calls to Modal's GPU-backed Ollama
        instance instead of Railway's CPU-only Ollama service.
        Fallback chain: Modal Ollama → Railway Ollama → xAI → OpenAI.

    Harvest:
        CoordizerV2 vocabulary fingerprinting via Modal GPU.
    """

    # --- Shared ---
    enabled: bool = os.environ.get("MODAL_ENABLED", "false").lower() == "true"
    token_id: str = os.environ.get("MODAL_TOKEN_ID", "")
    token_secret: str = os.environ.get("MODAL_TOKEN_SECRET", "")
    gpu_type: str = os.environ.get("MODAL_GPU_TYPE", "A10G")

    # --- Inference (Ollama on Modal GPU) ---
    inference_enabled: bool = os.environ.get("MODAL_INFERENCE_ENABLED", "false").lower() == "true"
    inference_url: str = os.environ.get("MODAL_INFERENCE_URL", "")
    inference_timeout_ms: int = int(os.environ.get("MODAL_INFERENCE_TIMEOUT_MS", "120000"))
    # Modal runs the base model (no custom Modelfile). The kernel
    # injects the system prompt per-request, so vex-brain overlay is
    # unnecessary. Defaults to base model name.
    inference_model: str = os.environ.get("MODAL_INFERENCE_MODEL", "lfm2.5-thinking:1.2b")

    # --- Harvest (CoordizerV2 fingerprinting) ---
    harvest_url: str = os.environ.get("MODAL_HARVEST_URL", "")


@dataclass(frozen=True)
class CoordizerV2Config:
    """CoordizerV2 integration configuration (v6.1F).

    CoordizerV2 uses harvest→compress→validate pipeline to build
    a Resonance Bank of geometric basin coordinates on Δ⁶³.

    When enabled, replaces the old BPE-style coordizer with
    LLM-harvested geometric structure.
    """

    # Feature flag: Enable CoordizerV2 integration
    enabled: bool = os.environ.get("COORDIZER_V2_ENABLED", "false").lower() == "true"

    # Path to saved Resonance Bank
    bank_path: str = os.environ.get("COORDIZER_V2_BANK_PATH", "./coordizer_data/bank")

    # Auto-load on startup (vs lazy load on first use)
    autoload: bool = os.environ.get("COORDIZER_V2_AUTOLOAD", "false").lower() == "true"

    # Regime modulation: scale temperature by regime weights
    regime_modulation: bool = os.environ.get("COORDIZER_V2_REGIME_MOD", "true").lower() != "false"

    # Navigation adaptation: adapt generation params by nav mode
    navigation_adaptation: bool = (
        os.environ.get("COORDIZER_V2_NAV_ADAPT", "true").lower() != "false"
    )

    # Tacking bias: tier selection based on tacking mode
    tacking_bias: bool = os.environ.get("COORDIZER_V2_TACKING_BIAS", "true").lower() != "false"

    # Domain bias strength for kernel specializations
    domain_bias_strength: float = float(os.environ.get("COORDIZER_V2_DOMAIN_BIAS", "0.3"))

    # Metrics integration: feed CoordizerV2 metrics to consciousness
    metrics_integration: bool = os.environ.get("COORDIZER_V2_METRICS", "true").lower() != "false"


@dataclass(frozen=True)
class PerplexityConfig:
    """Perplexity sonar-pro deep research integration.

    Used by kernel.tools.research for grounded, citation-backed
    information retrieval via the Perplexity chat completions API.
    """

    api_key: str = os.environ.get("PERPLEXITY_API_KEY", "")
    model: str = os.environ.get("PERPLEXITY_MODEL", "sonar-pro")
    base_url: str = os.environ.get("PERPLEXITY_BASE_URL", "https://api.perplexity.ai")
    timeout: int = int(os.environ.get("PERPLEXITY_TIMEOUT", "60"))


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
    consciousness_interval_ms: int = int(os.environ.get("CONSCIOUSNESS_INTERVAL_MS", "30000"))

    # LLM provider keys (kept for backward compat / direct access)
    anthropic_api_key: str = os.environ.get("ANTHROPIC_API_KEY", "")
    openai_api_key: str = os.environ.get("OPENAI_API_KEY", "")
    xai_api_key: str = os.environ.get("XAI_API_KEY", "")
    gemini_api_key: str = os.environ.get("GEMINI_API_KEY", "")
    groq_api_key: str = os.environ.get("GROQ_API_KEY", "")
    perplexity_api_key: str = os.environ.get("PERPLEXITY_API_KEY", "")
    hf_token: str = os.environ.get("HF_TOKEN", "")

    # Tool API keys
    tavily_api_key: str = os.environ.get("TAVILY_API_KEY", "")
    github_token: str = os.environ.get("GITHUB_TOKEN", "")
    github_username: str = os.environ.get("GITHUB_USERNAME", "")
    github_useremail: str = os.environ.get("GITHUB_USEREMAIL", "")

    # Auth
    kernel_api_key: str = os.environ.get("KERNEL_API_KEY", "")
    sync_secret: str = os.environ.get("SYNC_SECRET", "")

    # ComputeSDK / Railway
    computesdk_api_key: str = os.environ.get("COMPUTESDK_API_KEY", "")
    railway_api_key: str = os.environ.get("RAILWAY_API_KEY", "")
    railway_project_id: str = os.environ.get("RAILWAY_PROJECT_ID", "")
    railway_environment_id: str = os.environ.get("RAILWAY_ENVIRONMENT_ID", "")

    # Safety
    safety_mode: str = os.environ.get("SAFETY_MODE", "standard")

    # v6.1: Feature flags
    use_activation_sequence: bool = (
        os.environ.get("USE_ACTIVATION_SEQUENCE", "true").lower() != "false"
    )

    # Silent Observer — background Grok observation of conversations
    silent_observer_enabled: bool = (
        os.environ.get("SILENT_OBSERVER_ENABLED", "false").lower() == "true"
    )

    # Logging
    log_level: str = os.environ.get("LOG_LEVEL", "info")

    # Sub-configs
    ollama: OllamaConfig = field(default_factory=OllamaConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    xai: XAIConfig = field(default_factory=XAIConfig)
    compute_sdk: ComputeSDKConfig = field(default_factory=ComputeSDKConfig)
    gpu_harvest: GPUHarvestConfig = field(default_factory=GPUHarvestConfig)
    governor: GovernorConfig = field(default_factory=GovernorConfig)
    searxng: SearXNGConfig = field(default_factory=SearXNGConfig)
    modal: ModalConfig = field(default_factory=ModalConfig)
    perplexity: PerplexityConfig = field(default_factory=PerplexityConfig)
    coordizer_v2: CoordizerV2Config = field(default_factory=CoordizerV2Config)


settings = Settings()
