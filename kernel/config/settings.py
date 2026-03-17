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
    model: str = os.environ.get("OLLAMA_MODEL", "vex-brain")
    enabled: bool = os.environ.get("OLLAMA_ENABLED", "true").lower() != "false"
    timeout_ms: int = int(os.environ.get("OLLAMA_TIMEOUT_MS", "300000"))