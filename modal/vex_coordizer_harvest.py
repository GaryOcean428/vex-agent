"""
Modal GPU Function — CoordizerV2 Harvest + PGA Compress

Runs Qwen3.5-4B (default) or larger Qwen models (set via HARVEST_MODEL_ID
env var, e.g. Qwen/Qwen3.5-35B-A3B) in NF4 on A100 GPU.
Computes full V-dimensional probability distributions AND runs PGA
compress on-GPU, returning only 64D basin coords + 32D lens coords.

This eliminates the V8 string limit issue — no 248K-float fingerprints
are ever serialized to JSON. All heavy compute stays on GPU.

Deploy:
    modal deploy modal/vex_coordizer_harvest.py

Single ASGI app (1 endpoint slot) serving all routes:
    GET  /health   — Health check.
    POST /harvest  — Raw fingerprints (small requests only, for debugging).
    POST /coordize — Full pipeline: text -> fingerprints -> PGA -> 64D basin coords.

Lens dimension: 32 (from eigenvalue analysis: cumulative variance 0.7661 at dim 32).
Basin dimension: 64 (frozen). First 32 dims from PGA, rest zero-padded, unit normalized.

Model persistence:
    Weights cached on Modal Volume "vex-models" — persists across deploys.
    QLoRA adapter loaded from /models/adapters/harvest-qlora if available.
    This model is the harvest substrate — it evolves with every training run.

Note: HARVEST_MODEL_ID defaults to "Qwen/Qwen3.5-4B" in code as a safe
fallback. For production harvesting on 35B-A3B, set the Modal secret
`model` key HARVEST_MODEL_ID=Qwen/Qwen3.5-35B-A3B.
"""
