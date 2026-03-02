"""
Modal GPU Function — CoordizerV2 Harvest Endpoint

This file is deployed to Modal separately from the Railway app.
It provides a GPU-backed HTTP endpoint that:

1. Loads an LLM (cached on Modal Volume)
2. Runs forward passes on provided text corpus
3. Returns FULL probability distributions per token position
4. Computes per-token Fréchet means (sqrt-space averaging)

Deploy:
    modal deploy modal/vex_coordizer_harvest.py

    After deploy, Modal prints the endpoint URL. Update Railway env:
        MODAL_HARVEST_URL=https://garyocean428--vex-coordizer-harvest-<hash>.modal.run

Endpoint:
    POST / — protected by X-Api-Key header (KERNEL_API_KEY).
    Accepts JSON body matching HarvestRequest schema.

Cost estimate:
    A10G: ~$0.000306/sec → ~$0.16 per 10K samples
    T4:   ~$0.000164/sec → ~$0.09 per 10K samples

CRITICAL: Returns full V-dimensional distributions, NOT top-k.
The tail of the distribution carries geometric information that
top-k approximations destroy.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import modal

if TYPE_CHECKING:
    from starlette.requests import Request

# --- Configuration --------------------------------------------------------
# HARVEST_MODEL_ID: HuggingFace model to load for probability-distribution
# extraction.  Default is GLM-4.7-Flash (4.7B params, fits A10G, 65K vocab).
# GLM-4.7-Flash ensures token-ID alignment with the Modal inference model
# so resonance bank fingerprints use the same vocabulary.  Fallback option:
# "LiquidAI/LFM2.5-1.2B-Thinking" (smaller, faster, but different tokenizer).
# See kernel/config/settings.py GPUHarvestConfig for the Railway-side
# mirror of this setting.
HARVEST_MODEL_ID = os.environ.get("HARVEST_MODEL_ID", "zai-org/GLM-4.7-Flash")
KERNEL_API_KEY = os.environ.get("KERNEL_API_KEY", "")

app = modal.App("vex-coordizer-harvest")

# Persistent volume for model weights (cached across cold starts)
model_volume = modal.Volume.from_name("vex-models", create_if_missing=True)

# Image with ML dependencies
ml_image = modal.Image.debian_slim(python_version="3.14").pip_install(
    "torch>=2.1",
    "transformers>=4.40",
    "accelerate",
    "numpy>=1.26",
    "pydantic>=2.0",
    "fastapi[standard]",
)


@app.cls(
    gpu="A10G",
    image=ml_image,
    timeout=600,
    scaledown_window=300,
    volumes={"/models": model_volume},
    secrets=[modal.Secret.from_name("model")],
)
class CoordizerHarvester:
    """GPU-backed harvester for CoordizerV2.

    Supports dynamic model selection per request with in-memory caching.
    Default model is loaded on container start for fast cold starts.
    Additional models can be loaded on-demand if specified in requests.
    """

    @modal.enter()
    def load_model(self):
        """Load default model on container start (runs once per container)."""
        if not KERNEL_API_KEY:
            print(
                "WARNING: KERNEL_API_KEY not set — harvest endpoint is "
                "unauthenticated. Set it in the 'model' Modal secret."
            )

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        # Initialize model cache: {model_id: (tokenizer, model, vocab_size)}
        self._model_cache: dict[str, tuple] = {}

        # Load default model
        default_model_id = HARVEST_MODEL_ID
        cache_dir = "/models/hub"

        print(f"Loading default model: {default_model_id}")
        tokenizer = AutoTokenizer.from_pretrained(default_model_id, cache_dir=cache_dir)
        model = AutoModelForCausalLM.from_pretrained(
            default_model_id,
            cache_dir=cache_dir,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
        model.eval()
        vocab_size = tokenizer.vocab_size

        # Store in cache
        self._model_cache[default_model_id] = (tokenizer, model, vocab_size)

        # Set as current active model
        self.tokenizer = tokenizer
        self.model = model
        self.vocab_size = vocab_size
        self.current_model_id = default_model_id

        print(f"Default model loaded. Vocab size: {vocab_size}")

        # Commit volume so weights persist across cold starts
        model_volume.commit()

    def _load_model(self, model_id: str) -> tuple:
        """Load a model on-demand and cache it.

        Returns: (tokenizer, model, vocab_size)
        """
        if model_id in self._model_cache:
            print(f"Using cached model: {model_id}")
            return self._model_cache[model_id]

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        cache_dir = "/models/hub"
        print(f"Loading new model: {model_id}")

        tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            cache_dir=cache_dir,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
        model.eval()
        vocab_size = tokenizer.vocab_size

        # Cache for future requests
        self._model_cache[model_id] = (tokenizer, model, vocab_size)

        print(f"Model {model_id} loaded. Vocab size: {vocab_size}")
        model_volume.commit()

        return tokenizer, model, vocab_size

    @modal.fastapi_endpoint(method="GET")
    async def health(self):
        """Health check — returns model metadata once loaded."""
        return {
            "status": "ok",
            "current_model_id": getattr(self, "current_model_id", None),
            "vocab_size": getattr(self, "vocab_size", None),
            "cached_models": list(getattr(self, "_model_cache", {}).keys()),
        }

    @modal.fastapi_endpoint(method="POST")
    async def harvest(self, request: Request):
        """GPU harvest endpoint (X-Api-Key protected).

        Auth:
            Requires X-Api-Key header matching KERNEL_API_KEY env var.
            If KERNEL_API_KEY is not set, auth is skipped (dev mode).

        Request JSON body:
            {
                "texts": [str, ...],
                "model_id": str (optional, uses default if not specified),
                "batch_size": int (default 32),
                "max_length": int (default 512),
                "min_contexts": int (default 5)
            }

        Response:
            {
                "success": true,
                "model_id": "zai-org/GLM-4.7-Flash",
                "vocab_size": 65536,
                "total_tokens_processed": 12345,
                "tokens": {
                    "42": {
                        "string": "hello",
                        "fingerprint": [0.001, 0.002, ...],
                        "context_count": 15
                    },
                    ...
                },
                "elapsed_seconds": 45.2
            }
        """
        import time

        import numpy as np
        import torch
        from starlette.responses import JSONResponse

        # Auth: validate X-Api-Key if KERNEL_API_KEY is configured
        if KERNEL_API_KEY:
            provided_key = request.headers.get("x-api-key", "")
            if provided_key != KERNEL_API_KEY:
                reason = "missing" if not provided_key else "invalid"
                print(f"Auth failure ({reason} api key) from {request.client.host if request.client else 'unknown'}")
                return JSONResponse(
                    status_code=403,
                    content={"error": "invalid or missing api key"},
                )

        start = time.time()

        body = await request.json()
        texts = body.get("texts", [])
        requested_model = body.get("model_id", None)
        batch_size = body.get("batch_size", 32)
        max_length = body.get("max_length", 512)
        min_contexts = body.get("min_contexts", 5)

        if not texts:
            return {"success": False, "error": "No texts provided"}

        # Select model: request model_id → current model → default
        target_model_id = requested_model or self.current_model_id

        # Load model if not already active
        if target_model_id != self.current_model_id:
            try:
                tokenizer, model, vocab_size = self._load_model(target_model_id)
                # Switch active model
                self.tokenizer = tokenizer
                self.model = model
                self.vocab_size = vocab_size
                self.current_model_id = target_model_id
            except Exception as e:
                return {
                    "success": False,
                    "error": f"Failed to load model {target_model_id}: {e}",
                }

        EPS = 1e-12

        # Accumulators: sqrt-space averaging for Fréchet mean
        dist_sums_sqrt: dict[int, np.ndarray] = {}
        dist_counts: dict[int, int] = {}
        total_tokens = 0

        for batch_start in range(0, len(texts), batch_size):
            batch = texts[batch_start : batch_start + batch_size]

            for text in batch:
                try:
                    input_ids = self.tokenizer.encode(
                        text,
                        return_tensors="pt",
                        truncation=True,
                        max_length=max_length,
                    ).to(self.model.device)

                    with torch.no_grad():
                        outputs = self.model(input_ids)
                        logits = outputs.logits[0]

                    # FULL softmax distribution — not top-k
                    probs = torch.softmax(logits, dim=-1).cpu().numpy()
                    ids = input_ids[0].cpu().numpy()

                    for pos in range(len(ids) - 1):
                        token_id = int(ids[pos])
                        output_dist = probs[pos].astype(np.float64)

                        # Ensure strictly positive for Fisher-Rao
                        output_dist = np.maximum(output_dist, EPS)
                        output_dist = output_dist / output_dist.sum()

                        sqrt_dist = np.sqrt(output_dist)

                        if token_id not in dist_sums_sqrt:
                            dist_sums_sqrt[token_id] = sqrt_dist.copy()
                            dist_counts[token_id] = 1
                        else:
                            dist_sums_sqrt[token_id] += sqrt_dist
                            dist_counts[token_id] += 1

                        total_tokens += 1

                except Exception as e:
                    print(f"Error processing text: {e}")
                    continue

        # Compute Fréchet means and build response
        tokens_response = {}
        for token_id, sqrt_sum in dist_sums_sqrt.items():
            count = dist_counts[token_id]
            if count < min_contexts:
                continue

            # Fréchet mean via sqrt-space averaging
            mean_sqrt = sqrt_sum / count
            mean_dist = mean_sqrt * mean_sqrt
            mean_dist = mean_dist / mean_dist.sum()

            try:
                token_str = self.tokenizer.decode([token_id])
            except Exception:
                token_str = f"<token_{token_id}>"

            tokens_response[str(token_id)] = {
                "string": token_str,
                "fingerprint": mean_dist.tolist(),
                "context_count": count,
            }

        elapsed = time.time() - start

        return {
            "success": True,
            "model_id": self.current_model_id,
            "vocab_size": self.vocab_size,
            "total_tokens_processed": total_tokens,
            "tokens": tokens_response,
            "elapsed_seconds": round(elapsed, 2),
        }


@app.function(
    image=ml_image,
    volumes={"/models": model_volume},
    timeout=1200,
)
def download_model(model_id: str = HARVEST_MODEL_ID):
    """One-time model download to Modal Volume.

    Run: modal run modal/vex_coordizer_harvest.py::download_model
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cache_dir = "/models/hub"
    print(f"Downloading {model_id} to {cache_dir}...")

    AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)
    AutoModelForCausalLM.from_pretrained(model_id, cache_dir=cache_dir)

    model_volume.commit()
    print(f"Done. Model cached at {cache_dir}")
