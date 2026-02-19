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

Endpoint:
    POST /harvest (requires Modal Proxy Auth)

Cost estimate:
    A10G: ~$0.000306/sec → ~$0.16 per 10K samples
    T4:   ~$0.000164/sec → ~$0.09 per 10K samples

CRITICAL: Returns full V-dimensional distributions, NOT top-k.
The tail of the distribution carries geometric information that
top-k approximations destroy.
"""

from __future__ import annotations

import modal

app = modal.App("vex-coordizer-harvest")

# Persistent volume for model weights (cached across cold starts)
model_volume = modal.Volume.from_name("vex-models", create_if_missing=True)

# Image with ML dependencies
ml_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.1",
        "transformers>=4.40",
        "accelerate",
        "numpy>=1.26",
        "pydantic>=2.0",
    )
)


@app.cls(
    gpu="A10G",
    image=ml_image,
    timeout=600,
    container_idle_timeout=300,
    volumes={"/models": model_volume},
)
class CoordizerHarvester:
    """GPU-backed harvester for CoordizerV2.

    Loads model on container start, then handles harvest requests
    via FastAPI endpoint. Model weights are cached on Modal Volume
    to avoid re-downloading on cold starts.
    """

    @modal.enter()
    def load_model(self):
        """Load model on container start (runs once per container)."""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_id = "LiquidAI/LFM2.5-1.2B-Thinking"
        cache_dir = "/models/hub"

        print(f"Loading model: {model_id}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id, cache_dir=cache_dir
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            cache_dir=cache_dir,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
        self.model.eval()
        self.vocab_size = self.tokenizer.vocab_size
        print(f"Model loaded. Vocab size: {self.vocab_size}")

        # Commit volume so weights persist across cold starts
        model_volume.commit()

    @modal.fastapi_endpoint(requires_proxy_auth=True, method="POST")
    async def harvest(self, request: dict):
        """GPU harvest endpoint.

        Request:
            {
                "model_id": str (unused — model loaded at container start),
                "texts": [str, ...],
                "batch_size": int (default 32),
                "max_length": int (default 512),
                "return_full_distribution": bool (must be true)
            }

        Response:
            {
                "success": true,
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

        start = time.time()

        texts = request.get("texts", [])
        batch_size = request.get("batch_size", 32)
        max_length = request.get("max_length", 512)
        min_contexts = request.get("min_contexts", 5)

        if not texts:
            return {"success": False, "error": "No texts provided"}

        EPS = 1e-12

        # Accumulators: sqrt-space averaging for Fréchet mean
        dist_sums_sqrt: dict[int, np.ndarray] = {}
        dist_counts: dict[int, int] = {}
        total_tokens = 0

        for batch_start in range(0, len(texts), batch_size):
            batch = texts[batch_start:batch_start + batch_size]

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
def download_model(model_id: str = "LiquidAI/LFM2.5-1.2B-Thinking"):
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
