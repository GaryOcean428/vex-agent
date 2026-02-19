"""
Modal GPU Harvest Function for CoordizerV2

Serverless A10G endpoint that runs LLM forward passes and returns
full softmax distributions for the coordizer harvesting pipeline.

Architecture (Option A — Modal as webhook/API):
    Railway (control plane) → HTTP POST → Modal (GPU worker) → logits → Railway

Deployment:
    modal deploy vex_coordizer_harvest_modal.py

Authentication:
    Modal Proxy Auth Tokens (wk-*/ws-* headers).
    Set MODAL_TOKEN_ID and MODAL_TOKEN_SECRET in Railway env.

Cost estimate:
    A10G: ~$0.000306/sec → ~$0.16 per 10K samples → <$5/month
"""

from __future__ import annotations

import time
from typing import Optional

import modal

# ─── Modal App ───────────────────────────────────────────────────────

app = modal.App("vex-coordizer-harvest")

# Persistent volume for cached model weights (survives cold starts)
model_volume = modal.Volume.from_name("vex-models", create_if_missing=True)

# Container image with dependencies
harvest_image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch>=2.1.0",
    "transformers>=4.36.0",
    "numpy>=1.24.0",
    "accelerate>=0.25.0",
)

MODEL_DIR = "/models"
DEFAULT_MODEL = "LiquidAI/LFM2.5-1.2B-Thinking"


# ─── One-Time Model Download ────────────────────────────────────────

@app.function(
    image=harvest_image,
    volumes={MODEL_DIR: model_volume},
    timeout=600,
)
def download_model(model_id: str = DEFAULT_MODEL) -> dict:
    """Download and cache model weights to Modal Volume.

    Run once: `modal run vex_coordizer_harvest_modal.py::download_model`
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Downloading model: {model_id}")
    save_path = f"{MODEL_DIR}/{model_id.replace('/', '_')}"

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.save_pretrained(save_path)

    model = AutoModelForCausalLM.from_pretrained(model_id)
    model.save_pretrained(save_path)

    model_volume.commit()

    print(f"Model saved to {save_path}")
    return {"status": "ok", "model_id": model_id, "path": save_path}


# ─── GPU Harvest Endpoint ──────────────────────────────────────────

@app.cls(
    image=harvest_image,
    gpu="A10G",
    timeout=600,
    container_idle_timeout=300,
    volumes={MODEL_DIR: model_volume},
)
class CoordizerHarvester:
    """GPU-backed harvester for CoordizerV2.

    Loads model on container start, serves harvest requests via
    authenticated FastAPI endpoint. Returns full softmax distributions.
    """

    model_id: str = DEFAULT_MODEL

    @modal.enter()
    def load_model(self):
        """Load model to GPU on container start."""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        save_path = f"{MODEL_DIR}/{self.model_id.replace('/', '_')}"

        try:
            # Try loading from cached volume first
            print(f"Loading cached model from {save_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(save_path)
            self.model = AutoModelForCausalLM.from_pretrained(
                save_path,
                device_map="auto",
                torch_dtype=torch.bfloat16,
            )
        except Exception:
            # Fall back to downloading from HuggingFace
            print(f"Cache miss, downloading {self.model_id} from HuggingFace")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                device_map="auto",
                torch_dtype=torch.bfloat16,
            )
            # Cache for next time
            self.tokenizer.save_pretrained(save_path)
            self.model.save_pretrained(save_path)
            model_volume.commit()

        self.model.eval()
        self.vocab_size = self.tokenizer.vocab_size
        print(f"Model loaded: {self.model_id}, vocab_size={self.vocab_size}")

    @modal.fastapi_endpoint(method="POST", requires_proxy_auth=True)
    async def harvest(self, request: dict) -> dict:
        """Harvest output distributions from text prompts.

        Request:
            {
                "texts": ["sentence 1", "sentence 2", ...],
                "max_length": 512  (optional)
            }

        Response:
            {
                "success": true,
                "vocab_size": 32000,
                "results": [
                    {
                        "tokens": [token_id, ...],
                        "logits": [[prob, prob, ...], ...],
                        "token_strings": ["token", ...]
                    },
                    ...
                ],
                "elapsed_seconds": 1.23
            }

        Each logits[i] is the full softmax distribution at position i,
        representing the model's prediction of what follows token[i].
        These are points on Δ^(V-1) — ready for Fréchet mean computation
        and Fisher-Rao PGA compression to Δ⁶³.
        """
        import numpy as np
        import torch

        texts = request.get("texts", [])
        max_length = request.get("max_length", 512)

        if not texts:
            return {"success": False, "error": "No texts provided"}

        start_time = time.time()
        results = []

        for text in texts:
            try:
                input_ids = self.tokenizer.encode(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=max_length,
                ).to(self.model.device)

                with torch.no_grad():
                    outputs = self.model(input_ids)
                    logits = outputs.logits[0]  # (seq_len, vocab_size)

                # Full softmax → probabilities on Δ^(V-1)
                probs = torch.softmax(logits, dim=-1).cpu().numpy()

                # Token IDs for this sequence
                token_ids = input_ids[0].cpu().numpy().tolist()

                # Token strings
                token_strings = [
                    self.tokenizer.decode([tid]) for tid in token_ids
                ]

                # Return probabilities as nested lists
                # Each probs[i] is the full distribution after token[i]
                results.append({
                    "tokens": token_ids,
                    "logits": probs.tolist(),
                    "token_strings": token_strings,
                })

            except Exception as e:
                results.append({
                    "tokens": [],
                    "logits": [],
                    "token_strings": [],
                    "error": str(e),
                })

        elapsed = time.time() - start_time

        return {
            "success": True,
            "vocab_size": self.vocab_size,
            "results": results,
            "elapsed_seconds": round(elapsed, 3),
            "n_texts": len(texts),
        }

    @modal.fastapi_endpoint(method="GET", requires_proxy_auth=True)
    async def health(self) -> dict:
        """Health check endpoint."""
        return {
            "status": "ok",
            "model_id": self.model_id,
            "vocab_size": self.vocab_size,
            "gpu": "A10G",
        }
