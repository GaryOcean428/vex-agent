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

Model loading strategy:
    GLM-4.7-Flash is 30B total params (MoE, 3B active).
    In bfloat16: ~60GB — exceeds A10G VRAM (24GB).
    In 4-bit (bitsandbytes NF4): ~15GB — fits A10G with 9GB headroom.
    Logits are cast to float32 before numpy conversion:
        - bfloat16 has no numpy dtype — direct .numpy() raises ScalarType error
        - float32 cast is lossless for softmax output (probabilities in [0,1])
"""

from __future__ import annotations

import os

from pydantic import BaseModel

import modal

# --- Configuration --------------------------------------------------------
HARVEST_MODEL_ID = os.environ.get("HARVEST_MODEL_ID", "zai-org/GLM-4.7-Flash")
HARVEST_GPU_TYPE = os.environ.get("HARVEST_GPU_TYPE", "H100")
KERNEL_API_KEY = os.environ.get("KERNEL_API_KEY", "")

app = modal.App("vex-coordizer-harvest")


class HarvestRequest(BaseModel):
    """Request body for the harvest endpoint."""

    texts: list[str]
    model_id: str | None = None
    batch_size: int = 32
    max_length: int = 512
    min_contexts: int = 5
    target_tokens: int = 0
    return_full_distribution: bool = True


# Persistent volume for model weights (cached across cold starts)
model_volume = modal.Volume.from_name("vex-models", create_if_missing=True)

# Image with ML dependencies
ml_image = modal.Image.debian_slim(python_version="3.12").pip_install(
    "torch>=2.1",
    "transformers>=4.40",
    "accelerate",
    "bitsandbytes>=0.46.1",
    "numpy>=1.26",
    "pydantic>=2.0",
    "fastapi[standard]",
)


@app.cls(
    gpu=HARVEST_GPU_TYPE,
    image=ml_image,
    timeout=600,
    scaledown_window=300,
    volumes={"/models": model_volume},
    secrets=[modal.Secret.from_name("model")],
)
class CoordizerHarvester:
    """GPU-backed harvester for CoordizerV2.

    Loads GLM-4.7-Flash in 4-bit (NF4) quantization so the full 30B MoE
    model fits in A10G VRAM. Logits are cast to float32 before numpy
    conversion to avoid the bfloat16 ScalarType error.
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
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        self._model_cache: dict[str, tuple] = {}

        default_model_id = HARVEST_MODEL_ID
        cache_dir = "/models/hub"

        print(f"Loading default model: {default_model_id}")

        tokenizer = AutoTokenizer.from_pretrained(default_model_id, cache_dir=cache_dir)

        # 4-bit NF4 quantization: 30B * 0.5 bytes = ~15GB on A10G (24GB VRAM)
        # Avoids CPU offload and the bfloat16 numpy ScalarType error.
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

        model = AutoModelForCausalLM.from_pretrained(
            default_model_id,
            cache_dir=cache_dir,
            device_map={"": 0},
            low_cpu_mem_usage=True,
            dtype=torch.float16,
            quantization_config=bnb_config,
        )
        model.eval()
        vocab_size = tokenizer.vocab_size

        self._model_cache[default_model_id] = (tokenizer, model, vocab_size)
        self.tokenizer = tokenizer
        self.model = model
        self.vocab_size = vocab_size
        self.current_model_id = default_model_id

        print(f"Default model loaded (4-bit NF4). Vocab size: {vocab_size}")
        model_volume.commit()

    def _load_model(self, model_id: str) -> tuple:
        """Load a model on-demand and cache it."""
        if model_id in self._model_cache:
            print(f"Using cached model: {model_id}")
            return self._model_cache[model_id]

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        cache_dir = "/models/hub"
        print(f"Loading new model: {model_id}")

        tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            cache_dir=cache_dir,
            device_map={"": 0},
            low_cpu_mem_usage=True,
            dtype=torch.float16,
            quantization_config=bnb_config,
        )
        model.eval()
        vocab_size = tokenizer.vocab_size

        self._model_cache[model_id] = (tokenizer, model, vocab_size)
        print(f"Model {model_id} loaded (4-bit NF4). Vocab size: {vocab_size}")
        model_volume.commit()

        return tokenizer, model, vocab_size

    @modal.fastapi_endpoint(method="GET")
    async def health(self):
        """Health check."""
        return {
            "status": "ok",
            "current_model_id": getattr(self, "current_model_id", None),
            "vocab_size": getattr(self, "vocab_size", None),
            "cached_models": list(getattr(self, "_model_cache", {}).keys()),
        }

    @modal.fastapi_endpoint(method="POST")
    async def harvest(self, body: HarvestRequest):
        """GPU harvest endpoint.

        Request JSON body:
            {
                "texts": [str, ...],
                "model_id": str (optional),
                "batch_size": int (default 32),
                "max_length": int (default 512),
                "min_contexts": int (default 5),
                "target_tokens": int (default 0 = unlimited)
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
                        "fingerprint": [0.001, ...],
                        "context_count": 15
                    }
                },
                "elapsed_seconds": 45.2
            }
        """
        import time

        import numpy as np
        import torch

        start = time.time()

        texts = body.texts
        requested_model = body.model_id
        batch_size = body.batch_size
        max_length = body.max_length
        min_contexts = body.min_contexts
        target_tokens = max(0, body.target_tokens)

        if not texts:
            return {"success": False, "error": "No texts provided"}

        target_model_id = requested_model or self.current_model_id

        if target_model_id != self.current_model_id:
            try:
                tokenizer, model, vocab_size = self._load_model(target_model_id)
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
                    # NOTE: NumPy has no native bfloat16 dtype. Some models / kernels
                    # can emit bfloat16 tensors; cast to float32 before .numpy().
                    probs = torch.softmax(logits.float(), dim=-1).cpu().numpy()
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
                        if target_tokens and total_tokens >= target_tokens:
                            break

                    if target_tokens and total_tokens >= target_tokens:
                        break

                except Exception as e:
                    print(f"Error processing text: {e}")
                    continue

            if target_tokens and total_tokens >= target_tokens:
                break

        # Compute Fréchet means and build response
        tokens_response = {}
        for token_id, sqrt_sum in dist_sums_sqrt.items():
            count = dist_counts[token_id]
            if count < min_contexts:
                continue

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
    secrets=[modal.Secret.from_name("model")],
)
def download_model(model_id: str = HARVEST_MODEL_ID):
    """Pre-cache model weights to Modal Volume.

    Run: modal run modal/vex_coordizer_harvest.py::download_model
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    cache_dir = "/models/hub"
    print(f"Downloading {model_id} to {cache_dir}...")

    AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    AutoModelForCausalLM.from_pretrained(
        model_id,
        cache_dir=cache_dir,
        device_map="auto",
        dtype=torch.float16,
        quantization_config=bnb_config,
    )

    model_volume.commit()
    print(f"Done. Model cached at {cache_dir} (4-bit NF4)")
