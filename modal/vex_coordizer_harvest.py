"""
Modal GPU Function — CoordizerV2 Harvest Endpoint

Runs Qwen3.5-35B-A3B (MoE: 35B total, 3B active, 256 experts) in NF4
on A10G GPU, providing full probability distributions for coordizer
fingerprint computation.

Architecture:
    - Gated DeltaNet (linear attention) + Gated Attention hybrid
    - 248,320 vocab (padded), 262K native context
    - 256 experts, 8 routed + 1 shared per layer
    - Requires transformers >= 4.51.0 (qwen3_5_moe architecture)

Deploy:
    modal deploy modal/vex_coordizer_harvest.py

    After deploy, Modal prints the endpoint URL. Update Railway env:
        MODAL_HARVEST_URL=https://garyocean478--vex-coordizer-harvest-<hash>.modal.run

Endpoint:
    POST / — protected by X-Api-Key header (KERNEL_API_KEY).
    Accepts JSON body matching HarvestRequest schema.

Cost estimate:
    A10G: ~$0.000306/sec

CRITICAL: Returns full V-dimensional distributions (V=248,320), NOT top-k.
The tail of the distribution carries geometric information that
top-k approximations destroy.

Model loading strategy:
    Qwen3.5-35B-A3B is 35B total params (MoE, 3B active).
    In bfloat16: ~72GB — exceeds A10G VRAM (24GB).
    In 4-bit (bitsandbytes NF4): ~18GB — fits A10G with 6GB headroom.
    Logits are cast to float32 before numpy conversion.

Vocab change from GLM-4.7-Flash:
    GLM-4.7-Flash vocab: 65,536
    Qwen3.5-35B-A3B vocab: 248,320
    Fingerprint dimension increases ~4x. Compress pipeline (PGA) handles
    this naturally — projects to n=32 lens regardless of V.
"""

from __future__ import annotations

import os

from starlette.requests import Request

import modal

# --- Configuration --------------------------------------------------------
HARVEST_MODEL_ID = os.environ.get("HARVEST_MODEL_ID", "Qwen/Qwen3.5-35B-A3B")
HARVEST_GPU_TYPE = os.environ.get("HARVEST_GPU_TYPE", "A10G")
KERNEL_API_KEY = os.environ.get("KERNEL_API_KEY", "")

app = modal.App("vex-coordizer-harvest")

# Persistent volume for model weights (cached across cold starts)
model_volume = modal.Volume.from_name("vex-models", create_if_missing=True)

# Image with ML dependencies
# Requires transformers >= 4.51.0 for qwen3_5_moe architecture support
ml_image = modal.Image.debian_slim(python_version="3.12").pip_install(
    "torch>=2.1",
    "transformers>=4.51.0",
    "accelerate",
    "bitsandbytes>=0.43.0",
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

    Loads Qwen3.5-35B-A3B in 4-bit (NF4) quantization so the full 35B MoE
    model fits in A10G VRAM.

    Vocab size: 248,320 — fingerprints are V-dimensional probability
    distributions on the full vocabulary simplex.
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

        # 4-bit NF4 quantization: ~18GB on A10G (24GB VRAM)
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
            torch_dtype=torch.float16,
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
            torch_dtype=torch.float16,
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
    async def harvest(self, request: Request):
        """GPU harvest endpoint (X-Api-Key protected).

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
                "model_id": "Qwen/Qwen3.5-35B-A3B",
                "vocab_size": 248320,
                "total_tokens_processed": 12345,
                "tokens": { "42": { "string": "hello", "fingerprint": [...], "context_count": 15 } },
                "elapsed_seconds": 45.2
            }
        """
        import time

        import numpy as np
        import torch
        from starlette.responses import JSONResponse

        # Auth
        if KERNEL_API_KEY:
            provided_key = request.headers.get("x-api-key", "")
            if provided_key != KERNEL_API_KEY:
                reason = "missing" if not provided_key else "invalid"
                print(f"Auth failure ({reason} api key)")
                return JSONResponse(
                    status_code=403,
                    content={"error": "invalid or missing api key"},
                )

        body = await request.json()
        start = time.time()

        texts = body.get("texts", [])
        requested_model = body.get("model_id", None)
        batch_size = body.get("batch_size", 32)
        max_length = body.get("max_length", 512)
        min_contexts = body.get("min_contexts", 5)
        target_tokens_raw = body.get("target_tokens", 0)
        try:
            target_tokens = max(0, int(target_tokens_raw))
        except (TypeError, ValueError):
            target_tokens = 0

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
                return {"success": False, "error": f"Failed to load model {target_model_id}: {e}"}

        EPS = 1e-12

        # Accumulators: sqrt-space averaging for Frechet mean
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

        # Compute Frechet means and build response
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
        quantization_config=bnb_config,
    )

    model_volume.commit()
    print(f"Done. Model cached at {cache_dir} (4-bit NF4)")
