"""
Modal GPU Function — CoordizerV2 Harvest Endpoint

Runs Qwen3.5-4B (dense, 4B params) in NF4 on A10G GPU (~2GB VRAM),
providing full probability distributions for coordizer fingerprint
computation.

Deploy:
    modal deploy modal/vex_coordizer_harvest.py

Endpoint:
    POST / — protected by X-Api-Key header (KERNEL_API_KEY).

Cost estimate:
    A10G: ~$0.000306/sec

CRITICAL: Returns full V-dimensional distributions, NOT top-k.

Model persistence:
    Weights cached on Modal Volume "vex-models" — persists across deploys.
    This model is the harvest substrate and future fine-tuning base.
"""

import os

import modal
from starlette.requests import Request

# --- Configuration --------------------------------------------------------
HARVEST_MODEL_ID = os.environ.get("HARVEST_MODEL_ID", "Qwen/Qwen3.5-4B")
HARVEST_GPU_TYPE = os.environ.get("HARVEST_GPU_TYPE", "A10G")
KERNEL_API_KEY = os.environ.get("KERNEL_API_KEY", "")

app = modal.App("vex-coordizer-harvest")

model_volume = modal.Volume.from_name("vex-models", create_if_missing=True)

ml_image = modal.Image.debian_slim(python_version="3.12").pip_install(
    "torch>=2.1",
    "transformers>=4.48.0",
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

    @modal.enter()
    def load_model(self):
        if not KERNEL_API_KEY:
            print("WARNING: KERNEL_API_KEY not set — harvest endpoint is unauthenticated.")

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        self._model_cache = {}
        default_model_id = HARVEST_MODEL_ID
        cache_dir = "/models/hub"

        print(f"Loading default model: {default_model_id}")

        tokenizer = AutoTokenizer.from_pretrained(default_model_id, cache_dir=cache_dir)

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

    @modal.fastapi_endpoint(method="GET")
    async def health(self):
        return {"status": "ok", "model_id": self.current_model_id, "vocab_size": self.vocab_size}

    @modal.fastapi_endpoint(method="POST")
    async def harvest(self, request: Request):
        """Harvest fingerprints from text. Returns full V-dim probability distributions."""
        import time
        import numpy as np
        import torch

        if KERNEL_API_KEY:
            api_key = request.headers.get("x-api-key", "")
            if api_key != KERNEL_API_KEY:
                return {"error": "Invalid API key", "success": False}

        body = await request.json()
        texts = body.get("texts", [])
        model_id = body.get("model_id", self.current_model_id)
        batch_size = body.get("batch_size", 32)
        max_length = body.get("max_length", 512)
        min_contexts = body.get("min_contexts", 5)
        target_tokens = body.get("target_tokens", 0)

        if not texts:
            return {"error": "No texts provided", "success": False}

        tokenizer, model, vocab_size = self.tokenizer, self.model, self.vocab_size
        start = time.time()

        all_input_ids = []
        for text in texts:
            encoded = tokenizer.encode(text, add_special_tokens=True)
            if len(encoded) > max_length:
                encoded = encoded[:max_length]
            all_input_ids.append(encoded)

        token_fingerprints = {}
        total_tokens = 0

        with torch.no_grad():
            for batch