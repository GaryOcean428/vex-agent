"""
Modal GPU Function — CoordizerV2 Harvest Endpoint

Runs Qwen3.5-4B (dense, 4B params) in NF4 on A10G GPU (~2GB VRAM).

Deploy: modal deploy modal/vex_coordizer_harvest.py

Model persistence:
    Weights cached on Modal Volume "vex-models" — persists across deploys.
    Future fine-tuning base (like Matrix/GPT-4.1 fine-tunes).
"""

import os

import modal

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
    async def harvest(self, data: dict):
        """Harvest fingerprints from text.

        Uses Modal's canonical `data: dict` pattern for POST bodies.
        Auth via api_key field in body (not headers — Modal fastapi_endpoint
        doesn't expose raw Request without starlette import at deploy time).

        Body: {
            "texts": [...], "api_key": "...",
            "min_contexts": 5, "max_length": 512,
            "batch_size": 32, "target_tokens": 0
        }
        """
        import time
        import numpy as np
        import torch

        if KERNEL_API_KEY:
            api_key = data.get("api_key", "")
            if api_key != KERNEL_API_KEY:
                return {"error": "Invalid API key", "success": False}

        texts = data.get("texts", [])
        batch_size = data.get("batch_size", 32)
        max_length = data.get("max_length", 512)
        min_contexts = data.get("min_contexts", 5)
        target_tokens = data.get("target_tokens", 0)

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
            for batch_start in range(0, len(all_input_ids), batch_size):
                batch = all_input_ids[batch_start:batch_start + batch_size]
                for input_ids in batch:
                    ids_tensor = torch.tensor([input_ids], device="cuda")
                    outputs = model(ids_tensor)
                    logits = outputs.logits[0]
                    probs = torch.nn.functional.softmax(logits.float(), dim=-1)
                    probs_np = probs.cpu().numpy()

                    for pos in range(len(input_ids)):
                        tid = input_ids[pos]
                        if tid not in token_fingerprints:
                            token_fingerprints[tid] = []
                        token_fingerprints[tid].append(probs_np[pos])
                        total_tokens += 1

                    if target_tokens > 0 and total_tokens >= target_tokens:
                        break
                if target_tokens > 0 and total_tokens >= target_tokens:
                    break

        result_tokens = {}
        for tid, fp_list in token_fingerprints.items():
            if len(fp_list) < min_contexts:
                continue
            mean_fp = np.mean(fp_list, axis=0)
            mean_fp = mean_fp / mean_fp.sum()
            token_str = tokenizer.decode([tid])
            result_tokens[token_str] = {
                "token_id": tid,
                "count": len(fp_list),
                "fingerprint": mean_fp.tolist(),
            }

        elapsed = time.time() - start

        return {
            "success": True,
            "model_id": self.current_model_id,
            "vocab_size": vocab_size,
            "total_tokens_processed": total_tokens,
            "unique_tokens_returned": len(result_tokens),
            "elapsed_seconds": round(elapsed, 2),
            "tokens": result_tokens,
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
        model_id, cache_dir=cache_dir, device_map="auto", quantization_config=bnb_config,
    )
    model_volume.commit()
    print(f"Done. Model cached at {cache_dir} (4-bit NF4)")
