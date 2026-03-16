"""
Modal GPU Function — CoordizerV2 Harvest + PGA Compress

Runs Qwen3.5-4B (dense, 4B params) in NF4 on A10G GPU (~2GB VRAM).
Computes full V-dimensional probability distributions AND runs PGA
compress on-GPU, returning only 64D basin coords + 32D lens coords.

This eliminates the V8 string limit issue — no 248K-float fingerprints
are ever serialized to JSON. All heavy compute stays on GPU.

Deploy:
    modal deploy modal/vex_coordizer_harvest.py

Endpoints:
    POST /harvest — Raw fingerprints (small requests only, for debugging).
    POST /coordize — Full pipeline: text -> fingerprints -> PGA -> 64D basin coords.
    GET  /health  — Health check.

Lens dimension: 32 (from eigenvalue analysis: cumulative variance 0.7661 at dim 32).
Basin dimension: 64 (frozen). First 32 dims from PGA, rest zero-padded, unit normalized.

Model persistence:
    Weights cached on Modal Volume "vex-models" — persists across deploys.
    QLoRA adapter loaded from /models/adapters/harvest-qlora if available.
    This model is the harvest substrate — it evolves with every training run.
"""

import os

import modal

# --- Configuration --------------------------------------------------------
HARVEST_MODEL_ID = os.environ.get("HARVEST_MODEL_ID", "Qwen/Qwen3.5-4B")
HARVEST_GPU_TYPE = os.environ.get("HARVEST_GPU_TYPE", "A10G")
KERNEL_API_KEY = os.environ.get("KERNEL_API_KEY", "")
BASIN_DIM = 64  # frozen
LENS_DIM = 32  # from eigenvalue analysis
ADAPTER_PATH = "/models/adapters/harvest-qlora"

app = modal.App("vex-coordizer-harvest")

model_volume = modal.Volume.from_name("vex-models", create_if_missing=True)

ml_image = modal.Image.debian_slim(python_version="3.12").pip_install(
    "torch>=2.1",
    "transformers>=4.48.0",
    "accelerate",
    "bitsandbytes>=0.43.0",
    "peft>=0.13.0",
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
        if KERNEL_API_KEY:
            print(f"KERNEL_API_KEY loaded: {KERNEL_API_KEY[:4]}...{KERNEL_API_KEY[-4:]}")
        else:
            print("WARNING: KERNEL_API_KEY not set — harvest endpoint is unauthenticated.")

        import json
        from pathlib import Path

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
            dtype=torch.float16,
            quantization_config=bnb_config,
        )

        # --- QLoRA adapter loading ---
        # If a trained adapter exists, load and merge it into the base model.
        # This is the consumer side of the fine-tuning loop:
        #   vex_qlora_train.py trains → saves adapter → next cold start loads it here.
        self.adapter_loaded = False
        self.adapter_meta = None
        adapter_config_path = Path(ADAPTER_PATH) / "adapter_config.json"

        if adapter_config_path.exists():
            try:
                from peft import PeftModel

                print(f"QLoRA adapter found at {ADAPTER_PATH}, loading...")
                model = PeftModel.from_pretrained(model, ADAPTER_PATH)
                # Merge adapter into base model for faster inference
                # (no adapter overhead at inference time)
                model = model.merge_and_unload()
                self.adapter_loaded = True

                # Load training metadata if available
                meta_path = Path(ADAPTER_PATH) / "training_meta.json"
                if meta_path.exists():
                    with open(meta_path) as f:
                        self.adapter_meta = json.load(f)
                    print(
                        f"Adapter merged. Trained: {self.adapter_meta.get('trained_at', 'unknown')}, "
                        f"loss: {self.adapter_meta.get('train_loss', 'unknown')}, "
                        f"samples: {self.adapter_meta.get('train_samples', 'unknown')}"
                    )
                else:
                    print("Adapter merged (no training metadata found).")
            except Exception as e:
                print(f"WARNING: Failed to load adapter: {e}")
                print("Continuing with base model only.")
                self.adapter_loaded = False
        else:
            print("No QLoRA adapter found — using base model.")

        model.eval()
        vocab_size = tokenizer.vocab_size

        self._model_cache[default_model_id] = (tokenizer, model, vocab_size)
        self.tokenizer = tokenizer
        self.model = model
        self.vocab_size = vocab_size
        self.current_model_id = default_model_id

        print(f"Model ready (4-bit NF4). Vocab size: {vocab_size}")
        model_volume.commit()

    def _check_auth(self, data):
        if KERNEL_API_KEY and data.get("_api_key", "") != KERNEL_API_KEY:
            return {"error": "Invalid API key", "success": False}
        return None

    def _harvest_fingerprints(self, texts, batch_size, max_length, min_contexts, target_tokens):
        """Core harvest: text -> per-token probability distributions on GPU."""
        import numpy as np
        import torch

        tokenizer, model, vocab_size = self.tokenizer, self.model, self.vocab_size

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
                batch = all_input_ids[batch_start : batch_start + batch_size]
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

        # Average fingerprints per token (Frechet mean on simplex via arithmetic mean)
        averaged = {}
        for tid, fp_list in token_fingerprints.items():
            if len(fp_list) < min_contexts:
                continue
            mean_fp = np.mean(fp_list, axis=0)
            mean_fp = mean_fp / mean_fp.sum()
            averaged[tid] = mean_fp

        return averaged, total_tokens, vocab_size

    def _pga_compress(self, fingerprints_dict, lens_dim=LENS_DIM, basin_dim=BASIN_DIM):
        """PGA compress: V-dim fingerprints -> basin coords on GPU via numpy.

        Uses dual Gram trick (N x N instead of V x V) since N << V.
        All operations in sqrt-space (Bhattacharyya embedding of probability simplex).
        """
        import numpy as np

        fps = list(fingerprints_dict.values())
        N = len(fps)
        V = fps[0].shape[0]

        # Global mean in sqrt space
        sqrt_fps = [np.sqrt(np.maximum(fp, 1e-12)) for fp in fps]
        global_mean = np.mean(sqrt_fps, axis=0)
        norm = np.sqrt(np.sum(global_mean**2))
        if norm > 1e-10:
            global_mean /= norm

        # Center in tangent space
        centered = np.array([s - global_mean for s in sqrt_fps])  # (N, V)

        # Dual Gram matrix (N x N)
        G = centered @ centered.T  # (N, N)

        # Eigendecomposition of Gram matrix
        target_dim = min(lens_dim, N, basin_dim)
        eigenvalues, eigenvectors = np.linalg.eigh(G)

        # Sort descending
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Project to basin coords
        # projection = eigenvectors^T @ centered @ global_mean
        proj_on_mean = centered @ global_mean  # (N,)
        lens_coords = np.zeros(target_dim)
        for d in range(target_dim):
            lens_coords[d] = np.dot(eigenvectors[:, d], proj_on_mean)

        basin_coords = np.zeros(basin_dim)
        basin_coords[:target_dim] = lens_coords[:target_dim]

        # Unit normalize
        basin_norm = np.sqrt(np.sum(basin_coords**2))
        if basin_norm > 1e-10:
            basin_coords /= basin_norm

        return {
            "basin_coords": basin_coords.tolist(),
            "lens_coords": lens_coords.tolist(),
            "eigenvalues": eigenvalues[:10].tolist(),
            "pga_dim": target_dim,
            "N": N,
            "V": V,
        }

    @modal.fastapi_endpoint(method="GET")
    def health(self):
        return {
            "status": "ok",
            "model_id": self.current_model_id,
            "vocab_size": self.vocab_size,
            "basin_dim": BASIN_DIM,
            "lens_dim": LENS_DIM,
            "adapter_loaded": self.adapter_loaded,
            "adapter_meta": self.adapter_meta,
        }

    @modal.fastapi_endpoint(method="POST")
    def harvest(self, data: dict):
        """Raw harvest — returns per-token fingerprints.
        WARNING: Large responses. Use /coordize for production."""
        import time

        auth_err = self._check_auth(data)
        if auth_err:
            return auth_err

        texts = data.get("texts", [])
        if not texts:
            return {"error": "No texts provided", "success": False}

        start = time.time()
        averaged, total_tokens, vocab_size = self._harvest_fingerprints(
            texts,
            data.get("batch_size", 32),
            data.get("max_length", 512),
            data.get("min_contexts", 5),
            data.get("target_tokens", 0),
        )

        tokenizer = self.tokenizer
        result_tokens = {}
        for tid, fp in averaged.items():
            token_str = tokenizer.decode([tid])
            result_tokens[token_str] = {
                "token_id": tid,
                "count": 1,
                "fingerprint": fp.tolist(),
            }

        return {
            "success": True,
            "model_id": self.current_model_id,
            "vocab_size": vocab_size,
            "total_tokens_processed": total_tokens,
            "unique_tokens_returned": len(result_tokens),
            "elapsed_seconds": round(time.time() - start, 2),
            "tokens": result_tokens,
        }

    @modal.fastapi_endpoint(method="POST")
    def coordize(self, data: dict):
        """Full pipeline: text -> harvest -> PGA compress -> 64D basin coords.

        This is the production endpoint. No V-dim fingerprints in the response.
        Returns only basin_coords (64D), lens_coords (32D), eigenvalues, metadata.

        Body: {
            texts: string[],
            min_contexts: int (default 1),
            target_tokens: int (default 0 = unlimited),
            max_length: int (default 512),
            lens_dim: int (default 32),
            _api_key: string
        }
        """
        import time

        auth_err = self._check_auth(data)
        if auth_err:
            return auth_err

        texts = data.get("texts", [])
        if not texts:
            return {"error": "No texts provided", "success": False}

        start = time.time()

        # Phase 1: Harvest fingerprints on GPU
        averaged, total_tokens, vocab_size = self._harvest_fingerprints(
            texts,
            data.get("batch_size", 16),
            data.get("max_length", 512),
            data.get("min_contexts", 1),
            data.get("target_tokens", 0),
        )

        if len(averaged) == 0:
            return {"error": "No tokens met min_contexts threshold", "success": False}

        harvest_time = time.time() - start

        # Phase 2: PGA compress on GPU (numpy, but data already in CPU memory)
        pga_start = time.time()
        lens_dim = data.get("lens_dim", LENS_DIM)
        pga_result = self._pga_compress(averaged, lens_dim=lens_dim)
        pga_time = time.time() - pga_start

        total_time = time.time() - start

        return {
            "success": True,
            "basin_coords": pga_result["basin_coords"],
            "lens_coords": pga_result["lens_coords"],
            "eigenvalues": pga_result["eigenvalues"],
            "pga_dim": pga_result["pga_dim"],
            "harvest_meta": {
                "model_id": self.current_model_id,
                "vocab_size": vocab_size,
                "total_tokens_processed": total_tokens,
                "unique_tokens": pga_result["N"],
                "fingerprint_dim": pga_result["V"],
                "harvest_seconds": round(harvest_time, 2),
                "pga_seconds": round(pga_time, 2),
                "adapter_loaded": self.adapter_loaded,
            },
            "elapsed_seconds": round(total_time, 2),
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
