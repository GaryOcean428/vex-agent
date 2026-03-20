"""
Modal GPU Function — CoordizerV2 Harvest + PGA Compress

Runs Qwen3.5-35B-A3B (MoE, 35B total / 3B active) in NF4 on A100 GPU (~18GB VRAM).
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
HARVEST_GPU_TYPE = os.environ.get("HARVEST_GPU_TYPE", "a100-80gb")
KERNEL_API_KEY = os.environ.get("KERNEL_API_KEY", "")
BASIN_DIM = 64  # frozen
LENS_DIM = 32  # from eigenvalue analysis
ADAPTER_PATH = "/models/adapters/harvest-qlora"

app = modal.App("vex-coordizer-harvest")

model_volume = modal.Volume.from_name("vex-models", create_if_missing=True)

ml_image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.12")
    .apt_install("g++", "ninja-build")
    .env({"CXX": "g++", "CC": "gcc"})
    .pip_install(
        "torch>=2.1",
        "transformers>=4.48.0",
        "accelerate",
        "bitsandbytes>=0.43.0",
        "peft>=0.13.0",
        "numpy>=1.26",
        "pydantic>=2.0",
        "fastapi[standard]",
        # Qwen3.5 hybrid architecture: linear attention fast path
        "causal-conv1d>=1.4.0",
        "flash-linear-attention",
    )
)


@app.cls(
    gpu=HARVEST_GPU_TYPE,
    image=ml_image,
    timeout=1200,
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
                # Size mismatch = adapter trained on a different base model.
                # DO NOT DELETE — the adapter may be correct for the intended model
                # (e.g., 35B-A3B adapter when HARVEST_MODEL_ID defaults to 4B).
                # Fix: set the Modal secret HARVEST_MODEL_ID to match the adapter.
                if "size mismatch" in str(e):
                    print(
                        f"ADAPTER MODEL MISMATCH at {ADAPTER_PATH}. "
                        f"Current base: {default_model_id}. "
                        f"Adapter was trained on a different model. "
                        f"Fix: set Modal secret 'model' key HARVEST_MODEL_ID "
                        f"to match the adapter's training base."
                    )
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

    def _harvest_fingerprints(
        self, texts, batch_size, max_length, min_contexts, target_resonances,
        *, compute_curvature: bool = False,
    ):
        """Core harvest: text -> per-coordinate probability distributions on GPU.

        When compute_curvature=True, also extracts attention entropy for manifold
        curvature estimation (Mao et al., Scientific Reports Jan 2026).
        Curvature extraction is opt-in because output_attentions adds significant
        GPU memory overhead (O(layers·heads·seq_len²)).
        """
        import numpy as np
        import torch

        tokenizer, model, vocab_size = self.tokenizer, self.model, self.vocab_size

        all_input_ids = []
        for text in texts:
            encoded = tokenizer.encode(text, add_special_tokens=True)
            if len(encoded) > max_length:
                encoded = encoded[:max_length]
            all_input_ids.append(encoded)

        resonance_fingerprints = {}
        # Attention curvature tracking (Mao et al. Sci Reports 2026)
        # R(h) ∝ C × e^(-α × H(attention)) where H = attention entropy
        attention_entropies: dict[int, list[float]] = {}
        total_resonances = 0

        with torch.no_grad():
            for batch_start in range(0, len(all_input_ids), batch_size):
                batch = all_input_ids[batch_start : batch_start + batch_size]
                for input_ids in batch:
                    ids_tensor = torch.tensor([input_ids], device="cuda")
                    outputs = model(
                        ids_tensor,
                        output_attentions=compute_curvature,
                    )
                    logits = outputs.logits[0]

                    # Extract attention entropy per position (opt-in)
                    if compute_curvature and hasattr(outputs, "attentions") and outputs.attentions:
                        seq_len = len(input_ids)
                        n_layers = len(outputs.attentions)
                        # Accumulate entropy on GPU to avoid per-layer CPU sync
                        per_pos_entropy_gpu = torch.zeros(seq_len, device="cuda")
                        for layer_attn in outputs.attentions:
                            attn_probs = layer_attn[0].mean(dim=0)  # (seq_len, seq_len)
                            attn_clamped = torch.clamp(attn_probs, min=1e-12)
                            entropy = -torch.sum(attn_clamped * torch.log(attn_clamped), dim=-1)
                            per_pos_entropy_gpu += entropy
                        per_pos_entropy_gpu /= n_layers
                        # Single CPU transfer at the end
                        per_pos_entropy = per_pos_entropy_gpu.cpu().numpy()
                        del per_pos_entropy_gpu, outputs.attentions
                        for pos in range(len(input_ids)):
                            tid = input_ids[pos]
                            if tid not in attention_entropies:
                                attention_entropies[tid] = []
                            attention_entropies[tid].append(float(per_pos_entropy[pos]))
                    # Linear projection to simplex (no exponential warping — QIG purity)
                    clamped = torch.clamp(logits.float(), min=0.0)
                    raw_row_sums = clamped.sum(dim=-1, keepdim=True)
                    row_sums = raw_row_sums.clamp(min=1e-12)
                    probs = clamped / row_sums
                    # Fallback: if a row is all zeros after clamping, use uniform distribution
                    zero_rows = raw_row_sums == 0
                    if zero_rows.any():
                        vocab_dim = probs.size(-1)
                        uniform_row = torch.full(
                            (vocab_dim,),
                            1.0 / float(vocab_dim),
                            device=probs.device,
                            dtype=probs.dtype,
                        )
                        probs[zero_rows.expand_as(probs)] = uniform_row
                    probs_np = probs.cpu().numpy()

                    for pos in range(len(input_ids)):
                        tid = input_ids[pos]
                        if tid not in resonance_fingerprints:
                            resonance_fingerprints[tid] = []
                        resonance_fingerprints[tid].append(probs_np[pos])
                        total_resonances += 1

                    if target_resonances > 0 and total_resonances >= target_resonances:
                        break
                if target_resonances > 0 and total_resonances >= target_resonances:
                    break

        # Fréchet mean on simplex via sqrt-coordinate averaging
        averaged = {}
        for tid, fp_list in resonance_fingerprints.items():
            if len(fp_list) < min_contexts:
                continue
            # Average in sqrt-space (Bhattacharyya embedding), then back-project
            sqrt_fps = [np.sqrt(np.maximum(fp, 1e-12)) for fp in fp_list]
            mean_sqrt = np.mean(sqrt_fps, axis=0)
            mean_fp = mean_sqrt * mean_sqrt
            mean_fp = mean_fp / mean_fp.sum()
            averaged[tid] = mean_fp

        # Compute per-token curvature from attention entropy
        # R(h) ∝ C × e^(-α × H(attention))  (Mao et al. Sci Reports 2026)
        # High entropy = low curvature (uniform/generic region)
        # Low entropy = high curvature (information-dense region)
        CURVATURE_ALPHA = 1.0
        CURVATURE_C = 1.0
        curvature: dict[int, float] = {}
        for tid, ent_list in attention_entropies.items():
            if tid in averaged:
                mean_entropy = float(np.mean(ent_list))
                curvature[tid] = CURVATURE_C * float(np.exp(-CURVATURE_ALPHA * mean_entropy))

        return averaged, total_resonances, vocab_size, curvature

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
            lens_coords[d] = np.dot(eigenvectors[:, d], proj_on_mean)  # QIG-EXEMPT: tangent space projection at Fréchet mean

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
        """Raw harvest — returns per-coordinate fingerprints.
        WARNING: Large responses. Use /coordize for production."""
        import time

        auth_err = self._check_auth(data)
        if auth_err:
            return auth_err

        texts = data.get("texts", [])
        if not texts:
            return {"error": "No texts provided", "success": False}

        start = time.time()
        averaged, total_resonances, vocab_size, curvature = self._harvest_fingerprints(
            texts,
            data.get("batch_size", 32),
            data.get("max_length", 512),
            data.get("min_contexts", 5),
            data.get("target_resonances", 0),
            compute_curvature=bool(data.get("compute_curvature", False)),
        )

        tokenizer = self.tokenizer
        result_coords = {}
        for tid, fp in averaged.items():
            token_str = tokenizer.decode([tid])
            result_coords[token_str] = {
                "coord_id": tid,
                "count": 1,
                "fingerprint": fp.tolist(),
                "curvature": curvature.get(tid, 0.0),
            }

        return {
            "success": True,
            "model_id": self.current_model_id,
            "vocab_size": vocab_size,
            "total_resonances_processed": total_resonances,
            "unique_coords_returned": len(result_coords),
            "elapsed_seconds": round(time.time() - start, 2),
            "coordinates": result_coords,
        }

    @modal.fastapi_endpoint(method="POST")
    def coordize(self, data: dict):
        """Full pipeline: text -> harvest -> PGA compress -> 64D basin coords.

        This is the production endpoint. No V-dim fingerprints in the response.
        Returns only basin_coords (64D), lens_coords (32D), eigenvalues, metadata.

        Body: {
            texts: string[],
            min_contexts: int (default 1),
            target_resonances: int (default 0 = unlimited),
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
        averaged, total_resonances, vocab_size, curvature = self._harvest_fingerprints(
            texts,
            data.get("batch_size", 16),
            data.get("max_length", 512),
            data.get("min_contexts", 1),
            data.get("target_resonances", 0),
            compute_curvature=bool(data.get("compute_curvature", False)),
        )

        if len(averaged) == 0:
            return {
                "error": "No coordinates met min_contexts threshold",
                "success": False,
            }

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
                "total_resonances_processed": total_resonances,
                "unique_tokens": pga_result["N"],
                "fingerprint_dim": pga_result["V"],
                "harvest_seconds": round(harvest_time, 2),
                "pga_seconds": round(pga_time, 2),
                "adapter_loaded": self.adapter_loaded,
            },
            "curvature": {
                "mean": round(float(sum(curvature.values()) / max(len(curvature), 1)), 6),
                "n_tokens": len(curvature),
                "high_curvature_count": sum(1 for c in curvature.values() if c > 0.5),
                "low_curvature_count": sum(1 for c in curvature.values() if c < 0.1),
            },
            "elapsed_seconds": round(total_time, 2),
        }


@app.function(
    gpu=HARVEST_GPU_TYPE,
    image=ml_image,
    volumes={"/models": model_volume},
    timeout=3600,
    secrets=[modal.Secret.from_name("model")],
)
def download_model(model_id: str = HARVEST_MODEL_ID):
    """Pre-cache model weights to Modal Volume. Skips if already cached.

    Run once before first harvest to avoid long cold starts:
        modal run modal/vex_coordizer_harvest.py::download_model
    """
    from pathlib import Path

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    cache_dir = "/models/hub"

    # Check if weights are already cached on the persistent volume
    model_cache = Path(cache_dir) / f"models--{model_id.replace('/', '--')}"
    if model_cache.exists() and any(model_cache.rglob("*.safetensors")):
        print(f"Model {model_id} already cached at {model_cache}. Skipping download.")
        tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)
        print(f"Verified tokenizer. Vocab size: {tokenizer.vocab_size}")
        return

    print(f"Downloading {model_id} to {cache_dir} (first time — may take 10-20 min)...")

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
        device_map={"": 0},
        quantization_config=bnb_config,
        low_cpu_mem_usage=True,
    )
    model_volume.commit()
    print(f"Done. Model cached at {cache_dir} (4-bit NF4, persists across runs)")
