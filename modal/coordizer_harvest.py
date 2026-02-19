"""
Modal GPU Function for Vex Coordizer Harvesting

Deploys a GPU-backed web endpoint that:
1. Loads a language model from Modal Volume (cached)
2. Runs inference on provided prompts
3. Captures full probability distributions (logits)
4. Returns vocabulary tokens and their probability distributions

This function is called by Railway's gpu_harvest.py via HTTP POST.
"""

from pathlib import Path
from typing import Any

import modal

# ═══════════════════════════════════════════════════════════════
# MODAL CONFIGURATION
# ═══════════════════════════════════════════════════════════════

# Create or retrieve persistent Volume for model caching
volume = modal.Volume.from_name("vex-coordizer-models", create_if_missing=True)
MODEL_DIR = Path("/models")

# Define image with dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.5.1",
        "transformers==4.46.3",
        "accelerate==1.2.1",
        "huggingface_hub==0.26.5",
        "fastapi[standard]==0.115.6",
        "pydantic==2.10.5",
    )
    .env({"HF_XET_HIGH_PERFORMANCE": "1"})  # Fast Hugging Face downloads
)

app = modal.App(name="vex-coordizer-harvest", image=image)

# ═══════════════════════════════════════════════════════════════
# MODEL DOWNLOADER (run once to populate Volume)
# ═══════════════════════════════════════════════════════════════


@app.function(
    volumes={MODEL_DIR: volume},
    timeout=3600,  # 1 hour for large model downloads
)
def download_model(
    model_id: str = "meta-llama/Llama-3.2-3B",
    revision: str | None = None,
):
    """Download model to Modal Volume (one-time setup).

    Usage:
        modal run vex_coordizer_harvest_modal.py::download_model \
            --model-id "meta-llama/Llama-3.2-3B"
    """
    from huggingface_hub import snapshot_download

    print(f"📥 Downloading {model_id} to {MODEL_DIR}...")
    snapshot_download(
        repo_id=model_id,
        local_dir=MODEL_DIR / model_id.replace("/", "--"),
        revision=revision,
    )
    volume.commit()  # Persist to Volume
    print(f"✅ Model cached to Volume: {model_id}")


# ═══════════════════════════════════════════════════════════════
# HARVEST ENDPOINT (GPU-backed web endpoint)
# ═══════════════════════════════════════════════════════════════


@app.cls(
    gpu="A10G",  # Nvidia A10G: $0.000306/sec = $1.10/hour
    volumes={MODEL_DIR: volume},
    container_idle_timeout=300,  # Keep warm for 5 minutes
    timeout=600,  # 10 minutes max per request
)
class CoordizerHarvester:
    """GPU-backed coordizer harvester.

    Loads model once on container startup, then serves multiple
    harvest requests from the same container.
    """

    @modal.enter()
    def load_model(self):
        """Load model once on container startup (cached from Volume)."""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_id = "meta-llama/Llama-3.2-3B"
        model_path = MODEL_DIR / model_id.replace("/", "--")

        print(f"🔄 Loading model from {model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            local_files_only=True,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            local_files_only=True,
            device_map="auto",
            torch_dtype=torch.float16,  # Half precision for speed
        )
        self.model.eval()  # Inference mode
        print(f"✅ Model loaded: {model_id}")

    @modal.method()
    @modal.fastapi_endpoint(method="POST", requires_proxy_auth=True)
    def harvest(self, data: dict[str, Any]) -> dict[str, Any]:
        """Run coordizer harvest: capture vocabulary probability distributions.

        Request body:
            {
                "prompts": ["The nature of consciousness is", ...],
                "target_tokens": 2000,
                "batch_size": 32
            }

        Response:
            {
                "success": true,
                "vocab_size": 2000,
                "tokens": ["token_0", "token_1", ...],
                "logits": [[0.1, 0.2, ...], ...],  # Raw logits for each token
                "elapsed_seconds": 45.2
            }
        """
        import time

        import torch

        start_time = time.time()

        # Extract request parameters
        prompts = data.get("prompts", [])
        target_tokens = data.get("target_tokens", 2000)
        batch_size = data.get("batch_size", 32)

        if not prompts:
            return {
                "success": False,
                "error": "No prompts provided",
            }

        print(
            f"🎯 Harvest request: {len(prompts)} prompts, "
            f"target={target_tokens}, batch_size={batch_size}"
        )

        # Collect vocabulary tokens and their logits
        vocab_tokens = []
        vocab_logits = []
        seen_tokens = set()

        with torch.no_grad():
            for prompt in prompts:
                # Tokenize prompt
                inputs = self.tokenizer(
                    prompt,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512,
                ).to(self.model.device)

                # Run inference
                outputs = self.model(**inputs, output_hidden_states=False)
                logits = outputs.logits[0, -1, :]  # Last token logits

                # Get top-k tokens by probability
                top_k = min(target_tokens, logits.shape[0])
                top_logits, top_indices = torch.topk(logits, k=top_k)

                # Collect unique tokens
                for idx, logit_value in zip(
                    top_indices.cpu().tolist(),
                    top_logits.cpu().tolist(),
                ):
                    if idx not in seen_tokens:
                        token_str = self.tokenizer.decode([idx])
                        vocab_tokens.append(token_str)
                        vocab_logits.append(logit_value)
                        seen_tokens.add(idx)

                    if len(vocab_tokens) >= target_tokens:
                        break

                if len(vocab_tokens) >= target_tokens:
                    break

        elapsed = time.time() - start_time
        print(
            f"✅ Harvest complete: {len(vocab_tokens)} tokens in {elapsed:.1f}s"
        )

        return {
            "success": True,
            "vocab_size": len(vocab_tokens),
            "tokens": vocab_tokens,
            "logits": vocab_logits,
            "elapsed_seconds": elapsed,
        }


# ═══════════════════════════════════════════════════════════════
# DEPLOYMENT
# ═══════════════════════════════════════════════════════════════

# Deploy with:
#   modal deploy vex_coordizer_harvest_modal.py
#
# The endpoint URL will be printed (e.g., https://harvest-dev.modal.run)
# Add requires_proxy_auth=True to secure it with Modal-Key/Modal-Secret headers
