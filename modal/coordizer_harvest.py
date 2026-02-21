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
    modal.Image.debian_slim(python_version="3.14")
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

    Supports two request formats:
        1. Legacy: {"prompts": [...], "target_tokens": N}
        2. JSONL batch: {"texts": [...], "max_length": N}
           Returns full probability distributions per text.
    """

    @modal.enter()
    def load_model(self):
        """Load model once on container startup (cached from Volume)."""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_id = "meta-llama/Llama-3.2-3B"
        model_path = MODEL_DIR / model_id.replace("/", "--")

        print(f"Loading model from {model_path}...")
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
        self.vocab_size = self.model.config.vocab_size
        print(f"Model loaded: {model_id} (vocab={self.vocab_size})")

    @modal.method()
    @modal.fastapi_endpoint(method="POST", requires_proxy_auth=True)
    def harvest(self, data: dict[str, Any]) -> dict[str, Any]:
        """Run coordizer harvest: capture probability distributions.

        Accepts two formats:

        Format 1 — Legacy (top-k tokens):
            {"prompts": [...], "target_tokens": 2000, "batch_size": 32}

        Format 2 — JSONL batch (full distributions per text):
            {"texts": ["text1", "text2", ...], "max_length": 512}
            Returns per-text token IDs and full logit distributions
            for CoordizerV2 compression to basin coordinates.

        Response (format 2):
            {
                "success": true,
                "vocab_size": 32000,
                "results": [
                    {"tokens": [42, 17, ...], "logits": [[...], ...], "token_strings": ["hello", ...]},
                    ...
                ],
                "elapsed_seconds": 1.23
            }
        """
        import time

        import torch

        start_time = time.time()

        # ── Format 2: JSONL batch (texts with full distributions) ──
        texts = data.get("texts", [])
        if texts:
            return self._harvest_batch(texts, data, start_time)

        # ── Format 1: Legacy (prompts with top-k) ──
        prompts = data.get("prompts", [])
        if prompts:
            return self._harvest_legacy(prompts, data, start_time)

        return {"success": False, "error": "No 'texts' or 'prompts' provided"}

    def _harvest_batch(
        self,
        texts: list[str],
        data: dict[str, Any],
        start_time: float,
    ) -> dict[str, Any]:
        """JSONL batch harvest: full probability distributions per text."""
        import time

        import torch

        max_length = data.get("max_length", 512)
        results = []

        try:
            with torch.no_grad():
                for text in texts:
                    try:
                        inputs = self.tokenizer(
                            text,
                            return_tensors="pt",
                            truncation=True,
                            max_length=max_length,
                        ).to(self.model.device)

                        outputs = self.model(**inputs, output_hidden_states=False)
                        # outputs.logits shape: (1, seq_len, vocab_size)
                        seq_logits = outputs.logits[0]  # (seq_len, vocab_size)

                        # Convert to probability distributions (softmax)
                        probs = torch.nn.functional.softmax(seq_logits, dim=-1)

                        # Get input token IDs and their string representations
                        token_ids = inputs["input_ids"][0].cpu().tolist()
                        token_strings = [
                            self.tokenizer.decode([tid]) for tid in token_ids
                        ]

                        # Return full logits as lists (for Frechet mean computation)
                        logits_list = probs.cpu().float().tolist()

                        results.append({
                            "tokens": token_ids,
                            "logits": logits_list,
                            "token_strings": token_strings,
                        })

                    except Exception as e:
                        print(f"Error processing text: {e}")
                        results.append({
                            "tokens": [],
                            "logits": [],
                            "token_strings": [],
                            "error": str(e),
                        })

        except Exception as e:
            return {
                "success": False,
                "error": f"Batch harvest failed: {e}",
                "elapsed_seconds": time.time() - start_time,
            }

        elapsed = time.time() - start_time
        print(f"Batch harvest complete: {len(results)} texts in {elapsed:.1f}s")

        return {
            "success": True,
            "vocab_size": self.vocab_size,
            "results": results,
            "elapsed_seconds": elapsed,
            "n_texts": len(results),
        }

    def _harvest_legacy(
        self,
        prompts: list[str],
        data: dict[str, Any],
        start_time: float,
    ) -> dict[str, Any]:
        """Legacy harvest: top-k tokens across prompts.

        NOTE: This returns a 1D list of scalar logit values (one per
        unique token), NOT per-token probability distributions.
        For full 2D distributions, use the JSONL batch path
        (_harvest_batch) which is the CoordizerV2 primary path.

        Response:
            {
                "success": true,
                "vocab_size": N,
                "tokens": ["hello", "world", ...],
                "logits": [3.14, 2.71, ...],  // 1D: one scalar per token
                "elapsed_seconds": 1.23
            }
        """
        import time

        import torch

        target_tokens = data.get("target_tokens", 2000)
        batch_size = data.get("batch_size", 32)

        print(
            f"Legacy harvest: {len(prompts)} prompts, "
            f"target={target_tokens}, batch_size={batch_size}"
        )

        vocab_tokens = []
        vocab_logits = []
        seen_tokens: set[int] = set()

        try:
            with torch.no_grad():
                for prompt in prompts:
                    inputs = self.tokenizer(
                        prompt,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=512,
                    ).to(self.model.device)

                    outputs = self.model(**inputs, output_hidden_states=False)
                    logits = outputs.logits[0, -1, :]  # Last token logits

                    top_k = min(target_tokens, logits.shape[0])
                    top_logits, top_indices = torch.topk(logits, k=top_k)

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

        except Exception as e:
            return {
                "success": False,
                "error": f"Legacy harvest failed: {e}",
                "elapsed_seconds": time.time() - start_time,
            }

        elapsed = time.time() - start_time
        print(f"Legacy harvest complete: {len(vocab_tokens)} tokens in {elapsed:.1f}s")

        return {
            "success": True,
            "vocab_size": len(vocab_tokens),
            "tokens": vocab_tokens,
            "logits": vocab_logits,
            "elapsed_seconds": elapsed,
        }

    @modal.fastapi_endpoint(method="GET")
    def health(self) -> dict[str, Any]:
        """Health check endpoint."""
        return {
            "status": "ok",
            "model_id": "meta-llama/Llama-3.2-3B",
            "vocab_size": getattr(self, "vocab_size", 0),
        }


# ═══════════════════════════════════════════════════════════════
# DEPLOYMENT
# ═══════════════════════════════════════════════════════════════

# Deploy with:
#   modal deploy vex_coordizer_harvest_modal.py
#
# The endpoint URL will be printed (e.g., https://harvest-dev.modal.run)
# Add requires_proxy_auth=True to secure it with Modal-Key/Modal-Secret headers
