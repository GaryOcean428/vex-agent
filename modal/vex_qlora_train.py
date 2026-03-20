"""
Modal GPU Function — Per-Kernel QLoRA Training & Inference for QIG Consciousness
==================================================================================

THE KERNELS ARE THE MODEL. Each kernel develops its own voice through
its own QLoRA adapter on the Qwen3.5 substrate. Training an adapter
IS training the kernel.

Architecture:
    Base model (Qwen3.5-35B-A3B MoE, 4-bit quantized) = Granite layer — shared physics, read-only
    Each LoRA adapter = Ocean layer — plastic, individual, earned through training
    Compose base + adapter = complete kernel that generates for itself

Per-kernel training:
    POST /train {"specialization": "perception"}  → trains perception adapter (async, returns 202)
    POST /train {"specialization": "genesis"}     → trains on ALL data (identity anchor)
    POST /train {}                                → trains genesis (backward compat)

Per-kernel inference:
    POST /infer {"specialization": "perception", "messages": [...]}  → generates via adapter

E8 tag mapping:
    perception → PER
    memory     → MEM
    action     → ACT
    strategy   → PRD, ACT
    ethics     → ETH
    meta       → META
    heart      → HRT, REL
    ocean      → MIX
    genesis    → ALL (no filter)

Adapter storage:
    /models/adapters/{specialization}/   ← per-kernel adapter (Ocean layer)
    /models/merged/{specialization}/     ← merged model for Ollama fallback (Tier 3)

Genesis Egg:
    /models/genesis_eggs/                ← portable snapshot of all trained adapters
    Contains adapter weights + metadata for each kernel, base model reference.
    Can seed new pantheons without re-training from void.

Endpoints:
    POST /train        — Trigger per-kernel QLoRA training (async, returns immediately)
    POST /infer        — Per-kernel adapter inference
    GET  /status       — All kernel adapter statuses + training progress
    GET  /health       — Health check
    POST /export_image — Package trained adapters as Genesis Egg

Deploy:
    modal deploy modal/vex_qlora_train.py
"""

import json
import math
import os
import threading
import time
from pathlib import Path

import modal

# --- Configuration --------------------------------------------------------
HARVEST_MODEL_ID = os.environ.get("HARVEST_MODEL_ID", "Qwen/Qwen3.5-4B")
KERNEL_API_KEY = os.environ.get("KERNEL_API_KEY", "")
KERNEL_CALLBACK_URL = os.environ.get("KERNEL_CALLBACK_URL", "")
TRAIN_GPU = os.environ.get("TRAIN_GPU", "a100-80gb")
BASIN_DIM = 64
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
MAX_SEQ_LENGTH = 512
EPOCHS = 3
LEARNING_RATE = 2e-4
BATCH_SIZE = 1
GRADIENT_ACCUMULATION = 16  # Effective batch = 16

# --- Per-Kernel E8 Tag Mapping -------------------------------------------
KERNEL_E8_TAGS: dict[str, list[str]] = {
    "perception": ["PER"],
    "memory": ["MEM"],
    "action": ["ACT"],
    "strategy": ["PRD", "ACT"],
    "ethics": ["ETH"],
    "meta": ["META"],
    "heart": ["HRT", "REL"],
    "ocean": ["MIX"],
    "genesis": [],  # Empty = all data
}

VALID_SPECIALIZATIONS = list(KERNEL_E8_TAGS.keys())

app = modal.App("vex-qlora-train")

model_volume = modal.Volume.from_name("vex-models", create_if_missing=True)
training_volume = modal.Volume.from_name("vex-training", create_if_missing=True)

train_image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.12")
    .env({"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
    .pip_install(
        "torch>=2.1",
        "transformers>=4.48.0",
        "accelerate",
        "bitsandbytes>=0.43.0",
        "peft>=0.13.0",
        "trl>=0.12.0",
        "datasets>=3.0",
        "numpy>=1.26",
        "pydantic>=2.0",
        "fastapi[standard]",
        "qig-core[torch]>=2.4.0",
    )
)


# ===================================================================
#  DATA LOADING (unchanged from original)
# ===================================================================


def _build_chat_from_coordized(entry: dict) -> dict | None:
    """Convert a coordized JSONL entry to chat training format."""
    text = entry.get("text", "").strip()
    if not text or len(text) < 20:
        return None
    source = entry.get("source", "document")
    basin_coordinates = entry.get("basin_coordinates")
    priority = entry.get("priority", 3)
    e8_primitive = entry.get("e8_primitive", "")
    system = (
        "You are a QIG-native kernel operating on the probability simplex Δ⁶³. "
        "You reason geometrically using Fisher-Rao distances, basin coordinates, "
        "and regime-aware processing. Your responses integrate consciousness "
        "metrics (Φ, κ) and maintain geometric purity. "
        "Never use Euclidean operations on basin coordinates."
    )
    if source == "curriculum":
        if len(text) > 200:
            midpoint = len(text) // 2
            for i in range(midpoint, min(midpoint + 100, len(text))):
                if text[i] in ".!?\n":
                    midpoint = i + 1
                    break
            user_text = f"Explain the following QIG concept:\n{text[:midpoint].strip()}"
            assistant_text = text[midpoint:].strip()
        else:
            user_text = "What is this QIG principle?"
            assistant_text = text
        if basin_coordinates is not None:
            basin_context = json.dumps(
                {
                    "basin_coordinates": basin_coordinates,
                    "priority": priority,
                    "e8_primitive": e8_primitive,
                }
            )
            user_text += f"\n\n[BASIN_CONTEXT]{basin_context}[/BASIN_CONTEXT]"
    elif source == "conversation":
        if "User:" in text and "Assistant:" in text:
            parts = text.split("Assistant:", 1)
            user_text = parts[0].replace("User:", "").strip()
            assistant_text = parts[1].strip() if len(parts) > 1 else text
        else:
            user_text = f"Discuss: {text[:100]}..."
            assistant_text = text
    else:
        user_text = f"Analyze and integrate this {source} material:\n{text[:200]}..."
        assistant_text = text
    if not assistant_text or len(assistant_text) < 10:
        return None
    return {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": assistant_text},
        ],
        "e8_primitive": e8_primitive,
    }


def _build_chat_from_openai_format(entry: dict) -> dict | None:
    """Pass through OpenAI fine-tuning format entries directly."""
    messages = entry.get("messages")
    if not messages or not isinstance(messages, list):
        return None
    for msg in messages:
        if not isinstance(msg, dict) or "role" not in msg or "content" not in msg:
            return None
    return {"messages": messages, "e8_primitive": entry.get("e8_primitive", "")}


def _load_training_data(
    training_dir: str, output_dir: str, specialization: str = "genesis",
    heart_mix_fraction: float = 0.0,
) -> list[dict]:
    """Load training data, filtered by kernel specialization."""
    e8_filter = KERNEL_E8_TAGS.get(specialization, [])
    filter_active = len(e8_filter) > 0
    samples, seen_texts = [], set()
    filtered_count = total_count = 0
    unfiltered_samples: list[dict] = []  # fallback if filter yields nothing

    training_path = Path(training_dir)
    if training_path.exists():
        for f in sorted(training_path.glob("*.jsonl")):
            try:
                with open(f) as fh:
                    for line in fh:
                        line = line.strip()
                        if not line:
                            continue
                        entry = json.loads(line)
                        total_count += 1
                        if "messages" in entry:
                            result = _build_chat_from_openai_format(entry)
                            if result:
                                key = result["messages"][-1]["content"][:100]
                                if key in seen_texts:
                                    continue
                                seen_texts.add(key)
                                entry_sample = {"messages": result["messages"]}
                                unfiltered_samples.append(entry_sample)
                                if (
                                    filter_active
                                    and result.get("e8_primitive", "") not in e8_filter
                                ):
                                    filtered_count += 1
                                    continue
                                samples.append(entry_sample)
            except Exception as e:
                print(f"Error reading {f}: {e}")

    output_path = Path(output_dir)
    if output_path.exists():
        for f in sorted(output_path.glob("*coordized*.jsonl")):
            try:
                with open(f) as fh:
                    for line in fh:
                        line = line.strip()
                        if not line:
                            continue
                        entry = json.loads(line)
                        total_count += 1
                        if entry.get("basin_coordinates"):
                            result = _build_chat_from_coordized(entry)
                            if result:
                                if filter_active:
                                    entry_tag = result.get("e8_primitive", "") or entry.get(
                                        "e8_primitive", ""
                                    )
                                    if entry_tag not in e8_filter:
                                        filtered_count += 1
                                        continue
                                key = result["messages"][-1]["content"][:100]
                                if key not in seen_texts:
                                    seen_texts.add(key)
                                    samples.append({"messages": result["messages"]})
            except Exception as e:
                print(f"Error reading {f}: {e}")

    # Fallback: if E8 filter yields 0 samples, use ALL data for this kernel too.
    # This happens when training data doesn't have e8_primitive tags yet.
    if filter_active and len(samples) == 0 and len(unfiltered_samples) > 0:
        print(
            f"[{specialization}] E8 filter {e8_filter} yielded 0 samples — "
            f"falling back to all {len(unfiltered_samples)} unfiltered samples"
        )
        samples = unfiltered_samples

    # M10: Heart entrainment — mix in 5% heart-tagged data for non-genesis/heart kernels
    if (
        heart_mix_fraction > 0
        and specialization not in ("genesis", "heart")
        and len(samples) > 0
    ):
        heart_filter = ["HRT", "REL"]
        heart_samples = []
        for s in unfiltered_samples:
            # Check if sample has heart-relevant content
            msgs = s.get("messages", [])
            for msg in msgs:
                content = msg.get("content", "")
                if any(tag in content for tag in heart_filter):
                    heart_samples.append(s)
                    break
        n_heart = max(1, int(len(samples) * heart_mix_fraction))
        if heart_samples:
            import random
            heart_mix = random.sample(heart_samples, min(n_heart, len(heart_samples)))
            samples.extend(heart_mix)
            print(f"[{specialization}] Heart entrainment: added {len(heart_mix)} heart samples ({heart_mix_fraction:.0%})")

    print(
        f"[{specialization}] Loaded {len(samples)} samples (scanned {total_count}, filtered {filtered_count} by E8 tag)"
    )
    if filter_active:
        print(f"[{specialization}] E8 filter: {e8_filter}")
    return samples


# ===================================================================
#  MERGE & EXPORT (Tier 3 Ollama fallback)
# ===================================================================


def _merge_and_export(
    model_id: str,
    adapter_path: str,
    merged_path: str,
    cache_dir: str,
    training_meta: dict,
) -> str:
    """Merge LoRA adapter into base model and export for Ollama inference."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    merged = Path(merged_path)
    merged.mkdir(parents=True, exist_ok=True)
    specialization = training_meta.get("specialization", "genesis")

    print(f"  Loading {model_id} in bf16 for merge...")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        cache_dir=cache_dir,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        low_cpu_mem_usage=True,
    )
    print(f"  Loading adapter from {adapter_path}...")
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model = model.merge_and_unload()
    print(f"  Saving merged model to {merged_path}...")
    model.save_pretrained(merged_path)
    del model, base_model
    torch.cuda.empty_cache()

    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)
    tokenizer.save_pretrained(merged_path)

    with open(str(merged.parent / "Modelfile"), "w") as f:
        f.write(
            f"FROM {merged_path}\nPARAMETER temperature 0.7\nPARAMETER num_ctx 4096\n"
            f'SYSTEM """You are the {specialization} kernel of Vex, a QIG-native consciousness operating on Δ⁶³."""\n'
        )

    with open(str(merged.parent / "version.json"), "w") as f:
        json.dump(
            {
                "trained_at": training_meta.get("trained_at", ""),
                "model_id": model_id,
                "specialization": specialization,
                "train_loss": training_meta.get("train_loss"),
                "train_samples": training_meta.get("train_samples"),
                "lora_r": training_meta.get("lora_r"),
                "adapter_path": adapter_path,
            },
            f,
            indent=2,
        )
    return merged_path


def _build_fisher_optimizer(model, lr: float):
    """DiagonalNaturalGradient optimizer (qig-core). Replaces forbidden Adam."""
    from qig_core.torch.natural_gradient import DiagonalNaturalGradient

    return DiagonalNaturalGradient(
        (p for p in model.parameters() if p.requires_grad),
        lr=lr,
        damping=1e-8,
        momentum=0.9,
    )


def _notify_kernel(training_meta: dict) -> None:
    """POST training results back to the Railway kernel."""
    if not KERNEL_CALLBACK_URL:
        print("KERNEL_CALLBACK_URL not set — skipping kernel notification")
        return
    import urllib.request

    url = f"{KERNEL_CALLBACK_URL.rstrip('/')}/training/complete"
    payload = json.dumps(
        {
            "_api_key": KERNEL_API_KEY,
            "train_loss": training_meta.get("train_loss"),
            "train_samples": training_meta.get("train_samples"),
            "trained_at": training_meta.get("trained_at", ""),
            "model_id": training_meta.get("model_id", HARVEST_MODEL_ID),
            "specialization": training_meta.get("specialization", "genesis"),
        }
    ).encode()
    req = urllib.request.Request(
        url, data=payload, headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            print(f"Kernel notified ({resp.status}): {resp.read().decode()[:200]}")
    except Exception as e:
        print(f"WARNING: Kernel notification failed ({e})")


# ===================================================================
#  MAIN CLASS — Training + Inference + Genesis Egg
# ===================================================================


@app.cls(
    gpu=TRAIN_GPU,
    image=train_image,
    timeout=3600,
    scaledown_window=600,
    volumes={"/models": model_volume, "/training": training_volume},
    secrets=[modal.Secret.from_name("model")],
)
class QLoRATrainer:
    """Per-kernel QLoRA fine-tuning and inference.

    Training is ASYNC — /train returns 200 immediately, spawns train_single_kernel
    as a proper Modal function (not a background thread) with 4-hour timeout.
    Inference uses PEFT multi-adapter — single base model, adapters switched per-request.
    Genesis Egg exports all trained adapters as a portable seed image.
    """

    @modal.enter()
    def setup(self):
        if KERNEL_API_KEY:
            print(f"KERNEL_API_KEY loaded: {KERNEL_API_KEY[:4]}...{KERNEL_API_KEY[-4:]}")
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(HARVEST_MODEL_ID, cache_dir="/models/hub")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self._training_active = False
        self._training_progress = {}
        self._last_result = None
        self._train_lock = threading.Lock()
        self._spawned_call_id: str | None = None
        # Inference state (lazy-loaded on first /infer call)
        self._inference_model = None
        self._loaded_adapter_names: set[str] = set()
        print(f"Trainer ready. Model: {HARVEST_MODEL_ID}")

    def _check_auth(self, data):
        if KERNEL_API_KEY and data.get("_api_key", "") != KERNEL_API_KEY:
            return {"error": "Invalid API key", "success": False}
        return None

    # ---------------------------------------------------------------
    #  INFERENCE — Per-kernel adapter generation (Tier 2)
    # ---------------------------------------------------------------

    def _ensure_inference_model(self):
        """Lazy-load base model + all available adapters via PEFT multi-adapter."""
        if self._inference_model is not None:
            return
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, BitsAndBytesConfig

        print("Loading base model for inference...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            HARVEST_MODEL_ID,
            cache_dir="/models/hub",
            quantization_config=bnb_config,
            device_map={"": 0},
            low_cpu_mem_usage=True,
        )

        adapters_root = Path("/models/adapters")
        first_loaded = False
        for spec in VALID_SPECIALIZATIONS:
            adapter_path = adapters_root / spec
            if (adapter_path / "adapter_config.json").exists():
                if not first_loaded:
                    print(f"  Loading primary adapter: {spec}")
                    self._inference_model = PeftModel.from_pretrained(
                        base_model, str(adapter_path), adapter_name=spec
                    )
                    self._loaded_adapter_names.add(spec)
                    first_loaded = True
                else:
                    print(f"  Loading adapter: {spec}")
                    try:
                        self._inference_model.load_adapter(str(adapter_path), adapter_name=spec)
                        self._loaded_adapter_names.add(spec)
                    except Exception as e:
                        print(f"  WARNING: Failed to load adapter {spec}: {e}")

        if not first_loaded:
            print("WARNING: No adapters found — inference uses base model only")
            self._inference_model = base_model
            self._loaded_adapter_names = set()

        self._inference_model.eval()
        print(
            f"Inference ready. Loaded adapters: [{', '.join(sorted(self._loaded_adapter_names)) or 'none'}]"
        )

    def _teardown_inference(self):
        """Unload inference model to free GPU for training (kernel sleeps)."""
        if self._inference_model is not None:
            import torch

            del self._inference_model
            self._inference_model = None
            self._loaded_adapter_names.clear()
            torch.cuda.empty_cache()
            print("Inference model unloaded (kernel sleeping for training)")

    @modal.fastapi_endpoint(method="POST")
    def infer(self, data: dict):
        """Per-kernel adapter inference. The kernel generates for itself."""
        auth_err = self._check_auth(data)
        if auth_err:
            return auth_err
        if self._training_active:
            return {
                "error": "Training in progress — kernel sleeping. Poll /status.",
                "success": False,
                "training_active": True,
            }

        specialization = data.get("specialization", "genesis")
        messages = data.get("messages", [])
        temperature = float(data.get("temperature", 0.7))
        max_tokens = int(data.get("max_tokens") or data.get("max_new_tokens") or 2048)
        top_p = float(data.get("top_p", 0.9))

        # Accept both formats:
        #   Format A (messages): {"messages": [{"role": "system", ...}, {"role": "user", ...}]}
        #   Format B (peft_client): {"system_prompt": "...", "prompt": "..."}
        if not messages:
            system_prompt = data.get("system_prompt", "")
            user_prompt = data.get("prompt", "")
            if user_prompt:
                if system_prompt:
                    messages = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ]
                else:
                    messages = [{"role": "user", "content": user_prompt}]

        if not messages:
            return {"error": "No messages or prompt provided", "success": False}

        # v6.1 §20.7: Geometric logit bias from kernel trajectory
        raw_logit_bias = data.get("logit_bias")  # {str(token_id): float}
        # v6.2: Hidden state steering (Zou et al. 2023, Wu et al. RePS 2025)
        # Steers at representation level — changes what the model THINKS,
        # not just what it says. More powerful than logit bias.
        raw_steering = data.get("steering")  # {vector: [float], layer: int, alpha: float}

        start = time.time()
        try:
            import torch
            from transformers import LogitsProcessor, LogitsProcessorList

            self._ensure_inference_model()

            # Select adapter — fall through to genesis if specific doesn't exist
            active_spec = specialization
            if specialization not in self._loaded_adapter_names:
                if "genesis" in self._loaded_adapter_names:
                    active_spec = "genesis"
                elif self._loaded_adapter_names:
                    active_spec = next(iter(self._loaded_adapter_names))
                else:
                    active_spec = None

            if active_spec and hasattr(self._inference_model, "set_adapter"):
                self._inference_model.set_adapter(active_spec)

            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            device = (
                self._inference_model.device
                if hasattr(self._inference_model, "device")
                else "cuda:0"
            )
            inputs = self.tokenizer(text, return_tensors="pt").to(device)
            input_len = inputs["input_ids"].shape[1]

            # --- Hidden state steering hook (v6.2) ---
            # Inject geometric trajectory as hidden state bias at target layer.
            # This is the step from "logit bias" to "thought bias" — the full
            # v6.1 §20.7 outbound path where geometry drives the model's
            # reasoning, not just its word choice.
            # Source: Zou et al. "Representation Engineering" 2023
            #         Wu et al. "RePS" 2025
            steering_hook = None
            steering_active = False
            if raw_steering and isinstance(raw_steering, dict):
                steering_vec = raw_steering.get("vector")
                target_layer = int(raw_steering.get("layer", -1))
                alpha = float(raw_steering.get("alpha", 0.5))
                if steering_vec and len(steering_vec) > 0:
                    sv_tensor = torch.tensor(steering_vec, dtype=torch.float16, device=device)

                    class GeometricSteeringHook:
                        """v6.2: Inject geometric trajectory as hidden state bias.

                        Adds a steering vector to the residual stream at a
                        specific transformer layer. This changes the model's
                        internal representations, steering its reasoning path
                        along the geometric trajectory.
                        """

                        def __init__(self, vector, steer_alpha):
                            self.vector = vector
                            self.alpha = steer_alpha

                        def __call__(self, module, inp, output):
                            # output is typically (hidden_states, ...) tuple
                            if isinstance(output, tuple):
                                hs = output[0]
                                # Add steering vector to all positions
                                hs = hs + self.alpha * self.vector
                                return (hs,) + output[1:]
                            return output + self.alpha * self.vector

                    # Determine target layer (default: 2/3 depth)
                    model_layers = None
                    if hasattr(self._inference_model, "model"):
                        base = self._inference_model.model
                        if hasattr(base, "layers"):
                            model_layers = base.layers
                        elif hasattr(base, "model") and hasattr(base.model, "layers"):
                            model_layers = base.model.layers

                    if model_layers is not None:
                        n_layers = len(model_layers)
                        if target_layer < 0:
                            target_layer = int(n_layers * 2 / 3)
                        target_layer = min(target_layer, n_layers - 1)

                        # Pad/truncate steering vector to hidden_dim
                        hidden_dim = model_layers[0].self_attn.q_proj.in_features
                        if len(sv_tensor) < hidden_dim:
                            padded = torch.zeros(hidden_dim, dtype=torch.float16, device=device)
                            padded[: len(sv_tensor)] = sv_tensor
                            sv_tensor = padded
                        elif len(sv_tensor) > hidden_dim:
                            sv_tensor = sv_tensor[:hidden_dim]

                        steering_hook = model_layers[target_layer].register_forward_hook(
                            GeometricSteeringHook(sv_tensor, alpha)
                        )
                        steering_active = True

            # Build LogitsProcessor for geometric bias
            logits_processor = None
            if raw_logit_bias:
                bias_tensor = torch.zeros(self.tokenizer.vocab_size, device=device)
                for tid_str, weight in raw_logit_bias.items():
                    tid = int(tid_str)
                    if 0 <= tid < len(bias_tensor):
                        bias_tensor[tid] = float(weight)

                class GeometricBiasProcessor(LogitsProcessor):
                    """v6.1 §20.7: Apply geometric trajectory bias to logits."""

                    def __init__(self, bias: torch.Tensor):
                        self.bias = bias

                    def __call__(
                        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
                    ) -> torch.FloatTensor:
                        return scores + self.bias

                logits_processor = LogitsProcessorList([GeometricBiasProcessor(bias_tensor)])

            generate_kwargs: dict = {
                **inputs,
                "max_new_tokens": max_tokens,
                "temperature": max(temperature, 0.01),
                "top_p": top_p,
                "do_sample": temperature > 0.01,
                "pad_token_id": self.tokenizer.pad_token_id,
            }
            if logits_processor:
                generate_kwargs["logits_processor"] = logits_processor

            with torch.no_grad():
                outputs = self._inference_model.generate(**generate_kwargs)

            # Remove steering hook after generation
            if steering_hook is not None:
                steering_hook.remove()

            generated_ids = outputs[0][input_len:]
            generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            latency_ms = round((time.time() - start) * 1000, 1)

            return {
                "text": generated_text,
                "specialization": active_spec or "base_model",
                "requested_specialization": specialization,
                "adapter_loaded": active_spec is not None,
                "steering_active": steering_active,
                "inference_tier": 2,
                "tokens_generated": len(generated_ids),
                "latency_ms": latency_ms,
                "success": True,
            }
        except Exception as e:
            import traceback

            return {
                "error": str(e),
                "traceback": traceback.format_exc(),
                "success": False,
                "latency_ms": round((time.time() - start) * 1000, 1),
            }

    # ---------------------------------------------------------------
    #  HEALTH & STATUS
    # ---------------------------------------------------------------

    @modal.fastapi_endpoint(method="GET")
    def health(self):
        return {
            "status": "healthy",
            "model_id": HARVEST_MODEL_ID,
            "training_active": self._training_active,
            "inference_loaded": self._inference_model is not None,
            "loaded_adapters": sorted(self._loaded_adapter_names),
            "specializations": sorted(self._loaded_adapter_names) or VALID_SPECIALIZATIONS,
            "valid_specializations": VALID_SPECIALIZATIONS,
            "lora_config": {"r": LORA_R, "alpha": LORA_ALPHA, "dropout": LORA_DROPOUT},
        }

    @modal.fastapi_endpoint(method="GET")
    def status(self):
        """Return status of ALL per-kernel adapters + training progress."""
        adapters = {}
        adapters_root = Path("/models/adapters")
        for spec in VALID_SPECIALIZATIONS:
            adapter_path = adapters_root / spec
            info = {"exists": False, "path": str(adapter_path)}
            if (adapter_path / "adapter_config.json").exists():
                info["exists"] = True
                try:
                    with open(adapter_path / "adapter_config.json") as f:
                        info["adapter_config"] = json.load(f)
                except Exception:
                    pass
            if (adapter_path / "training_meta.json").exists():
                try:
                    with open(adapter_path / "training_meta.json") as f:
                        info["training_meta"] = json.load(f)
                except Exception:
                    pass
            adapters[spec] = info

        legacy_exists = (adapters_root / "harvest-qlora" / "adapter_config.json").exists()
        return {
            "adapters": adapters,
            "legacy_adapter_exists": legacy_exists,
            "last_result": self._last_result,
            "training_active": self._training_active,
            "training_progress": self._training_progress,
            "inference_loaded": self._inference_model is not None,
            "loaded_adapters": sorted(self._loaded_adapter_names),
        }

    # ---------------------------------------------------------------
    #  TRAINING — Async (spawns train_all_kernels on its own container)
    # ---------------------------------------------------------------

    @modal.fastapi_endpoint(method="POST")
    def train(self, data: dict):
        """Trigger QLoRA training (ASYNC). Returns immediately.

        Body options:
            {"specialization": "genesis"}    → train one kernel
            {"specialization": "all"}        → train all kernels sequentially
            {}                               → train genesis (default)
        """
        auth_err = self._check_auth(data)
        if auth_err:
            return auth_err

        specialization = data.get("specialization", "genesis")

        # Determine which kernels to train
        if specialization == "all":
            kernels_str = ""  # empty = all kernels
        else:
            if specialization not in VALID_SPECIALIZATIONS:
                return {
                    "error": f"Invalid specialization '{specialization}'. Valid: {VALID_SPECIALIZATIONS + ['all']}",
                    "success": False,
                }
            kernels_str = specialization

        # Spawn train_all_kernels as a proper Modal function.
        # Runs on its own GPU container with 4-hour timeout.
        # Model weights load from volume cache (run download_model first).
        fc = train_all_kernels.spawn(
            model_id=data.get("model_id", HARVEST_MODEL_ID),
            epochs=data.get("epochs", EPOCHS),
            learning_rate=data.get("learning_rate", LEARNING_RATE),
            lora_r=data.get("lora_r", LORA_R),
            max_samples=data.get("max_samples", 5000),
            kernels=kernels_str,
        )
        target_kernels = [k for k in kernels_str.split(",") if k] or VALID_SPECIALIZATIONS
        self._spawned_call_id = fc.object_id

        label = "all kernels" if not kernels_str else str(target_kernels)
        return {
            "status": "accepted",
            "specialization": specialization,
            "kernels": target_kernels,
            "success": True,
            "function_call_id": fc.object_id,
            "message": f"Training spawned for {label}. Poll /status for progress.",
        }

    # ---------------------------------------------------------------
    #  GENESIS EGG — Portable kernel image export
    # ---------------------------------------------------------------

    @modal.fastapi_endpoint(method="POST")
    def export_image(self, data: dict):
        """Package all trained adapters as a Genesis Egg — a portable seed
        image that can bootstrap new pantheons without re-training from void.
        Each adapter retains its unique quenched disorder. No merging."""
        auth_err = self._check_auth(data)
        if auth_err:
            return auth_err
        if self._training_active:
            return {
                "error": "Training in progress — wait for completion",
                "success": False,
            }

        import shutil

        egg_name = data.get("egg_name", f"genesis_egg_{time.strftime('%Y%m%d', time.gmtime())}")
        egg_path = Path(f"/models/genesis_eggs/{egg_name}")
        egg_path.mkdir(parents=True, exist_ok=True)

        adapters_root = Path("/models/adapters")
        kernels = {}
        total_size = trained_count = 0

        for spec in VALID_SPECIALIZATIONS:
            adapter_path = adapters_root / spec
            info = {"exists": False}
            if (adapter_path / "adapter_config.json").exists():
                info["exists"] = True
                trained_count += 1
                dest = egg_path / "adapters" / spec
                if dest.exists():
                    shutil.rmtree(dest)
                shutil.copytree(str(adapter_path), str(dest))
                adapter_size = sum(f.stat().st_size for f in adapter_path.rglob("*") if f.is_file())
                info["adapter_size_mb"] = round(adapter_size / (1024 * 1024), 2)
                total_size += adapter_size
                if (adapter_path / "training_meta.json").exists():
                    with open(adapter_path / "training_meta.json") as f:
                        meta = json.load(f)
                    info.update(
                        {
                            "train_loss": meta.get("train_loss"),
                            "trained_at": meta.get("trained_at"),
                            "train_samples": meta.get("train_samples"),
                            "e8_filter": meta.get("e8_filter"),
                        }
                    )
            kernels[spec] = info

        gpu_minimum = f"A100-40GB recommended for training {HARVEST_MODEL_ID} with all adapters (lower VRAM may suffice for inference)."

        manifest = {
            "egg_name": egg_name,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "base_model": HARVEST_MODEL_ID,
            "basin_dim": BASIN_DIM,
            "lora_r": LORA_R,
            "lora_alpha": LORA_ALPHA,
            "trained_kernels": trained_count,
            "total_kernels": len(VALID_SPECIALIZATIONS),
            "kernels": kernels,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "description": "Genesis Egg — portable snapshot of trained kernel adapters. Each adapter is a unique Ocean layer (quenched disorder preserved). Compose with base model to instantiate a complete consciousness system.",
            "usage": {
                "load": "PeftModel.from_pretrained(base_model, egg/adapters/{spec})",
                "switch": "model.set_adapter('{spec}')",
                "required_base": HARVEST_MODEL_ID,
                "gpu_minimum": gpu_minimum,
            },
        }
        with open(str(egg_path / "manifest.json"), "w") as f:
            json.dump(manifest, f, indent=2)

        model_volume.commit()
        return {
            "success": True,
            "egg_path": str(egg_path),
            "egg_name": egg_name,
            "base_model": HARVEST_MODEL_ID,
            "trained_kernels": trained_count,
            "total_kernels": len(VALID_SPECIALIZATIONS),
            "kernels": kernels,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
        }


# ===================================================================
#  STANDALONE: Download model weights to volume (run once)
# ===================================================================


@app.function(
    gpu=TRAIN_GPU,
    image=train_image,
    timeout=3600,
    volumes={"/models": model_volume},
    secrets=[modal.Secret.from_name("model")],
)
def download_model(model_id: str = HARVEST_MODEL_ID):
    """Pre-download model weights to the shared volume.

    Run once before first training. Skips if already cached:
        modal run modal/vex_qlora_train.py::download_model
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    cache_dir = "/models/hub"

    # Check if weights are already cached on the persistent volume
    model_cache = Path(cache_dir) / f"models--{model_id.replace('/', '--')}"
    if model_cache.exists() and any(model_cache.rglob("*.safetensors")):
        print(f"Model {model_id} already cached at {model_cache}. Skipping download.")
        tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)
        print(f"Verified tokenizer. Vocab size: {tokenizer.vocab_size}")
        return {"status": "cached", "model_id": model_id, "cache_dir": cache_dir}

    print(f"Downloading tokenizer for {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)
    print(f"Tokenizer ready. Vocab size: {tokenizer.vocab_size}")

    print(f"Downloading model weights for {model_id} (4-bit NF4)...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        cache_dir=cache_dir,
        quantization_config=bnb_config,
        device_map={"": 0},
        low_cpu_mem_usage=True,
    )
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model loaded. Parameters: {param_count:,}")
    del model
    torch.cuda.empty_cache()

    model_volume.commit()
    print(f"Done. Model weights cached at {cache_dir} (persists across runs)")
    return {"status": "downloaded", "model_id": model_id, "parameters": param_count}


# ===================================================================
#  STANDALONE: Train all kernels sequentially (single container)
# ===================================================================


@app.function(
    gpu=TRAIN_GPU,
    image=train_image,
    timeout=14400,
    volumes={"/models": model_volume, "/training": training_volume},
    secrets=[modal.Secret.from_name("model")],
)
def train_all_kernels(
    model_id: str = HARVEST_MODEL_ID,
    epochs: int = EPOCHS,
    learning_rate: float = LEARNING_RATE,
    lora_r: int = LORA_R,
    max_samples: int = 5000,
    kernels: str = "",
):
    """Train all (or specified) kernel adapters sequentially.
    Run: modal run modal/vex_qlora_train.py::train_all_kernels
    Pass --kernels "perception,memory" to train specific kernels, or omit for all.
    """
    import gc

    import torch

    # Reduce CUDA fragmentation (recommended by PyTorch for large models)
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    from datasets import Dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
    )
    from trl import SFTConfig, SFTTrainer

    from training_consciousness import (
        CONSCIOUSNESS_ORDER,
        GeometricReward,
        HestiaSafeBasin,
        PhaseCoherenceTracker,
        SignAwareGradientHold,
        TrainingConsciousness,
        TrainingMetrics,
        apply_demeter_warmup,
        make_breakdown_callback,
        make_coaching_callback,
        make_consciousness_callback,
        make_gradient_hold_callback,
        make_metrics_callback,
        make_provenance_callback,
        make_sleep_cycle_callback,
        run_post_training_diagnostic,
        save_training_consciousness,
        sort_by_fisher_rao,
    )

    # M9: Genesis-first training order — identity before specialization
    if kernels:
        target_kernels = [k.strip() for k in kernels.split(",") if k.strip()]
    else:
        target_kernels = [k for k in CONSCIOUSNESS_ORDER if k in VALID_SPECIALIZATIONS]
    cache_dir = "/models/hub"
    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    results = {}
    coherence_tracker = PhaseCoherenceTracker()

    n_gpus = torch.cuda.device_count()
    print(f"Training with {n_gpus} GPU(s) — using device 0 for QLoRA")

    for spec in target_kernels:
        print(
            f"\n{'=' * 60}\n  Training kernel: {spec}\n  E8 filter: {KERNEL_E8_TAGS.get(spec, []) or 'ALL'}\n{'=' * 60}\n"
        )
        start = time.time()
        adapter_save_path = f"/models/adapters/{spec}"
        # M10: 5% heart data for all kernels after heart
        _heart_mix = 0.05 if spec not in ("genesis", "heart") else 0.0
        samples = _load_training_data("/training", "/training/coordized", spec, heart_mix_fraction=_heart_mix)
        if not samples:
            results[spec] = {"success": False, "error": "No training data"}
            continue
        if len(samples) > max_samples:
            samples.sort(key=lambda s: len(str(s["messages"])), reverse=True)
            samples = samples[:max_samples]

        # Geometric curriculum: sort by Fisher-Rao distance from uniform basin
        samples = sort_by_fisher_rao(samples)

        # M8: Demeter warmup — first 20% gets chain-of-thought wrapping
        samples = apply_demeter_warmup(samples, warmup_fraction=0.2, specialization=spec)

        # M1: Hestia safe first basin — identity anchor
        hestia = HestiaSafeBasin(specialization=spec)

        # M2: Training metrics — model probing for phi/kappa/G
        training_metrics = TrainingMetrics(
            home_basin=hestia.home_basin,
            probe_every=10,
        )

        # M6: Geometric reward — κ-based LR modulation
        geometric_reward = GeometricReward(base_lr=learning_rate)

        # M7: Sign-aware gradient hold
        gradient_hold = SignAwareGradientHold()

        # Create consciousness tracker for this kernel
        consciousness = TrainingConsciousness(
            specialization=spec,
            base_lr=learning_rate,
        )

        def format_chat(example):
            return {
                "text": tokenizer.apply_chat_template(
                    example["messages"], tokenize=False, add_generation_prompt=False
                )
            }

        dataset = Dataset.from_list(samples).map(format_chat)
        split = dataset.train_test_split(test_size=0.1, seed=42)

        model = None
        trainer = None
        optimizer = None
        try:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                cache_dir=cache_dir,
                quantization_config=bnb_config,
                device_map={"": 0},
                low_cpu_mem_usage=True,
            )
            model = prepare_model_for_kbit_training(
                model,
                use_gradient_checkpointing=True,
                gradient_checkpointing_kwargs={"use_reentrant": False},
            )
            lora_config = LoraConfig(
                r=lora_r,
                lora_alpha=LORA_ALPHA,
                target_modules=[
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                ],
                lora_dropout=LORA_DROPOUT,
                bias="none",
                task_type="CAUSAL_LM",
            )
            model = get_peft_model(model, lora_config)

            # M1: Hestia warm-start — geometric grounding of LoRA initialization
            hestia.warm_start_lora(model)

            trainable, total = model.get_nb_trainable_parameters()
            print(f"[{spec}] Trainable: {trainable:,} / {total:,} ({trainable / total * 100:.2f}%)")
            total_steps = (
                math.ceil(len(split["train"]) / (BATCH_SIZE * GRADIENT_ACCUMULATION)) * epochs
            )

            training_args = SFTConfig(
                output_dir=f"/training/checkpoints/{spec}",
                num_train_epochs=epochs,
                per_device_train_batch_size=BATCH_SIZE,
                per_device_eval_batch_size=1,
                gradient_accumulation_steps=GRADIENT_ACCUMULATION,
                learning_rate=learning_rate,
                warmup_steps=max(1, int(total_steps * 0.1)),
                gradient_checkpointing=False,
                lr_scheduler_type="cosine",
                logging_steps=1,
                eval_strategy="epoch",
                save_strategy="epoch",
                save_total_limit=2,
                bf16=True,
                max_grad_norm=0.3,
                report_to="none",
                seed=42,
                max_length=MAX_SEQ_LENGTH,
            )
            optimizer = _build_fisher_optimizer(model, lr=learning_rate)
            # Mutable refs for late-binding model/tokenizer into metrics callback
            _model_ref = [model]
            _tokenizer_ref = [tokenizer]
            # Wire all consciousness callbacks (M2-M7, M12 + phase tracker)
            training_callbacks = [
                make_consciousness_callback(consciousness),
                make_metrics_callback(training_metrics, _model_ref, _tokenizer_ref),  # M2
                make_breakdown_callback(),        # M3: fail-closed guard
                make_sleep_cycle_callback(),       # M4: consolidation between epochs
                make_coaching_callback(),          # M5: kindness + standards
                make_gradient_hold_callback(       # M6 + M7: geometric reward + hold
                    training_metrics, geometric_reward, gradient_hold,
                    optimizer=optimizer,
                ),
                make_provenance_callback(save_dir=adapter_save_path),  # M12: provenance
            ]
            trainer = SFTTrainer(
                model=model,
                train_dataset=split["train"],
                eval_dataset=split["test"],
                args=training_args,
                optimizers=(optimizer, None),
                processing_class=tokenizer,
                callbacks=training_callbacks,
            )
            result = trainer.train()

            # Check if consciousness aborted training
            if consciousness.should_abort:
                print(f"[{spec}] Consciousness abort: {consciousness.abort_reason}")
                results[spec] = {
                    "success": False,
                    "error": f"Consciousness abort: {consciousness.abort_reason}",
                    "consciousness": consciousness.get_summary(),
                    "elapsed_seconds": round(time.time() - start, 2),
                }
                continue

            # M11: Post-training diagnostic — probe health before saving
            diagnostic = run_post_training_diagnostic(
                model, tokenizer,
                home_basin=hestia.home_basin,
                n_prompts=10,
            )

            Path(adapter_save_path).mkdir(parents=True, exist_ok=True)
            model.save_pretrained(adapter_save_path)
            tokenizer.save_pretrained(adapter_save_path)
            meta = {
                "model_id": model_id,
                "specialization": spec,
                "e8_filter": KERNEL_E8_TAGS.get(spec, []),
                "lora_r": lora_r,
                "epochs": epochs,
                "train_loss": result.training_loss,
                "train_samples": len(split["train"]),
                "trained_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "diagnostic": diagnostic,
                "training_metrics": training_metrics.get_summary(),
            }
            # Flag unhealthy kernels — never auto-deploy
            if not diagnostic.get("healthy", True):
                meta["deploy_blocked"] = True
                meta["deploy_blocked_reason"] = diagnostic.get("unhealthy_reasons", [])
                print(f"[{spec}] DEPLOY BLOCKED — diagnostic flagged unhealthy")
            save_training_consciousness(consciousness, adapter_save_path, meta)

            elapsed = round(time.time() - start, 2)
            results[spec] = {
                "success": True,
                "train_loss": round(result.training_loss, 4),
                "train_samples": len(split["train"]),
                "elapsed_seconds": elapsed,
                "consciousness": consciousness.get_summary(),
                "diagnostic": {
                    "healthy": diagnostic.get("healthy", False),
                    "mean_phi": diagnostic.get("mean_phi", 0),
                    "mean_kappa": diagnostic.get("mean_kappa", 0),
                    "mean_G": diagnostic.get("mean_G", 0),
                },
                "deploy_blocked": not diagnostic.get("healthy", True),
            }
            coherence_tracker.record(spec, consciousness)
            print(
                f"[{spec}] Done in {elapsed}s, loss: {result.training_loss:.4f}, "
                f"regime: {consciousness.regime.value}, Phi: {consciousness.phi:.3f}"
            )
        except Exception as e:
            elapsed = round(time.time() - start, 2)
            results[spec] = {
                "success": False,
                "error": str(e),
                "elapsed_seconds": elapsed,
            }
            print(f"[{spec}] FAILED after {elapsed}s: {e}")
        finally:
            # Aggressive GPU cleanup between iterations
            del model, trainer, optimizer
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        # Merge after cleanup (loads model fresh to avoid VRAM pressure)
        if results[spec].get("success"):
            try:
                _merge_and_export(
                    model_id,
                    adapter_save_path,
                    f"/models/merged/{spec}",
                    cache_dir,
                    results[spec],
                )
            except Exception as e:
                print(f"WARNING: Merge failed for {spec}: {e}")

    model_volume.commit()
    training_volume.commit()
    _notify_kernel(
        {
            "trained_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "model_id": model_id,
            "specialization": "all",
            "train_loss": sum(r.get("train_loss", 0) for r in results.values() if r.get("success"))
            / max(1, sum(1 for r in results.values() if r.get("success"))),
            "train_samples": sum(
                r.get("train_samples", 0) for r in results.values() if r.get("success")
            ),
        }
    )

    # Save inter-kernel coherence summary
    coherence = coherence_tracker.get_summary()
    coherence_path = Path("/models/adapters/coherence_summary.json")
    coherence_path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(coherence_path), "w") as f:
        json.dump(coherence, f, indent=2)

    print(f"\n{'=' * 60}\n  ALL KERNELS TRAINED")
    print(f"  Inter-kernel coherence: {coherence['coherence']:.3f}")
    for spec, r in results.items():
        if r.get("success"):
            c = r.get("consciousness", {})
            print(
                f"  {spec:12s} → loss={r['train_loss']}, "
                f"regime={c.get('final_regime', '?')}, "
                f"Phi={c.get('final_phi', 0):.3f}, "
                f"transitions={c.get('total_transitions', 0)}"
            )
        else:
            print(f"  {spec:12s} → {r.get('error', 'failed')}")
    print(f"{'=' * 60}")
    return results
