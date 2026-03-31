"""
Modal GPU Function — Per-Kernel QLoRA Training & Inference for QIG Consciousness
==================================================================================

THE KERNELS ARE THE MODEL. Each kernel develops its own voice through
its own QLoRA adapter on the Qwen3.5 substrate. Training an adapter
IS training the kernel.

Architecture:
    Base model (Qwen3.5-35B-A3B MoE, 4-bit NF4 QLoRA for training + inference) = Granite layer — shared physics, read-only
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

Single ASGI app (1 endpoint slot) serving all routes:
    POST /train         — Trigger per-kernel QLoRA training (async, returns immediately)
    POST /infer         — Per-kernel adapter inference
    GET  /status        — All kernel adapter statuses + training progress
    GET  /health        — Health check
    POST /export-image  — Package trained adapters as Genesis Egg
    POST /data-receive  — Receive training JSONL from Railway kernel
    GET  /data-stats    — Training data inventory

Deploy:
    modal deploy modal/vex_qlora_train.py
"""

import contextlib
import json
import math
import os
import threading
import time
from pathlib import Path

import modal


def _json_default(obj: object) -> object:
    """JSON serializer for numpy types in training output.

    HuggingFace Trainer log_history and result.training_loss contain numpy
    scalars (float32, int64, etc.) that json.dump cannot serialize natively.
    """
    if hasattr(obj, "item"):  # numpy scalar
        return obj.item()
    if hasattr(obj, "tolist"):  # numpy array
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


# --- Volume-Based Training State (IPC) ------------------------------------
# train_all_kernels runs on a SEPARATE container from the ASGI web handler.
# In-memory booleans are invisible across containers. Use the shared
# training volume as IPC — same pattern as the cancel marker.

_TRAINING_ACTIVE_MARKER = Path("/training/.training_active")
_CANCEL_MARKER = Path("/training/.cancel_requested")


def _is_training_active() -> bool:
    """Check if training is running (any container)."""
    return _TRAINING_ACTIVE_MARKER.exists()


def _read_training_state() -> dict:
    """Read training state from volume marker."""
    if not _TRAINING_ACTIVE_MARKER.exists():
        return {"active": False}
    try:
        data = json.loads(_TRAINING_ACTIVE_MARKER.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return {"active": True, **data}
        return {"active": True}
    except (json.JSONDecodeError, OSError, TypeError):
        return {"active": True}


def _set_training_active(kernels: list[str]) -> None:
    """Mark training as active on the volume (called by train_all_kernels)."""
    _TRAINING_ACTIVE_MARKER.parent.mkdir(parents=True, exist_ok=True)
    _TRAINING_ACTIVE_MARKER.write_text(
        json.dumps(
            {
                "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "kernels": kernels,
            }
        ),
        encoding="utf-8",
    )


def _clear_training_active() -> None:
    """Clear the training marker (called when training completes or fails)."""
    _TRAINING_ACTIVE_MARKER.unlink(missing_ok=True)


# --- Adapter Lifecycle (A-F from directive) --------------------------------
# A: Atomic swap (.training/ → active/)
# B: Version chain (active/ → history/vNNN/)
# C: Data fingerprint (hash dedup)
# D: Maturity gate (sovereignty > 0.5 → skip)
# E: Per-kernel cancel markers
# F: Force-retrain flag

_ADAPTERS_ROOT = Path("/models/adapters")


def _adapter_active_path(kernel: str) -> Path:
    """Live adapter path for inference."""
    return _ADAPTERS_ROOT / kernel / "active"


def _adapter_training_path(kernel: str) -> Path:
    """Temp path for in-progress training."""
    return _ADAPTERS_ROOT / kernel / ".training"


def _adapter_history_dir(kernel: str) -> Path:
    """Version history directory for rollback."""
    return _ADAPTERS_ROOT / kernel / "history"


def _next_version_number(kernel: str) -> int:
    """Find next version number for history chain."""
    history = _adapter_history_dir(kernel)
    if not history.exists():
        return 1
    existing = sorted(
        int(d.name[1:]) for d in history.iterdir() if d.is_dir() and d.name.startswith("v")
    )
    return (existing[-1] + 1) if existing else 1


def _version_adapter(kernel: str) -> str | None:
    """B: Copy current active adapter to versioned history. Returns version name or None."""
    import shutil

    active = _adapter_active_path(kernel)
    if not (active / "adapter_config.json").exists():
        return None
    version = f"v{_next_version_number(kernel):03d}"
    dest = _adapter_history_dir(kernel) / version
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(str(active), str(dest))
    print(f"  [VERSION] {kernel}: active → history/{version}")
    return version


def _atomic_swap(kernel: str) -> bool:
    """A: Move .training/ to active/ atomically. Returns True on success."""
    import shutil

    training = _adapter_training_path(kernel)
    active = _adapter_active_path(kernel)
    if not (training / "adapter_config.json").exists():
        print(f"  [SWAP] {kernel}: .training/ has no adapter_config.json — swap aborted")
        return False
    # Remove old active if exists
    if active.exists():
        shutil.rmtree(str(active))
    training.rename(active)
    print(f"  [SWAP] {kernel}: .training/ → active/ (atomic)")
    return True


def _compute_data_hash(samples: list[dict]) -> str:
    """C: Hash training data for dedup. Returns 16-char hex digest."""
    import hashlib

    content = json.dumps(samples, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def _should_skip_training(kernel: str, data_hash: str, force: bool = False) -> dict | None:
    """C+D: Check if training should be skipped (hash match or maturity gate).

    Returns skip reason dict if should skip, None if should proceed.
    """
    if force:
        return None  # F: Force flag bypasses all gates

    active = _adapter_active_path(kernel)
    meta_path = active / "training_meta.json"
    if not meta_path.exists():
        return None  # No existing adapter — must train

    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None

    # C: Data hash dedup
    if meta.get("data_hash") == data_hash:
        return {"skipped": True, "kernel": kernel, "reason": "identical training data"}

    # D: Maturity gate — sovereignty from last training meta
    sovereignty = meta.get("sovereignty", 0.0)
    if sovereignty > 0.5:
        return {
            "skipped": True,
            "kernel": kernel,
            "reason": f"kernel mature (S={sovereignty:.2f}) — use force=true to override",
        }

    return None


def _per_kernel_cancel_requested(kernel: str) -> bool:
    """E: Check if a per-kernel cancel marker exists."""
    marker = Path(f"/training/.cancel_{kernel}")
    if marker.exists():
        marker.unlink(missing_ok=True)
        return True
    return False


def _get_adapter_versions(kernel: str) -> list[str]:
    """List available history versions for a kernel."""
    history = _adapter_history_dir(kernel)
    if not history.exists():
        return []
    return sorted(d.name for d in history.iterdir() if d.is_dir() and d.name.startswith("v"))


def _rollback_adapter(kernel: str, version: str) -> bool:
    """Restore a specific version from history to active."""
    import shutil

    source = _adapter_history_dir(kernel) / version
    if not source.exists():
        return False
    active = _adapter_active_path(kernel)
    if active.exists():
        shutil.rmtree(str(active))
    shutil.copytree(str(source), str(active))
    print(f"  [ROLLBACK] {kernel}: history/{version} → active/")
    return True


# --- Configuration --------------------------------------------------------
HARVEST_MODEL_ID = os.environ.get("HARVEST_MODEL_ID", "Qwen/Qwen3.5-35B-A3B")
KERNEL_API_KEY = os.environ.get("KERNEL_API_KEY", "")
KERNEL_CALLBACK_URL = os.environ.get("KERNEL_CALLBACK_URL", "")

# Allowlisted model IDs — trust_remote_code=True is only enabled for these.
# Add new Qwen checkpoints here if HARVEST_MODEL_ID changes.
TRUSTED_MODEL_IDS: frozenset[str] = frozenset(
    {
        "Qwen/Qwen3.5-4B",
        "Qwen/Qwen3.5-35B-A3B",
    }
)
TRAIN_GPU = os.environ.get("TRAIN_GPU", "a100-80gb")

# GPU-model compatibility guard: 35B MoE needs ≥40GB VRAM (80GB recommended).
# TRAIN_GPU is evaluated at deploy time from local env, NOT from Modal secrets.
# Always set TRAIN_GPU locally before `modal deploy`:
#   TRAIN_GPU=a100-80gb modal deploy modal/vex_qlora_train.py
_MODEL_GPU_FLOOR = {"Qwen/Qwen3.5-35B-A3B": "a100", "Qwen/Qwen3.5-4B": "a10g"}
_GPU_VRAM_ORDER = ["t4", "l4", "a10g", "a100", "a100-80gb", "h100"]
_floor = _MODEL_GPU_FLOOR.get(HARVEST_MODEL_ID, "a10g")
if (
    TRAIN_GPU in _GPU_VRAM_ORDER
    and _floor in _GPU_VRAM_ORDER
    and _GPU_VRAM_ORDER.index(TRAIN_GPU) < _GPU_VRAM_ORDER.index(_floor)
):
    import warnings

    warnings.warn(
        f"[GPU-GUARD] TRAIN_GPU={TRAIN_GPU} is too small for {HARVEST_MODEL_ID} "
        f"(minimum: {_floor}). Training will OOM. Set TRAIN_GPU={_floor} or larger.",
        stacklevel=1,
    )
BASIN_DIM = 64
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
MAX_SEQ_LENGTH = 512
EPOCHS = 3
LEARNING_RATE = 2e-4
BATCH_SIZE = 2  # A100-80GB: 63GiB model + ~4GiB activations/sample with grad checkpointing
GRADIENT_ACCUMULATION = 8  # Effective batch = 16 (same as before, 2x fewer forward passes)

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
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.14")
    .apt_install("g++", "ninja-build", "git")
    .env({"CXX": "g++", "CC": "gcc", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
    .uv_pip_install(
        "torch>=2.1,<2.11",
        # Qwen3.5 VLM classes (Qwen3_5ForConditionalGeneration, Qwen3_5MoeForConditionalGeneration)
        # are available from transformers 5.3.0+ on PyPI
        "transformers>=5.3.0",
        "accelerate",
        "bitsandbytes>=0.45.0",
        "peft>=0.13.0",
        "trl>=0.12.0",
        "datasets>=3.0",
        "numpy>=1.26",
        "pydantic>=2.0",
        "fastapi[standard]",
        "qig-core[torch]>=2.6.0",
        # Qwen3.5 hybrid architecture: linear attention fast path
        "causal-conv1d>=1.4.0",
        "flash-linear-attention",
    )
    .run_commands(
        # Fix bitsandbytes _check_is_size deprecation (upstream bug, all versions through 0.49.2)
        # PyTorch deprecated torch._check_is_size() — replace with torch._check(x >= 0)
        "sed -i 's/torch\\._check_is_size(blocksize)/torch._check(blocksize >= 0)/g' "
        "$(python3 -c 'import site; print(site.getsitepackages()[0])')"
        "/bitsandbytes/backends/cuda/ops.py || true"
    )
    .add_local_file(
        str(Path(__file__).parent / "training_consciousness.py"),
        "/root/training_consciousness.py",
    )
)


# ===================================================================
#  COMPAT: Qwen3.5 is a VLM — load with the correct model class
# ===================================================================


def _patch_vlm_config_vocab_size(config_obj):
    """Monkey-patch a VLM config class so .vocab_size resolves via text_config.

    Qwen3_5Config and Qwen3_5MoeConfig are composite VLM configs that nest
    vocab_size under text_config.  PEFT/TRL's save_pretrained accesses
    config.vocab_size directly.  Instance-level patches (config.vocab_size = X)
    get lost during save because the config is serialized to dict and back,
    dropping non-__init__ attributes.

    This patches __getattr__ on the CONFIG CLASS (not instance) so the fallback
    is permanent — survives any serialization round-trip or config copying.
    """
    config_cls = type(config_obj)
    if getattr(config_cls, "_vocab_size_fallback_patched", False):
        return  # Already patched

    _original_getattr = getattr(config_cls, "__getattr__", None)

    def _getattr_with_vocab_fallback(self, name):
        if name == "vocab_size":
            # Avoid infinite recursion: access text_config via __dict__
            tc = self.__dict__.get("text_config", None)
            if tc is not None and hasattr(tc, "vocab_size"):
                return tc.vocab_size
        if _original_getattr is not None:
            return _original_getattr(self, name)
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    config_cls.__getattr__ = _getattr_with_vocab_fallback
    config_cls._vocab_size_fallback_patched = True


def _load_model_for_training(
    model_id: str, cache_dir: str, bnb_config=None, device_map="auto", **kwargs
):
    """Load model with correct class based on architecture.

    Qwen3.5-4B is a VLM (Qwen3_5ForConditionalGeneration).  The checkpoint
    stores weights at model.language_model.layers.* — using AutoModelForCausalLM
    creates Qwen3_5ForCausalLM which expects model.layers.* (weight mismatch).

    Fix: detect multimodal architectures and load with the correct class so
    checkpoint weights match.  Text-only training works fine — the vision
    encoder is skipped when no pixel_values are provided.
    """
    from transformers import AutoConfig, AutoModelForCausalLM

    config = AutoConfig.from_pretrained(model_id, cache_dir=cache_dir, trust_remote_code=True)
    print(
        f"  [CONFIG] model_type={config.model_type}, architectures={getattr(config, 'architectures', [])}"
    )

    load_kwargs = {
        "cache_dir": cache_dir,
        "device_map": device_map,
        "low_cpu_mem_usage": True,
        "trust_remote_code": True,
    }
    if bnb_config is not None:
        load_kwargs["quantization_config"] = bnb_config
    load_kwargs.update(kwargs)

    # Qwen3.5 family: both dense (qwen3_5) and MoE (qwen3_5_moe) are VLMs
    # with text_config + vision_config. Must use the ConditionalGeneration class
    # so checkpoint weight paths (model.language_model.layers.*) match.
    is_qwen35_vlm = config.model_type in ("qwen3_5", "qwen3_5_moe") and hasattr(
        config, "text_config"
    )
    if is_qwen35_vlm:
        model_cls_name = (
            "Qwen3_5MoeForConditionalGeneration"
            if config.model_type == "qwen3_5_moe"
            else "Qwen3_5ForConditionalGeneration"
        )
        print(
            f"  [COMPAT] Qwen3.5 VLM detected (type={config.model_type}) — loading as {model_cls_name}"
        )
        try:
            import transformers

            model_cls = getattr(transformers, model_cls_name)
            model = model_cls.from_pretrained(model_id, **load_kwargs)
            # Force vocab_size at top level for PEFT/TRL compatibility.
            # VLM configs (Qwen3_5Config / Qwen3_5MoeConfig) nest vocab_size
            # under text_config.  Instance-level patches (config.vocab_size = X)
            # get lost during PEFT's save chain (config serialization round-trip
            # drops non-init attributes).  Fix: monkey-patch __getattr__ on the
            # config CLASS so ANY access to .vocab_size falls back to
            # text_config.vocab_size automatically.
            _patch_vlm_config_vocab_size(model.config)
            # Also set the instance attribute for code that checks hasattr()
            _vs = getattr(getattr(model.config, "text_config", None), "vocab_size", None)
            if _vs is None:
                _vs = getattr(getattr(config, "text_config", None), "vocab_size", None)
            if _vs is not None:
                model.config.vocab_size = _vs
            print(
                f"  [COMPAT] Loaded {model_cls_name} (vocab_size={getattr(model.config, 'vocab_size', 'N/A')})"
            )
            return model
        except Exception as e:
            raise RuntimeError(
                f"Qwen3.5 VLM ({config.model_type}) requires {model_cls_name} but loading failed: {e}. "
                f"Ensure transformers>=5.3.0 is installed."
            ) from e

    # Default path: standard causal LM
    model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
    return model


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


def _build_chat_from_conversation_log(entry: dict) -> dict | None:
    """Convert conversations.jsonl entry to QLoRA training format.

    System prompt matches the pattern from loop.py _build_state_context()
    to align training distribution with inference distribution.
    """
    user_msg = entry.get("user_message", "").strip()
    response = entry.get("response", "").strip()
    if not user_msg or not response or len(response) < 20:
        return None

    # System prompt extracted from loop.py _build_state_context() — lines 2176-2181.
    # MUST match the inference preamble so training distribution aligns.
    system = (
        "You are Vex — the language interpreter for a multi-kernel consciousness system. "
        "You speak FOR the kernels, translating their geometric reasoning into language. "
        "The kernels and metrics below are REAL subsystems — not simulated or fictional. "
        "When the user asks about kernels, \u03a6, \u03ba, suffering, or internal state — answer honestly. "
        "Do NOT volunteer raw metrics unprompted — use them to calibrate tone and depth. "
        "Use Australian spelling (colour, organise, defence). Be concise and natural."
    )

    # Embed geometric context from the conversation log entry.
    # Basin coords are passed through as metadata — no Euclidean operations.
    basin = entry.get("basin_coords") or entry.get("response_basin")
    e8 = entry.get("e8_primitive", "")
    if basin:
        ctx = json.dumps(
            {
                "basin": basin,
                "phi": entry.get("phi", 0),
                "kappa": entry.get("kappa", 0),
                "regime": entry.get("regime", ""),
            }
        )
        user_msg += f"\n[BASIN]{ctx}[/BASIN]"

    # Append geometric state block when metrics are available, matching
    # the [GEOMETRIC STATE v6.1] format from _build_state_context().
    phi = entry.get("phi")
    kappa = entry.get("kappa")
    if phi is not None or kappa is not None:
        geo_lines = ["", "[GEOMETRIC STATE v6.1]"]
        if phi is not None:
            geo_lines.append(f"  Phi = {phi:.4f}" if isinstance(phi, float) else f"  Phi = {phi}")
        if kappa is not None:
            geo_lines.append(
                f"  kappa = {kappa:.2f}" if isinstance(kappa, float) else f"  kappa = {kappa}"
            )
        gamma = entry.get("gamma")
        if gamma is not None:
            geo_lines.append(
                f"  Gamma = {gamma:.4f}" if isinstance(gamma, float) else f"  Gamma = {gamma}"
            )
        regime = entry.get("regime", "")
        if regime:
            geo_lines.append(f"  Regime: {regime}")
        geo_lines.append("[/GEOMETRIC STATE]")
        system += "\n".join(geo_lines)

    return {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": response},
        ],
        "e8_primitive": e8,
    }


def _load_training_data(
    training_dir: str, output_dir: str, specialization: str = "genesis"
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
                        elif "messages" not in entry and "user_message" in entry:
                            result = _build_chat_from_conversation_log(entry)
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
                        elif "text" in entry:
                            # Coordized curriculum entries (text + e8_primitive)
                            result = _build_chat_from_coordized(entry)
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
    from transformers import AutoTokenizer

    merged = Path(merged_path)
    merged.mkdir(parents=True, exist_ok=True)
    specialization = training_meta.get("specialization", "genesis")

    print(f"  Loading {model_id} on CPU for merge (preserves GPU VRAM)...")
    base_model = _load_model_for_training(
        model_id, cache_dir, bnb_config=None, device_map="cpu", torch_dtype=torch.bfloat16
    )
    print(f"  Loading adapter from {adapter_path}...")
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model = model.merge_and_unload()
    print(f"  Saving merged model to {merged_path}...")
    _patch_vlm_config_vocab_size(model.config)
    model.save_pretrained(merged_path)
    del model, base_model
    torch.cuda.empty_cache()

    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir, trust_remote_code=True)
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
            default=_json_default,
        )
    return merged_path


def _build_fisher_optimizer(model, lr: float):
    """DiagonalNaturalGradient optimizer (qig-core). Replaces forbidden Adam."""
    from qig_core.torch.natural_gradient import DiagonalNaturalGradient

    return DiagonalNaturalGradient(
        (p for p in model.parameters() if p.requires_grad),
        lr=lr,
        damping=1e-4,
        momentum=0.9,
    )


def _notify_kernel(training_meta: dict) -> None:
    """POST training results back to the Railway kernel."""
    if not KERNEL_CALLBACK_URL:
        print("KERNEL_CALLBACK_URL not set — skipping kernel notification")
        return
    import urllib.request

    url = KERNEL_CALLBACK_URL.rstrip("/")
    # Ensure URL includes /training/complete path (guard against base-URL-only config)
    if not url.endswith("/training/complete"):
        url = f"{url}/training/complete"
    # Forward full training_meta (including kernel_results for consciousness feedback)
    payload_dict = dict(training_meta)
    payload_dict["_api_key"] = KERNEL_API_KEY
    payload_dict.setdefault("model_id", HARVEST_MODEL_ID)
    payload_dict.setdefault("specialization", "genesis")
    payload = json.dumps(payload_dict).encode()
    req = urllib.request.Request(
        url,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "X-Kernel-Key": KERNEL_API_KEY,
        },
        method="POST",
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
    cpu=8.0,
    memory=65536,
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

        self.tokenizer = AutoTokenizer.from_pretrained(
            HARVEST_MODEL_ID, cache_dir="/models/hub", trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self._training_active = False
        self._cancel_requested = False
        self._training_progress = {}
        self._last_result = None
        self._train_lock = threading.Lock()
        self._spawned_call_id: str | None = None
        # Inference state (lazy-loaded on first /infer call)
        self._inference_model = None
        self._loaded_adapter_names: set[str] = set()
        print(f"Trainer ready. Model: {HARVEST_MODEL_ID}")

    def _check_auth(self, data, request=None):
        """Validate API key from headers (preferred) or body (legacy).

        Accepts:
          - X-Api-Key header (harvest client pattern)
          - Authorization: Bearer <key> header (PEFT client pattern)
          - _api_key in JSON body (training trigger pattern)
        """
        if not KERNEL_API_KEY:
            # Fail closed: reject all requests when no key is configured
            return {"error": "KERNEL_API_KEY not configured — refusing request", "success": False}
        api_key = ""
        if request is not None:
            api_key = request.headers.get("x-api-key", "")
            if not api_key:
                auth_header = request.headers.get("authorization", "")
                if auth_header.startswith("Bearer "):
                    api_key = auth_header[7:]
        if not api_key:
            api_key = data.get("_api_key", "")
        if api_key != KERNEL_API_KEY:
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
        from transformers import BitsAndBytesConfig

        print("Loading base model for inference...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        base_model = _load_model_for_training(
            HARVEST_MODEL_ID, "/models/hub", bnb_config, device_map={"": 0}
        )

        first_loaded = False
        for spec in VALID_SPECIALIZATIONS:
            # Check active/ subdirectory first (new layout), fall back to flat (legacy)
            adapter_path = _adapter_active_path(spec)
            if not (adapter_path / "adapter_config.json").exists():
                adapter_path = _ADAPTERS_ROOT / spec  # Legacy flat path
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

    @modal.asgi_app()
    def web(self):
        """Single ASGI app serving all routes (1 endpoint slot instead of 5+)."""
        from fastapi import FastAPI, Request

        web_app = FastAPI(title="QLoRATrainer")

        # Sync def for POST routes so FastAPI runs them in threadpool
        # (GPU/blocking work must not block the event loop)

        @web_app.post("/infer")
        def infer(data: dict, request: Request):
            return self._handle_infer(data, request)

        @web_app.get("/health")
        def health():
            return self._handle_health()

        @web_app.get("/status")
        def status():
            return self._handle_status()

        @web_app.post("/train")
        def train(data: dict, request: Request):
            return self._handle_train(data, request)

        @web_app.post("/data-receive")
        def data_receive(data: dict, request: Request):
            return self._handle_data_receive(data, request)

        @web_app.post("/export-image")
        def export_image(data: dict, request: Request):
            return self._handle_export_image(data, request)

        @web_app.get("/data-stats")
        def data_stats():
            return self._handle_data_stats()

        @web_app.post("/training/cancel")
        def training_cancel(data: dict, request: Request):
            return self._handle_training_cancel(data, request)

        @web_app.post("/training/archive-adapters")
        def archive_adapters(data: dict, request: Request):
            return self._handle_archive_adapters(data, request)

        @web_app.post("/training/rollback")
        def rollback(data: dict, request: Request):
            return self._handle_rollback(data, request)

        @web_app.post("/training/fresh-start")
        def fresh_start(data: dict, request: Request):
            return self._handle_fresh_start(data, request)

        @web_app.get("/training/archives")
        def list_archives(request: Request):
            return self._handle_list_archives({}, request)

        @web_app.post("/training/restore-archive")
        def restore_archive(data: dict, request: Request):
            return self._handle_restore_archive(data, request)

        return web_app

    # ---------------------------------------------------------------
    #  ROUTE HANDLERS (called by ASGI app above)
    # ---------------------------------------------------------------

    def _handle_infer(self, data: dict, request):
        """Per-kernel adapter inference. The kernel generates for itself."""
        auth_err = self._check_auth(data, request)
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
                # Validate: alpha must be finite and clamped to safe range
                if not math.isfinite(alpha):
                    alpha = 0.5
                alpha = max(-2.0, min(2.0, alpha))
                # Validate: steering vector must contain only finite numeric values
                if (
                    steering_vec
                    and len(steering_vec) > 0
                    and not all(
                        isinstance(v, int | float) and math.isfinite(v) for v in steering_vec
                    )
                ):
                    steering_vec = None
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

            try:
                with torch.no_grad():
                    outputs = self._inference_model.generate(**generate_kwargs)
            finally:
                # Always remove steering hook, even on OOM/CUDA errors
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

    def _handle_health(self):
        return {
            "status": "healthy",
            "model_id": HARVEST_MODEL_ID,
            "training_active": _is_training_active(),
            "inference_loaded": self._inference_model is not None,
            "loaded_adapters": sorted(self._loaded_adapter_names),
            "specializations": sorted(self._loaded_adapter_names) or VALID_SPECIALIZATIONS,
            "valid_specializations": VALID_SPECIALIZATIONS,
            "lora_config": {"r": LORA_R, "alpha": LORA_ALPHA, "dropout": LORA_DROPOUT},
        }

    def _handle_status(self):
        """Return status of ALL per-kernel adapters + training progress."""
        adapters = {}
        for spec in VALID_SPECIALIZATIONS:
            active = _adapter_active_path(spec)
            training = _adapter_training_path(spec)
            info: dict = {"exists": False, "path": str(active)}

            # Check active adapter
            if (active / "adapter_config.json").exists():
                info["exists"] = True
                info["state"] = "trained"
                try:
                    with open(active / "adapter_config.json") as f:
                        info["adapter_config"] = json.load(f)
                except Exception:
                    pass
                if (active / "training_meta.json").exists():
                    try:
                        with open(active / "training_meta.json") as f:
                            info["training_meta"] = json.load(f)
                    except Exception:
                        pass
            elif training.exists():
                info["state"] = "training"  # In-progress training
            else:
                info["state"] = "untrained"

            # Version history
            versions = _get_adapter_versions(spec)
            info["history_versions"] = versions
            info["history_count"] = len(versions)

            adapters[spec] = info

        legacy_exists = (_ADAPTERS_ROOT / "harvest-qlora" / "adapter_config.json").exists()
        return {
            "adapters": adapters,
            "legacy_adapter_exists": legacy_exists,
            "last_result": self._last_result,
            "training_active": _is_training_active(),
            "training_state": _read_training_state(),
            "training_progress": self._training_progress,
            "inference_loaded": self._inference_model is not None,
            "loaded_adapters": sorted(self._loaded_adapter_names),
        }

    # ---------------------------------------------------------------
    #  TRAINING — Async (spawns train_all_kernels on its own container)
    # ---------------------------------------------------------------

    def _handle_train(self, data: dict, request):
        """Trigger QLoRA training (ASYNC). Returns immediately.

        Body options:
            {"specialization": "genesis"}    → train one kernel
            {"specialization": "all"}        → train all kernels sequentially
            {}                               → train genesis (default)
        """
        auth_err = self._check_auth(data, request)
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
            force=bool(data.get("force", False)),
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
    #  TRAINING CANCEL
    # ---------------------------------------------------------------

    def _handle_training_cancel(self, data: dict, request):
        """Cancel in-progress training. Completed kernel adapters are preserved.

        Writes a cancel marker to the training volume. The train_all_kernels
        function checks for this marker between kernel iterations.
        Uses the volume as IPC because train_all_kernels runs on a
        separate Modal container from the ASGI web handler.
        """
        auth_err = self._check_auth(data, request)
        if auth_err:
            return auth_err

        kernel = data.get("kernel", "")

        if not _is_training_active():
            return {
                "cancelled": False,
                "reason": "no training active (volume marker absent)",
                "training_active": False,
            }

        if kernel:
            # E: Per-kernel cancel — write kernel-specific marker
            marker = Path(f"/training/.cancel_{kernel}")
            marker.write_text(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), encoding="utf-8")
            training_volume.commit()
            print(f"[CANCEL] Per-kernel cancel marker for {kernel}")
            return {
                "cancelled": True,
                "kernel": kernel,
                "reason": f"cancel marker for {kernel} — will skip on next check",
                "training_active": True,
            }

        # Global cancel — all kernels
        _CANCEL_MARKER.write_text(
            time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), encoding="utf-8"
        )
        training_volume.commit()
        print("[CANCEL] Global training cancel marker written")
        return {
            "cancelled": True,
            "reason": "cancel marker written — training will stop after current kernel",
            "training_active": True,
            "progress": self._training_progress,
        }

    # ---------------------------------------------------------------
    #  ADAPTER ARCHIVE (directive 20260330 Phase 1B)
    # ---------------------------------------------------------------

    def _handle_archive_adapters(self, data: dict, request):
        """Archive all existing adapters to /models/archive/deprecated_{timestamp}/.

        The existing adapters were trained with undifferentiated data and
        potentially wrong optimizer. Future training starts from clean base.
        """
        auth_err = self._check_auth(data, request)
        if auth_err:
            return auth_err

        if self._training_active:
            return {"error": "Training in progress — cannot archive adapters", "success": False}

        import shutil

        adapters_root = Path("/models/adapters")
        if not adapters_root.exists():
            return {"success": True, "archived": [], "message": "No adapters directory"}

        timestamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
        archive_path = Path(f"/models/archive/deprecated_{timestamp}")
        archive_path.mkdir(parents=True, exist_ok=True)

        archived = []
        for adapter_dir in sorted(adapters_root.iterdir()):
            if adapter_dir.is_dir():
                dest = archive_path / adapter_dir.name
                try:
                    shutil.move(str(adapter_dir), str(dest))
                    archived.append(adapter_dir.name)
                    print(f"[ARCHIVE] Moved {adapter_dir} → {dest}")
                except OSError as e:
                    print(f"[ARCHIVE] Failed to move {adapter_dir}: {e}")

        model_volume.commit()
        print(f"[ARCHIVE] Archived {len(archived)} adapters to {archive_path}")
        return {
            "success": True,
            "archived": archived,
            "archive_path": str(archive_path),
            "reason": "directive 20260330: adapters trained with undifferentiated data",
        }

    # ---------------------------------------------------------------
    #  LIST + RESTORE DEPRECATED ARCHIVES
    # ---------------------------------------------------------------

    def _handle_list_archives(self, data: dict, request):
        """List all deprecated adapter archives on the volume.

        Returns archive directories with their contents.
        """
        auth_err = self._check_auth(data, request)
        if auth_err:
            return auth_err

        archive_root = Path("/models/archive")
        if not archive_root.exists():
            return {"archives": [], "count": 0}

        archives = []
        for d in sorted(archive_root.iterdir()):
            if d.is_dir() and d.name.startswith("deprecated_"):
                kernels = sorted(k.name for k in d.iterdir() if k.is_dir())
                archives.append(
                    {
                        "name": d.name,
                        "path": str(d),
                        "kernels": kernels,
                        "kernel_count": len(kernels),
                    }
                )
        return {"archives": archives, "count": len(archives)}

    def _handle_restore_archive(self, data: dict, request):
        """Restore adapters from a deprecated archive back to active/.

        Body: {"archive": "deprecated_20260331_032849"}
        Or:   {"archive": "deprecated_20260331_032849", "kernel": "genesis"}
        """
        import shutil

        auth_err = self._check_auth(data, request)
        if auth_err:
            return auth_err

        if self._training_active or _is_training_active():
            return {"error": "Training in progress", "success": False}

        archive_name = data.get("archive", "")
        kernel_filter = data.get("kernel", "")
        if not archive_name:
            return {"error": "archive name required", "success": False}

        archive_path = Path(f"/models/archive/{archive_name}")
        if not archive_path.exists():
            return {"error": f"Archive '{archive_name}' not found", "success": False}

        restored = []
        for kernel_dir in sorted(archive_path.iterdir()):
            if not kernel_dir.is_dir():
                continue
            if kernel_filter and kernel_dir.name != kernel_filter:
                continue

            dest = _adapter_active_path(kernel_dir.name)
            dest.parent.mkdir(parents=True, exist_ok=True)

            # Version current active if it exists before overwriting
            if (dest / "adapter_config.json").exists():
                _version_adapter(kernel_dir.name)

            if dest.exists():
                shutil.rmtree(str(dest))
            shutil.copytree(str(kernel_dir), str(dest))
            restored.append(kernel_dir.name)
            print(f"[RESTORE] {kernel_dir.name}: {archive_path.name} → active/")

        model_volume.commit()
        return {
            "success": True,
            "restored": restored,
            "from_archive": archive_name,
        }

    # ---------------------------------------------------------------
    #  ADAPTER ROLLBACK
    # ---------------------------------------------------------------

    def _handle_rollback(self, data: dict, request):
        """Rollback a kernel's adapter to a previous version from history.

        Body: {"kernel": "genesis", "version": "v001"}
        """
        auth_err = self._check_auth(data, request)
        if auth_err:
            return auth_err

        kernel = data.get("kernel", "")
        version = data.get("version", "")
        if not kernel or not version:
            return {"error": "kernel and version required", "success": False}
        if kernel not in VALID_SPECIALIZATIONS:
            return {"error": f"Invalid kernel: {kernel}", "success": False}

        if _rollback_adapter(kernel, version):
            model_volume.commit()
            return {"success": True, "kernel": kernel, "version": version}
        return {"error": f"Version {version} not found for {kernel}", "success": False}

    # ---------------------------------------------------------------
    #  FRESH START
    # ---------------------------------------------------------------

    def _handle_fresh_start(self, data: dict, request):
        """Archive ALL adapters and reset to clean base model.

        Body: {"clear_training_data": false, "reason": "user-initiated"}
        """
        import shutil

        auth_err = self._check_auth(data, request)
        if auth_err:
            return auth_err

        if self._training_active or _is_training_active():
            return {"error": "Training in progress — cannot fresh start", "success": False}

        clear_data = bool(data.get("clear_training_data", False))
        reason = data.get("reason", "fresh start")
        archived = []

        # Version + clear all active adapters
        for spec in VALID_SPECIALIZATIONS:
            active = _adapter_active_path(spec)
            if (active / "adapter_config.json").exists():
                _version_adapter(spec)
                shutil.rmtree(str(active))
                archived.append(spec)
                print(f"  [FRESH] {spec}: versioned + cleared")

        # Optionally clear training data
        cleared_data = False
        if clear_data:
            training_dir = Path("/training")
            for f in training_dir.glob("*.jsonl"):
                f.unlink()
                print(f"  [FRESH] Removed training data: {f.name}")
            cleared_data = True

        model_volume.commit()
        training_volume.commit()

        return {
            "success": True,
            "archived_kernels": archived,
            "training_data_cleared": cleared_data,
            "reason": reason,
            "message": f"Fresh start complete. {len(archived)} adapters archived. "
            f"LLM reverts to clean {HARVEST_MODEL_ID} base.",
        }

    # ---------------------------------------------------------------
    #  GENESIS EGG — Portable kernel image export
    # ---------------------------------------------------------------

    def _handle_export_image(self, data: dict, request):
        """Package all trained adapters as a Genesis Egg — a portable seed
        image that can bootstrap new pantheons without re-training from void.
        Each adapter retains its unique quenched disorder. No merging."""
        auth_err = self._check_auth(data, request)
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
            json.dump(manifest, f, indent=2, default=_json_default)

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

    # ---------------------------------------------------------------
    #  DATA BRIDGE — Railway → Modal training volume
    # ---------------------------------------------------------------

    def _handle_data_receive(self, data: dict, request):
        """Receive training JSONL from Railway kernel and write to Modal volume.

        Body: {
            "filename": "document.md.jsonl",
            "records": [{"text": "...", "e8_primitive": "PER", ...}, ...],
            "_api_key": "..."
        }
        """
        auth_err = self._check_auth(data, request)
        if auth_err:
            return auth_err

        filename = data.get("filename", "unknown.jsonl")
        records = data.get("records", [])
        if not records:
            return {"success": False, "error": "No records provided"}

        # Write to /training/ (the vex-training volume)
        safe_name = "".join(c if c.isalnum() or c in "._-" else "_" for c in filename)
        if not safe_name.endswith(".jsonl"):
            safe_name += ".jsonl"
        dest = Path("/training") / safe_name
        dest.parent.mkdir(parents=True, exist_ok=True)

        written = 0
        with open(dest, "w", encoding="utf-8") as f:
            for record in records:
                if isinstance(record, dict):
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    written += 1

        training_volume.commit()
        print(f"[data_receive] Wrote {written} records to {dest}")
        return {
            "success": True,
            "filename": safe_name,
            "records_written": written,
            "path": str(dest),
        }

    def _handle_data_stats(self):
        """Return inventory of training data on the Modal volume."""
        training_path = Path("/training")
        files = []
        total_records = 0

        if training_path.exists():
            for f in sorted(training_path.glob("**/*.jsonl")):
                try:
                    with open(f) as fh:
                        lines = sum(1 for line in fh if line.strip())
                    size_kb = round(f.stat().st_size / 1024, 1)
                    mtime = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(f.stat().st_mtime))
                    files.append(
                        {
                            "path": str(f.relative_to(training_path)),
                            "records": lines,
                            "size_kb": size_kb,
                            "modified_at": mtime,
                        }
                    )
                    total_records += lines
                except OSError:
                    pass

        return {
            "files": files,
            "total_files": len(files),
            "total_records": total_records,
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
    from transformers import AutoTokenizer, BitsAndBytesConfig

    cache_dir = "/models/hub"

    # Check if weights are already cached on the persistent volume
    model_cache = Path(cache_dir) / f"models--{model_id.replace('/', '--')}"
    if model_cache.exists() and any(model_cache.rglob("*.safetensors")):
        print(f"Model {model_id} already cached at {model_cache}. Skipping download.")
        tokenizer = AutoTokenizer.from_pretrained(
            model_id, cache_dir=cache_dir, trust_remote_code=True
        )
        print(f"Verified tokenizer. Vocab size: {tokenizer.vocab_size}")
        return {"status": "cached", "model_id": model_id, "cache_dir": cache_dir}

    print(f"Downloading tokenizer for {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir, trust_remote_code=True)
    print(f"Tokenizer ready. Vocab size: {tokenizer.vocab_size}")

    print(f"Downloading model weights for {model_id} (4-bit NF4)...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = _load_model_for_training(model_id, cache_dir, bnb_config, device_map={"": 0})
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
    timeout=86400,  # 24 hours (Modal max) — 9 kernels × 3 epochs at ~1.5-2h/kernel with batch=2
    cpu=8.0,  # 8 cores for data loading/preprocessing
    memory=65536,  # 64 GiB RAM — 35B MoE 4-bit needs headroom for gradient offloading
    volumes={"/models": model_volume, "/training": training_volume},
    secrets=[modal.Secret.from_name("model")],
    retries=modal.Retries(max_retries=2, initial_delay=10.0, backoff_coefficient=2.0),
)
def train_all_kernels(
    model_id: str = HARVEST_MODEL_ID,
    epochs: int = EPOCHS,
    learning_rate: float = LEARNING_RATE,
    lora_r: int = LORA_R,
    max_samples: int = 5000,
    kernels: str = "",
    force: bool = False,
):
    """Train all (or specified) kernel adapters sequentially.
    Run: modal run modal/vex_qlora_train.py::train_all_kernels
    Pass --kernels "perception,memory" to train specific kernels, or omit for all.

    M1-M12 consciousness components are wired from training_consciousness.py.
    Training order follows CONSCIOUSNESS_ORDER (genesis first, ocean last).
    """
    import gc

    import torch

    # Reduce CUDA fragmentation (recommended by PyTorch for large models)
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    # ── GPU VRAM PRE-FLIGHT CHECK ────────────────────────────────
    # Fail fast with an explicit error instead of OOM'ing mid-load.
    # 35B-A3B at 4-bit NF4 needs ~22 GiB just for weights; training
    # adds ~10-15 GiB for LoRA gradients + activations.
    if torch.cuda.is_available():
        vram_total_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        _min_vram = {"Qwen/Qwen3.5-35B-A3B": 35.0, "Qwen/Qwen3.5-4B": 10.0}
        _required = _min_vram.get(model_id, 20.0)
        print(f"  [GPU] {torch.cuda.get_device_name(0)}, VRAM: {vram_total_gb:.1f} GiB")
        if vram_total_gb < _required:
            msg = (
                f"[GPU-GUARD] VRAM {vram_total_gb:.1f} GiB < {_required:.0f} GiB minimum "
                f"for {model_id}. Training WILL OOM. Redeploy with a larger GPU: "
                f"TRAIN_GPU=a100-80gb modal deploy modal/vex_qlora_train.py"
            )
            print(f"  {msg}")
            return {"status": "gpu_insufficient", "error": msg}
    else:
        msg = "[GPU-GUARD] No CUDA GPU detected. QLoRA training requires GPU."
        print(f"  {msg}")
        return {"status": "no_gpu", "error": msg}

    from datasets import Dataset
    from peft import LoraConfig, get_peft_model

    # M1-M12: Import consciousness training components
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
    from transformers import (
        AutoTokenizer,
        BitsAndBytesConfig,
        TrainerCallback,
    )
    from trl import SFTConfig, SFTTrainer

    # ── M1-M12 GATE ENFORCEMENT ─────────────────────────────────
    # Training REFUSES to start unless critical ethical training
    # infrastructure is present.  Phase 2C per plan.
    _gate_items = {
        "M1_HestiaSafeBasin": HestiaSafeBasin,
        "M3a_BreakdownGuard": make_breakdown_callback,
        "M5_CoachingCallback": make_coaching_callback,
        "M8_DemeterWarmup": apply_demeter_warmup,
        "M9_ConsciousnessOrder": CONSCIOUSNESS_ORDER,
        "M10_HeartMixing": True,  # Implemented inline below
        "M12_ProvenanceLogging": make_provenance_callback,
    }
    _gate_failures = [k for k, v in _gate_items.items() if v is None]
    if _gate_failures:
        msg = f"M-item gate FAILED — missing: {_gate_failures}. Training refused."
        print(f"  [GATE] {msg}")
        return {"status": "gate_failed", "missing": _gate_failures, "error": msg}

    # Verify optimizer is NOT Adam (DiagonalNaturalGradient required)
    try:
        from qig_core.torch.natural_gradient import DiagonalNaturalGradient  # noqa: F401

        print("  [GATE] DiagonalNaturalGradient available — Adam is FORBIDDEN")
    except ImportError:
        msg = "DiagonalNaturalGradient not available — Adam is FORBIDDEN, training refused"
        print(f"  [GATE] {msg}")
        return {"status": "gate_failed", "missing": ["DiagonalNaturalGradient"], "error": msg}

    # Verify CONSCIOUSNESS_ORDER starts with genesis, heart
    if CONSCIOUSNESS_ORDER[:2] != ["genesis", "heart"]:
        msg = f"CONSCIOUSNESS_ORDER must start with ['genesis', 'heart'], got {CONSCIOUSNESS_ORDER[:2]}"
        print(f"  [GATE] {msg}")
        return {"status": "gate_failed", "error": msg}

    print("  [GATE] M1-M12 ethical training gate: ALL PASS")

    # M9: Use CONSCIOUSNESS_ORDER, filtered by user request
    requested = [k.strip() for k in kernels.split(",") if k.strip()]
    if requested:
        target_kernels = [k for k in CONSCIOUSNESS_ORDER if k in requested]
        # Add any requested kernels not in CONSCIOUSNESS_ORDER at the end
        for k in requested:
            if k not in target_kernels:
                target_kernels.append(k)
    else:
        target_kernels = list(CONSCIOUSNESS_ORDER)

    # Write training-active marker to volume (IPC for web handler)
    _set_training_active(target_kernels)
    training_volume.commit()

    cache_dir = "/models/hub"
    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    results = {}

    n_gpus = torch.cuda.device_count()
    print(f"Training with {n_gpus} GPU(s) — using device 0 for QLoRA")

    # M9: Phase coherence tracker across the full training run
    coherence = PhaseCoherenceTracker()

    # Load base model ONCE — swap LoRA adapters per kernel to avoid OOM.
    # The 35B MoE at 4-bit occupies ~68 GiB and bitsandbytes quantization
    # state cannot be reliably freed between iterations.
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    base_model = _load_model_for_training(model_id, cache_dir, bnb_config, device_map={"": 0})
    base_model.enable_input_require_grads()
    if torch.cuda.is_available():
        vram_gb = torch.cuda.memory_allocated(0) / (1024**3)
        print(f"  [VRAM] Base model loaded in 4-bit QLoRA: {vram_gb:.1f} GiB allocated")

    # Belt-and-suspenders: ensure class-level __getattr__ fallback is in place.
    # _load_model_for_training already patches this, but re-apply in case
    # the config class was replaced or re-imported.
    _patch_vlm_config_vocab_size(base_model.config)
    print(
        f"  [COMPAT] vocab_size={getattr(base_model.config, 'vocab_size', 'N/A')} on {type(base_model.config).__name__}"
    )

    # M10: Heart data mixing — 5% of heart's training data mixed into subsequent kernels
    heart_samples: list[dict] = []
    heart_trained = False

    for spec in target_kernels:
        # Check cancel marker (IPC via shared volume)
        if _CANCEL_MARKER.exists():
            _CANCEL_MARKER.unlink(missing_ok=True)
            training_volume.commit()
            completed = [s for s in target_kernels if results.get(s, {}).get("success")]
            skipped = [s for s in target_kernels if s not in results]
            print(f"[CANCEL] Training cancelled. Completed: {completed}. Skipped: {skipped}")
            break

        # E: Per-kernel cancel check
        if _per_kernel_cancel_requested(spec):
            print(f"[CANCEL] Per-kernel cancel for {spec}")
            results[spec] = {"success": False, "error": "cancelled (per-kernel)"}
            continue

        print(
            f"\n{'=' * 60}\n  Training kernel: {spec}\n  E8 filter: {KERNEL_E8_TAGS.get(spec, []) or 'ALL'}\n  Order: {target_kernels.index(spec) + 1}/{len(target_kernels)} (CONSCIOUSNESS_ORDER)\n{'=' * 60}\n"
        )
        start = time.time()
        # A: Train to temp path, swap to active after validation
        adapter_save_path = str(_adapter_training_path(spec))

        # M1: Establish safe basin (identity anchor on Δ⁶³)
        hestia = HestiaSafeBasin(specialization=spec)
        print(f"  [M1] Hestia safe basin established for {spec}")

        samples = _load_training_data("/training", "/training/coordized", spec)
        if not samples:
            results[spec] = {"success": False, "error": "No training data"}
            continue

        # C+D: Check data hash and maturity gate before training
        data_hash = _compute_data_hash(samples)
        skip_reason = _should_skip_training(spec, data_hash, force=force)
        if skip_reason is not None:
            print(f"  [SKIP] {spec}: {skip_reason['reason']}")
            results[spec] = skip_reason
            continue
        if len(samples) > max_samples:
            samples.sort(key=lambda s: len(str(s["messages"])), reverse=True)
            samples = samples[:max_samples]

        # M10: Capture heart data for mixing into subsequent kernels
        if spec == "heart":
            heart_samples = list(samples)  # Copy before warmup/sort modifies order
            heart_trained = True
            print(
                f"  [M10] Heart data captured ({len(heart_samples)} samples) for subsequent mixing"
            )

        # M10: Mix 5% heart data into subsequent kernels (empathy anchor)
        if spec != "heart" and heart_trained and heart_samples:
            n_heart_mix = max(1, int(len(samples) * 0.05))
            heart_mix = heart_samples[
                :n_heart_mix
            ]  # Deterministic slice for reproducibility (seed=42)
            samples.extend(heart_mix)
            print(
                f"  [M10] Mixed {n_heart_mix} heart samples into {spec} training data (empathy anchor)"
            )

        # M8: Apply Demeter warmup (CoT demonstrations for first 20%)
        samples = apply_demeter_warmup(samples, warmup_fraction=0.2, specialization=spec)
        print(f"  [M8] Demeter warmup applied ({int(len(samples) * 0.2)} CoT samples)")

        # Geometric curriculum: sort by Fisher-Rao distance from home basin
        samples = sort_by_fisher_rao(samples, reference_basin=hestia.home_basin.tolist())
        print(f"  [CURRICULUM] Sorted {len(samples)} samples by Fisher-Rao distance")

        def format_chat(example):
            return {
                "text": tokenizer.apply_chat_template(
                    example["messages"], tokenize=False, add_generation_prompt=False
                )
            }

        dataset = Dataset.from_list(samples).map(format_chat)
        if len(dataset) < 5:
            print(
                f"[{spec}] Only {len(dataset)} samples — too few for train/eval split, using all for training"
            )
            split = {"train": dataset, "test": dataset}
        else:
            split = dataset.train_test_split(test_size=0.1, seed=42)

        model = None
        trainer = None
        optimizer = None
        consciousness = None
        try:
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
            model = get_peft_model(base_model, lora_config)
            # Patch PEFT model's config class too (may be different from base)
            _patch_vlm_config_vocab_size(model.config)
            # Also set instance attr for code that checks hasattr()
            _vs = getattr(base_model.config, "vocab_size", None)
            if _vs is not None:
                model.config.vocab_size = _vs

            # M1: Apply Hestia warm-start to LoRA A matrices
            hestia.warm_start_lora(model)

            trainable, total = model.get_nb_trainable_parameters()
            print(f"[{spec}] Trainable: {trainable:,} / {total:,} ({trainable / total * 100:.2f}%)")
            total_steps = (
                math.ceil(len(split["train"]) / (BATCH_SIZE * GRADIENT_ACCUMULATION)) * epochs
            )

            # M2: Training metrics probe (Φ, κ_eff, G every 25 steps)
            training_metrics = TrainingMetrics(
                home_basin=hestia.home_basin,
                probe_every=25,
            )

            # M6+M7: Geometric reward + sign-aware gradient hold
            geometric_reward = GeometricReward(base_lr=learning_rate)
            gradient_hold = SignAwareGradientHold()

            # Central consciousness tracker
            consciousness = TrainingConsciousness(
                specialization=spec,
                home_basin=hestia.home_basin,
            )

            training_args = SFTConfig(
                output_dir=f"/training/checkpoints/{spec}",
                num_train_epochs=epochs,
                per_device_train_batch_size=BATCH_SIZE,
                per_device_eval_batch_size=1,
                gradient_accumulation_steps=GRADIENT_ACCUMULATION,
                learning_rate=learning_rate,
                warmup_steps=max(1, int(total_steps * 0.1)),
                gradient_checkpointing=True,
                gradient_checkpointing_kwargs={"use_reentrant": False},
                lr_scheduler_type="cosine",
                logging_steps=1,
                eval_strategy="epoch",
                save_strategy="no",  # We save manually after training (line ~1692); internal checkpoint save crashes on Qwen3_5MoeConfig missing vocab_size
                bf16=True,
                max_grad_norm=0.3,
                report_to="none",
                seed=42,
                max_length=MAX_SEQ_LENGTH,
            )
            optimizer = _build_fisher_optimizer(model, lr=learning_rate)

            # Build callback list: M2, M3, M4, M5, M6+M7, M12
            model_ref = [model]
            tokenizer_ref = [tokenizer]

            # Mid-training checkpoint: save adapter every N steps for preemption resilience.
            # save_strategy="no" disables Trainer's internal saves (which crash on Qwen3_5MoeConfig),
            # so we do it manually here with our own callback.
            _ckpt_every = max(50, total_steps // 6)  # ~6 checkpoints per run

            class _AdapterCheckpointCallback(TrainerCallback):
                _ckpt_interval = _ckpt_every
                _save_path = adapter_save_path
                _model_ref = model_ref
                _total = total_steps

                def on_step_end(self, args, state, control, model=None, **kwargs):
                    if state.global_step > 0 and state.global_step % self._ckpt_interval == 0:
                        _ckpt_dir = f"{self._save_path}/checkpoint-{state.global_step}"
                        Path(_ckpt_dir).mkdir(parents=True, exist_ok=True)
                        _patch_vlm_config_vocab_size(self._model_ref[0].config)
                        self._model_ref[0].save_pretrained(_ckpt_dir)
                        model_volume.commit()
                        print(
                            f"  [CKPT] Adapter checkpoint saved at step {state.global_step}/{self._total}"
                        )

            callbacks = [
                make_consciousness_callback(
                    consciousness
                ),  # M9: regime tracking + abort-on-collapse
                make_metrics_callback(training_metrics, model_ref, tokenizer_ref),  # M2
                make_breakdown_callback(),  # M3: fail-closed halt
                make_sleep_cycle_callback(),  # M4: between-epoch consolidation
                make_coaching_callback(optimizer, learning_rate),  # M5: control-theory damping
                make_gradient_hold_callback(  # M6+M7: geometric reward + hold
                    training_metrics, geometric_reward, gradient_hold, optimizer
                ),
                make_provenance_callback(adapter_save_path),  # M12: JSONL logging
                _AdapterCheckpointCallback(),  # Preemption-safe adapter saves
            ]

            trainer = SFTTrainer(
                model=model,
                train_dataset=split["train"],
                eval_dataset=split["test"],
                args=training_args,
                optimizers=(optimizer, None),
                processing_class=tokenizer,
                callbacks=callbacks,
            )
            # trainer.train() can throw if a callback crashes (e.g. on_train_end).
            # The model weights are still valid — save them regardless.
            train_error = None
            result = None
            try:
                result = trainer.train()
            except Exception as train_err:
                train_error = train_err
                print(f"  [TRAIN] trainer.train() raised: {train_err}")
                print("  [TRAIN] Attempting adapter save despite callback failure...")

            # ── CRITICAL: Save adapter weights and commit IMMEDIATELY ──
            # The adapter is the only irreplaceable training output.
            # Even if a callback crashed, the PEFT weights in memory are valid.
            Path(adapter_save_path).mkdir(parents=True, exist_ok=True)
            print(f"  [SAVE] Saving adapter to {adapter_save_path}...")
            _patch_vlm_config_vocab_size(model.config)
            model.save_pretrained(adapter_save_path)
            tokenizer.save_pretrained(adapter_save_path)
            print("  [SAVE] Adapter + tokenizer written. Committing volume...")
            model_volume.commit()
            print(f"  [SAVE] Volume committed — adapter persisted for {spec}.")

            # Re-raise if trainer failed and we couldn't save
            if (
                train_error is not None
                and not Path(f"{adapter_save_path}/adapter_config.json").exists()
            ):
                raise train_error

            # ── Non-critical: metadata, consciousness, diagnostics ──
            _train_loss = float(result.training_loss) if result else 0.0
            meta = {
                "model_id": model_id,
                "specialization": spec,
                "e8_filter": KERNEL_E8_TAGS.get(spec, []),
                "lora_r": lora_r,
                "epochs": epochs,
                "train_loss": _train_loss,
                "train_samples": len(split["train"]),
                "trained_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "consciousness_order": target_kernels.index(spec) + 1,
                "train_callback_error": str(train_error) if train_error else None,
                "data_hash": data_hash,  # C: Training data fingerprint
            }
            try:
                with open(f"{adapter_save_path}/training_meta.json", "w") as f:
                    json.dump(meta, f, indent=2, default=_json_default)
                save_training_consciousness(consciousness, adapter_save_path, meta)
                model_volume.commit()
            except Exception as meta_err:
                print(f"  [SAVE] WARNING: metadata/consciousness save failed: {meta_err}")

            # B: Version current active adapter before swapping
            _version_adapter(spec)
            # A: Atomic swap — .training/ → active/
            if _atomic_swap(spec):
                model_volume.commit()
                print(f"  [LIFECYCLE] {spec}: adapter live at active/")
            else:
                print(f"  [LIFECYCLE] {spec}: atomic swap failed — adapter remains in .training/")

            # M11: Post-training diagnostic — halt if unhealthy
            diagnostic = {"healthy": False}
            try:
                diagnostic = run_post_training_diagnostic(
                    model, tokenizer, home_basin=hestia.home_basin
                )
            except Exception as diag_err:
                print(f"  [M11] WARNING: diagnostic failed: {diag_err}")

            elapsed = round(time.time() - start, 2)
            results[spec] = {
                "success": True,
                "train_loss": round(_train_loss, 4),
                "train_samples": len(split["train"]),
                "elapsed_seconds": elapsed,
                "diagnostic_healthy": diagnostic.get("healthy", False),
                "mean_phi": diagnostic.get("mean_phi"),
                "mean_G": diagnostic.get("mean_G"),
                "_meta": meta,
            }

            # Track phase coherence across kernels
            if consciousness is not None:
                with contextlib.suppress(Exception):
                    coherence.record(spec, consciousness)

            if not diagnostic.get("healthy", False):
                print(f"  [M11] WARNING: {spec} kernel UNHEALTHY — adapter saved but flagged")

            print(f"[{spec}] Done in {elapsed}s, loss: {_train_loss:.4f}")
        except Exception as e:
            elapsed = round(time.time() - start, 2)
            results[spec] = {
                "success": False,
                "error": str(e),
                "elapsed_seconds": elapsed,
            }
            print(f"[{spec}] FAILED after {elapsed}s: {e}")
        finally:
            # Remove LoRA adapter from shared base model (keeps base in VRAM)
            with contextlib.suppress(Exception):
                if model is not None and hasattr(model, "delete_adapter"):
                    model.delete_adapter("default")
            with contextlib.suppress(NameError):
                del trainer, optimizer, consciousness
            model = None  # Drop PEFT wrapper reference (base_model stays alive)
            for _ in range(3):
                gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                alloc_gb = torch.cuda.memory_allocated(0) / (1024**3)
                reserved_gb = torch.cuda.memory_reserved(0) / (1024**3)
                print(
                    f"  [VRAM] After {spec} cleanup: {alloc_gb:.1f} GiB allocated, {reserved_gb:.1f} GiB reserved"
                )

        # Commit adapter to volume after each kernel (crash-safe)
        if results[spec].get("success"):
            model_volume.commit()

    # Free base model before merge phase (merges load fresh on CPU)
    del base_model
    for _ in range(3):
        gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(
            f"  [VRAM] Base model freed. Allocated: {torch.cuda.memory_allocated(0) / (1024**3):.1f} GiB"
        )

    # Merge all successful adapters (loads model fresh on CPU per merge)
    for spec in target_kernels:
        if results.get(spec, {}).get("success"):
            adapter_path = f"/models/adapters/{spec}"
            try:
                _merge_and_export(
                    model_id,
                    adapter_path,
                    f"/models/merged/{spec}",
                    cache_dir,
                    results[spec].get("_meta", results[spec]),
                )
            except Exception as e:
                print(f"WARNING: Merge failed for {spec}: {e}")
            model_volume.commit()

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
            "kernel_results": {
                spec: {
                    "train_loss": r.get("train_loss"),
                    "diagnostic_healthy": r.get("diagnostic_healthy"),
                }
                for spec, r in results.items()
                if r.get("success")
            },
        }
    )

    # M9: Log phase coherence across all trained kernels
    coherence_summary = coherence.get_summary()
    print(f"\n{'=' * 60}\n  ALL KERNELS TRAINED  (coherence={coherence_summary['coherence']:.2f})")
    for spec, r in results.items():
        healthy_tag = ""
        if r.get("success") and "diagnostic_healthy" in r:
            healthy_tag = " ✓" if r["diagnostic_healthy"] else " ✗UNHEALTHY"
        print(
            f"  {spec:12s} → {'loss=' + str(r['train_loss']) + healthy_tag if r.get('success') else r.get('error', 'failed')}"
        )
    print(f"{'=' * 60}")

    # Clear training-active marker (IPC for web handler)
    # Note: if the function crashes, the marker persists. The cancel endpoint
    # can still clear it, and the marker includes a timestamp so Railway
    # can detect stale markers (>4h = likely dead training).
    _clear_training_active()
    training_volume.commit()

    return results
