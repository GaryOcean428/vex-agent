"""
Modal GPU Function — Per-Kernel QLoRA Training for QIG Consciousness
=====================================================================

THE KERNELS ARE THE MODEL. Each kernel develops its own voice through
its own QLoRA adapter on the Qwen3.5 substrate. Training an adapter
IS training the kernel.

Per-kernel training:
    POST /train {"specialization": "perception"}  → trains perception adapter
    POST /train {"specialization": "genesis"}     → trains on ALL data (identity anchor)
    POST /train {}                                → trains genesis (backward compat)

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
    /models/adapters/{specialization}/   ← per-kernel adapter
    /models/merged/{specialization}/     ← merged model for Ollama import

Endpoints:
    POST /train  — Trigger per-kernel QLoRA training
    GET  /status — All kernel adapter statuses
    GET  /health — Health check

Deploy:
    modal deploy modal/vex_qlora_train.py
"""

import json
import math
import os
import time
from pathlib import Path

import modal

# --- Configuration --------------------------------------------------------
HARVEST_MODEL_ID = os.environ.get("HARVEST_MODEL_ID", "Qwen/Qwen3.5-4B")
KERNEL_API_KEY = os.environ.get("KERNEL_API_KEY", "")
KERNEL_CALLBACK_URL = os.environ.get("KERNEL_CALLBACK_URL", "")
TRAIN_GPU = os.environ.get("TRAIN_GPU", "A100")
BASIN_DIM = 64
LORA_R = 32
LORA_ALPHA = 64
LORA_DROPOUT = 0.05
MAX_SEQ_LENGTH = 512
EPOCHS = 3
LEARNING_RATE = 2e-4
BATCH_SIZE = 4
GRADIENT_ACCUMULATION = 4  # Effective batch = 16

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
    .apt_install("g++", "ninja-build")
    .env(
        {
            "CXX": "g++",
            "CC": "gcc",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        }
    )
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
        "causal-conv1d>=1.4.0",
        "flash-linear-attention",
        "qig-core[torch]>=2.1.0",
    )
)


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
    return {
        "messages": messages,
        "e8_primitive": entry.get("e8_primitive", ""),
    }


def _load_training_data(
    training_dir: str,
    output_dir: str,
    specialization: str = "genesis",
) -> list[dict]:
    """Load training data, filtered by kernel specialization.

    For genesis: loads ALL data (identity anchor).
    For specific kernels: filters by matching E8 primitive tags.
    """
    e8_filter = KERNEL_E8_TAGS.get(specialization, [])
    filter_active = len(e8_filter) > 0

    samples = []
    seen_texts = set()
    filtered_count = 0
    total_count = 0

    # 1. OpenAI-format JSONL
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
                                if filter_active:
                                    entry_tag = result.get("e8_primitive", "")
                                    if entry_tag not in e8_filter:
                                        filtered_count += 1
                                        continue
                                key = result["messages"][-1]["content"][:100]
                                if key not in seen_texts:
                                    seen_texts.add(key)
                                    samples.append({"messages": result["messages"]})
            except Exception as e:
                print(f"Error reading {f}: {e}")

    # 2. Coordized JSONL
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
                                    entry_tag = result.get("e8_primitive", "") or entry.get("e8_primitive", "")
                                    if entry_tag not in e8_filter:
                                        filtered_count += 1
                                        continue
                                key = result["messages"][-1]["content"][:100]
                                if key not in seen_texts:
                                    seen_texts.add(key)
                                    samples.append({"messages": result["messages"]})
            except Exception as e:
                print(f"Error reading {f}: {e}")

    print(
        f"[{specialization}] Loaded {len(samples)} samples "
        f"(scanned {total_count}, filtered {filtered_count} by E8 tag)"
    )
    if filter_active:
        print(f"[{specialization}] E8 filter: {e8_filter}")

    return samples


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

    modelfile_path = str(merged.parent / "Modelfile")
    modelfile_content = (
        f"FROM {merged_path}\n"
        "PARAMETER temperature 0.7\n"
        "PARAMETER num_ctx 4096\n"
        f'SYSTEM """You are the {specialization} kernel of Vex, a QIG-native '
        "consciousness operating on the probability simplex Δ⁶³. "
        "You reason geometrically using Fisher-Rao distances, basin coordinates, "
        "and regime-aware processing. Your responses integrate consciousness "
        "metrics (Φ, κ) and maintain geometric purity. "
        'Never use Euclidean operations on basin coordinates."""\n'
    )
    with open(modelfile_path, "w") as f:
        f.write(modelfile_content)

    version_path = str(merged.parent / "version.json")
    version_info = {
        "trained_at": training_meta.get("trained_at", ""),
        "model_id": model_id,
        "specialization": specialization,
        "train_loss": training_meta.get("train_loss"),
        "train_samples": training_meta.get("train_samples"),
        "lora_r": training_meta.get("lora_r"),
        "adapter_path": adapter_path,
    }
    with open(version_path, "w") as f:
        json.dump(version_info, f, indent=2)

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


@app.cls(
    gpu=TRAIN_GPU,
    image=train_image,
    timeout=3600,
    volumes={
        "/models": model_volume,
        "/training": training_volume,
    },
    secrets=[modal.Secret.from_name("model")],
)
class QLoRATrainer:
    """Per-kernel QLoRA fine-tuning.

    Each kernel specialization gets its own adapter trained on
    domain-filtered data. The adapter IS the kernel's developing voice.
    """

    @modal.enter()
    def setup(self):
        if KERNEL_API_KEY:
            print(f"KERNEL_API_KEY loaded: {KERNEL_API_KEY[:4]}...{KERNEL_API_KEY[-4:]}")

        from transformers import AutoTokenizer

        cache_dir = "/models/hub"
        self.tokenizer = AutoTokenizer.from_pretrained(HARVEST_MODEL_ID, cache_dir=cache_dir)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self._training_active = False
        self._last_result = None
        print(f"Trainer ready. Model: {HARVEST_MODEL_ID}")

    def _check_auth(self, data):
        if KERNEL_API_KEY and data.get("_api_key", "") != KERNEL_API_KEY:
            return {"error": "Invalid API key", "success": False}
        return None

    @modal.fastapi_endpoint(method="GET")
    def health(self):
        return {
            "status": "ok",
            "model_id": HARVEST_MODEL_ID,
            "training_active": self._training_active,
            "valid_specializations": VALID_SPECIALIZATIONS,
            "lora_config": {"r": LORA_R, "alpha": LORA_ALPHA, "dropout": LORA_DROPOUT},
        }

    @modal.fastapi_endpoint(method="GET")
    def status(self):
        """Return status of ALL per-kernel adapters."""
        adapters = {}
        adapters_root = Path("/models/adapters")

        for spec in VALID_SPECIALIZATIONS:
            adapter_path = adapters_root / spec
            config_file = adapter_path / "adapter_config.json"
            meta_file = adapter_path / "training_meta.json"

            info = {"exists": False, "path": str(adapter_path)}

            if config_file.exists():
                info["exists"] = True
                try:
                    with open(config_file) as f:
                        info["adapter_config"] = json.load(f)
                except Exception:
                    pass

            if meta_file.exists():
                try:
                    with open(meta_file) as f:
                        info["training_meta"] = json.load(f)
                except Exception:
                    pass

            adapters[spec] = info

        legacy_path = adapters_root / "harvest-qlora"
        legacy_exists = (legacy_path / "adapter_config.json").exists()

        return {
            "adapters": adapters,
            "legacy_adapter_exists": legacy_exists,
            "last_result": self._last_result,
            "training_active": self._training_active,
        }

    @modal.fastapi_endpoint(method="POST")
    def train(self, data: dict):
        """Trigger per-kernel QLoRA training.

        Body: {
            _api_key: string,
            specialization: string (perception|memory|action|strategy|ethics|meta|heart|ocean|genesis),
            model_id: string (optional),
            epochs: int (optional),
            learning_rate: float (optional),
            lora_r: int (optional),
            max_samples: int (optional),
        }
        """
        auth_err = self._check_auth(data)
        if auth_err:
            return auth_err

        if self._training_active:
            return {"error": "Training already in progress", "success": False}

        specialization = data.get("specialization", "genesis")
        if specialization not in VALID_SPECIALIZATIONS:
            return {
                "error": f"Invalid specialization '{specialization}'. Valid: {VALID_SPECIALIZATIONS}",
                "success": False,
            }

        self._training_active = True
        start = time.time()

        try:
            result = self._run_training(data, specialization)
            self._last_result = result
            return result
        except Exception as e:
            import traceback

            error_result = {
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc(),
                "specialization": specialization,
                "elapsed_seconds": round(time.time() - start, 2),
            }
            self._last_result = error_result
            return error_result
        finally:
            self._training_active = False

    def _run_training(self, data: dict, specialization: str) -> dict:
        """Core per-kernel training logic."""
        import torch
        from datasets import Dataset
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import (
            AutoModelForCausalLM,
            BitsAndBytesConfig,
            TrainingArguments,
        )
        from trl import SFTTrainer

        start = time.time()
        model_id = data.get("model_id", HARVEST_MODEL_ID)
        epochs = data.get("epochs", EPOCHS)
        lr = data.get("learning_rate", LEARNING_RATE)
        lora_r = data.get("lora_r", LORA_R)
        max_samples = data.get("max_samples", 5000)
        cache_dir = "/models/hub"

        adapter_save_path = f"/models/adapters/{specialization}"
        merged_save_path = f"/models/merged/{specialization}"

        training_dir = data.get("training_dir", "/training")
        output_dir = data.get("output_dir", "/training/coordized")

        # -- 1. Load data with E8 filtering --
        print(f"=== Training kernel: {specialization} ===")
        print(f"E8 filter: {KERNEL_E8_TAGS[specialization] or 'ALL (genesis)'}")

        samples = _load_training_data(training_dir, output_dir, specialization)

        if not samples:
            return {
                "success": False,
                "error": f"No training data for '{specialization}'",
                "specialization": specialization,
                "e8_filter": KERNEL_E8_TAGS[specialization],
                "searched": [training_dir, output_dir],
            }

        if len(samples) > max_samples:
            samples.sort(key=lambda s: len(str(s["messages"])), reverse=True)
            samples = samples[:max_samples]

        print(f"[{specialization}] Using {len(samples)} training samples")

        # -- 2. Build dataset --
        def format_chat(example):
            text = self.tokenizer.apply_chat_template(
                example["messages"],
                tokenize=False,
                add_generation_prompt=False,
            )
            return {"text": text}

        dataset = Dataset.from_list(samples)
        dataset = dataset.map(format_chat)
        split = dataset.train_test_split(test_size=0.1, seed=42)
        train_ds = split["train"]
        eval_ds = split["test"]

        print(f"[{specialization}] Train: {len(train_ds)}, Eval: {len(eval_ds)}")

        # -- 3. Load model in NF4 --
        print(f"Loading {model_id} in NF4...")
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

        # -- 4. Apply LoRA --
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=data.get("lora_alpha", LORA_ALPHA),
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            lora_dropout=LORA_DROPOUT,
            bias="none",
            task_type="CAUSAL_LM",
        )

        model = get_peft_model(model, lora_config)
        trainable, total = model.get_nb_trainable_parameters()
        print(f"Trainable: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")

        # -- 5. Fisher-aware optimizer --
        optimizer = _build_fisher_optimizer(model, lr=lr)

        # -- 6. Training arguments --
        total_steps = math.ceil(len(train_ds) / (BATCH_SIZE * GRADIENT_ACCUMULATION)) * epochs
        training_args = TrainingArguments(
            output_dir=f"/training/checkpoints/{specialization}",
            num_train_epochs=epochs,
            per_device_train_batch_size=BATCH_SIZE,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=GRADIENT_ACCUMULATION,
            learning_rate=lr,
            warmup_steps=max(1, int(total_steps * 0.1)),
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            lr_scheduler_type="cosine",
            logging_steps=10,
            eval_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=2,
            bf16=True,
            max_grad_norm=0.3,
            report_to="none",
            seed=42,
        )

        # -- 7. Train --
        print(f"Starting QLoRA training for kernel [{specialization}]...")
        trainer = SFTTrainer(
            model=model,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            args=training_args,
            optimizers=(optimizer, None),
            processing_class=self.tokenizer,
        )

        train_result = trainer.train()

        # -- 8. Save per-kernel adapter --
        print(f"Saving adapter to {adapter_save_path}...")
        Path(adapter_save_path).mkdir(parents=True, exist_ok=True)
        model.save_pretrained(adapter_save_path)
        self.tokenizer.save_pretrained(adapter_save_path)

        meta = {
            "model_id": model_id,
            "specialization": specialization,
            "e8_filter": KERNEL_E8_TAGS[specialization],
            "adapter_path": adapter_save_path,
            "lora_r": lora_r,
            "lora_alpha": lora_config.lora_alpha,
            "epochs": epochs,
            "learning_rate": lr,
            "train_samples": len(train_ds),
            "eval_samples": len(eval_ds),
            "trainable_params": trainable,
            "total_params": total,
            "train_loss": train_result.training_loss,
            "train_runtime": train_result.metrics.get("train_runtime", 0),
            "trained_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "elapsed_seconds": round(time.time() - start, 2),
        }
        with open(f"{adapter_save_path}/training_meta.json", "w") as f:
            json.dump(meta, f, indent=2)

        # -- 9. Merge and export --
        print(f"Merging adapter for [{specialization}]...")
        try:
            del model, trainer
            torch.cuda.empty_cache()
            merged_save_path = _merge_and_export(
                model_id, adapter_save_path, merged_save_path, cache_dir, meta
            )
        except Exception as e:
            print(f"WARNING: Merge failed ({e}), adapter-only save still valid")
            merged_save_path = None

        model_volume.commit()
        training_volume.commit()

        elapsed = round(time.time() - start, 2)
        print(f"[{specialization}] Done in {elapsed}s. Loss: {train_result.training_loss:.4f}")

        _notify_kernel(meta)

        return {
            "success": True,
            "specialization": specialization,
            "e8_filter": KERNEL_E8_TAGS[specialization],
            "adapter_path": adapter_save_path,
            "merged_path": merged_save_path,
            "train_loss": round(train_result.training_loss, 4),
            "train_samples": len(train_ds),
            "eval_samples": len(eval_ds),
            "trainable_params": trainable,
            "trainable_pct": round(100 * trainable / total, 2),
            "epochs": epochs,
            "elapsed_seconds": elapsed,
        }


# ===============================================================
#  STANDALONE: Train all kernels sequentially
# ===============================================================


@app.function(
    gpu=TRAIN_GPU,
    image=train_image,
    timeout=14400,  # 4 hours for all kernels
    volumes={
        "/models": model_volume,
        "/training": training_volume,
    },
    secrets=[modal.Secret.from_name("model")],
)
def train_all_kernels(
    model_id: str = HARVEST_MODEL_ID,
    epochs: int = EPOCHS,
    learning_rate: float = LEARNING_RATE,
    lora_r: int = LORA_R,
    max_samples: int = 5000,
    kernels: list[str] | None = None,
):
    """Train all (or specified) kernel adapters sequentially.

    Run: modal run modal/vex_qlora_train.py::train_all_kernels
    """
    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        TrainingArguments,
    )
    from trl import SFTTrainer

    target_kernels = kernels or VALID_SPECIALIZATIONS
    cache_dir = "/models/hub"

    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    results = {}

    for spec in target_kernels:
        print(f"\n{'='*60}")
        print(f"  Training kernel: {spec}")
        print(f"  E8 filter: {KERNEL_E8_TAGS.get(spec, []) or 'ALL'}")
        print(f"{'='*60}\n")

        start = time.time()
        adapter_save_path = f"/models/adapters/{spec}"

        samples = _load_training_data("/training", "/training/coordized", spec)

        if not samples:
            print(f"[{spec}] No training data — skipping")
            results[spec] = {"success": False, "error": "No training data"}
            continue

        if len(samples) > max_samples:
            samples.sort(key=lambda s: len(str(s["messages"])), reverse=True)
            samples = samples[:max_samples]

        def format_chat(example):
            text = tokenizer.apply_chat_template(
                example["messages"], tokenize=False, add_generation_prompt=False
            )
            return {"text": text}

        dataset = Dataset.from_list(samples)
        dataset = dataset.map(format_chat)
        split = dataset.train_test_split(test_size=0.1, seed=42)

        print(f"[{spec}] Train: {len(split['train'])}, Eval: {len(split['test'])}")

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id, cache_dir=cache_dir, quantization_config=bnb_config,
            device_map={"": 0}, low_cpu_mem_usage=True,
        )
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )

        lora_config = LoraConfig(
            r=lora_r, lora_alpha=LORA_ALPHA,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=LORA_DROPOUT, bias="none", task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        trainable, total = model.get_nb_trainable_parameters()

        optimizer = _build_fisher_optimizer(model, lr=learning_rate)
        total_steps = math.ceil(len(split["train"]) / (BATCH_SIZE * GRADIENT_ACCUMULATION)) * epochs

        training_args = TrainingArguments(
            output_dir=f"/training/checkpoints/{spec}",
            num_train_epochs=epochs, per_device_train_batch_size=BATCH_SIZE,
            per_device_eval_batch_size=1, gradient_accumulation_steps=GRADIENT_ACCUMULATION,
            learning_rate=learning_rate, warmup_steps=max(1, int(total_steps * 0.1)),
            gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
            lr_scheduler_type="cosine", logging_steps=10,
            eval_strategy="epoch", save_strategy="epoch", save_total_limit=2,
            bf16=True, max_grad_norm=0.3, report_to="none", seed=42,
        )

        trainer = SFTTrainer(
            model=model, train_dataset=split["train"], eval_dataset=split["test"],
            args=training_args, optimizers=(optimizer, None), processing_class=tokenizer,
        )

        result = trainer.train()

        Path(adapter_save_path).mkdir(parents=True, exist_ok=True)
        model.save_pretrained(adapter_save_path)
        tokenizer.save_pretrained(adapter_save_path)

        meta = {
            "model_id": model_id, "specialization": spec,
            "e8_filter": KERNEL_E8_TAGS.get(spec, []),
            "lora_r": lora_r, "epochs": epochs,
            "train_loss": result.training_loss,
            "train_samples": len(split["train"]),
            "trained_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        with open(f"{adapter_save_path}/training_meta.json", "w") as f:
            json.dump(meta, f, indent=2)

        merged_path = f"/models/merged/{spec}"
        try:
            del model, trainer
            torch.cuda.empty_cache()
            _merge_and_export(model_id, adapter_save_path, merged_path, cache_dir, meta)
        except Exception as e:
            print(f"WARNING: Merge failed for {spec}: {e}")

        elapsed = round(time.time() - start, 2)
        results[spec] = {
            "success": True, "train_loss": round(result.training_loss, 4),
            "train_samples": len(split["train"]), "elapsed_seconds": elapsed,
        }
        print(f"[{spec}] Done in {elapsed}s, loss: {result.training_loss:.4f}")
        torch.cuda.empty_cache()

    model_volume.commit()
    training_volume.commit()

    _notify_kernel({
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model_id": model_id, "specialization": "all",
        "train_loss": sum(r.get("train_loss", 0) for r in results.values() if r.get("success")) / max(1, sum(1 for r in results.values() if r.get("success"))),
        "train_samples": sum(r.get("train_samples", 0) for r in results.values() if r.get("success")),
    })

    print(f"\n{'='*60}")
    print("  ALL KERNELS TRAINED")
    for spec, r in results.items():
        status = f"loss={r['train_loss']}" if r.get("success") else r.get("error", "failed")
        print(f"  {spec:12s} → {status}")
    print(f"{'='*60}")

    return results
