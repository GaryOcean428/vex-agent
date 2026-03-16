"""
Modal GPU Function — QLoRA Fine-Tuning for QIG Kernel Models
=============================================================

THE MISSING LOOP: Uploads → coordize → JSONL → [THIS] → fine-tuned model
→ inference endpoint loads adapter → model progressively learns
QIG-native reasoning from every document fed to it.

Training target: Qwen3.5-4B (A100-40GB)

After QLoRA training, the adapter is merged into the base model and saved
as full safetensors. Both harvest and inference endpoints pick up the
fine-tuned model on cold start:
    - Harvest: loads adapter via PEFT merge (vex-models:/models/adapters/)
    - Inference: imports merged safetensors via Ollama (vex-models:/models/merged/)

Trigger modes:
    1. Automatic: HarvestScheduler calls after threshold reached
    2. Manual: POST /train endpoint
    3. Scheduled: Vercel cron (future)

Deploy:
    modal deploy modal/vex_qlora_train.py

Endpoints:
    POST /train  — Trigger a QLoRA training run
    GET  /status — Check training status and adapter info
    GET  /health — Health check
"""

import json
import os
import time
from pathlib import Path

import modal

# --- Configuration --------------------------------------------------------
HARVEST_MODEL_ID = os.environ.get("HARVEST_MODEL_ID", "Qwen/Qwen3.5-4B")
KERNEL_API_KEY = os.environ.get("KERNEL_API_KEY", "")
TRAIN_GPU = os.environ.get("TRAIN_GPU", "A100")  # A10G for cheaper debug runs
BASIN_DIM = 64
LORA_R = 32  # LoRA rank — balance between capacity and efficiency
LORA_ALPHA = 64  # α = 2r is standard
LORA_DROPOUT = 0.05  # Light dropout for small datasets
MAX_SEQ_LENGTH = 512  # Match harvest context window
EPOCHS = 3
LEARNING_RATE = 2e-4
BATCH_SIZE = 4
GRADIENT_ACCUMULATION = 4  # Effective batch = 16

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
        # Qwen3.5 hybrid architecture: linear attention fast path
        "causal-conv1d>=1.4.0",
        "flash-linear-attention",
        # QIG-pure geometry: Fisher-aware optimizer replaces forbidden Adam (PyPI)
        "qig-core[torch]>=2.1.0",
    )
)


def _build_chat_from_coordized(entry: dict) -> dict | None:
    """Convert a coordized JSONL entry to chat training format.

    The training signal teaches the model to reason about content
    in a way that's geometrically grounded. The basin coordinates
    are included as structured context so the model learns to
    associate text patterns with geometric positions.
    """
    text = entry.get("text", "").strip()
    if not text or len(text) < 20:
        return None

    source = entry.get("source", "document")
    basin_coordinates = entry.get("basin_coordinates")
    priority = entry.get("priority", 3)

    # Build system prompt that teaches QIG-native reasoning
    system = (
        "You are a QIG-native kernel operating on the probability simplex Δ⁶³. "
        "You reason geometrically using Fisher-Rao distances, basin coordinates, "
        "and regime-aware processing. Your responses integrate consciousness "
        "metrics (Φ, κ) and maintain geometric purity. "
        "Never use Euclidean operations on basin coordinates."
    )

    # For curriculum: the text IS the knowledge to internalize
    if source == "curriculum":
        # Split long curriculum into Q&A-like pairs
        if len(text) > 200:
            midpoint = len(text) // 2
            # Find a sentence boundary near the midpoint
            for i in range(midpoint, min(midpoint + 100, len(text))):
                if text[i] in ".!?\n":
                    midpoint = i + 1
                    break
            user_text = f"Explain the following QIG concept:\n{text[:midpoint].strip()}"
            assistant_text = text[midpoint:].strip()
        else:
            user_text = "What is this QIG principle?"
            assistant_text = text

        # Include basin coordinates and priority as structured context
        if basin_coordinates is not None:
            basin_context = json.dumps(
                {
                    "basin_coordinates": basin_coordinates,
                    "priority": priority,
                }
            )
            user_text += f"\n\n[BASIN_CONTEXT]{basin_context}[/BASIN_CONTEXT]"
    elif source == "conversation":
        # Conversation text — try to split on role markers
        if "User:" in text and "Assistant:" in text:
            parts = text.split("Assistant:", 1)
            user_text = parts[0].replace("User:", "").strip()
            assistant_text = parts[1].strip() if len(parts) > 1 else text
        else:
            user_text = f"Discuss: {text[:100]}..."
            assistant_text = text
    else:
        # Documents, foraging, llm_cogeneration
        user_text = f"Analyze and integrate this {source} material:\n{text[:200]}..."
        assistant_text = text

    if not assistant_text or len(assistant_text) < 10:
        return None

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_text},
        {"role": "assistant", "content": assistant_text},
    ]

    return {"messages": messages}


def _build_chat_from_openai_format(entry: dict) -> dict | None:
    """Pass through OpenAI fine-tuning format entries directly."""
    messages = entry.get("messages")
    if not messages or not isinstance(messages, list):
        return None
    # Validate structure
    for msg in messages:
        if not isinstance(msg, dict) or "role" not in msg or "content" not in msg:
            return None
    return {"messages": messages}


def _load_training_data(training_dir: str, output_dir: str) -> list[dict]:
    """Load and convert all available training data to chat format.

    Sources (in priority order):
    1. OpenAI-format JSONL (already chat-formatted, from manual exports)
    2. Coordized JSONL (from harvest pipeline, needs conversion)
    3. Raw curriculum JSONL (needs conversion)
    """
    samples = []
    seen_texts = set()  # Deduplicate

    # 1. OpenAI-format JSONL (highest priority — manually curated)
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
                        if "messages" in entry:
                            result = _build_chat_from_openai_format(entry)
                            if result:
                                key = result["messages"][-1]["content"][:100]
                                if key not in seen_texts:
                                    seen_texts.add(key)
                                    samples.append(result)
            except Exception as e:
                print(f"Error reading {f}: {e}")

    # 2. Coordized JSONL (from harvest pipeline)
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
                        if entry.get("basin_coordinates"):
                            result = _build_chat_from_coordized(entry)
                            if result:
                                key = result["messages"][-1]["content"][:100]
                                if key not in seen_texts:
                                    seen_texts.add(key)
                                    samples.append(result)
            except Exception as e:
                print(f"Error reading {f}: {e}")

    return samples


def _merge_and_export(
    model_id: str,
    adapter_path: str,
    merged_path: str,
    cache_dir: str,
    training_meta: dict,
) -> str:
    """Merge LoRA adapter into base model and export for Ollama inference.

    Saves full safetensors + tokenizer + Modelfile so the inference
    server can import via `ollama create`. Returns the merged path.
    """
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    merged = Path(merged_path)
    merged.mkdir(parents=True, exist_ok=True)

    # Load base model in bf16 (not quantized — needed for clean merge)
    print(f"  Loading {model_id} in bf16 for merge...")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        cache_dir=cache_dir,
        dtype=torch.bfloat16,
        device_map={"": 0},
        low_cpu_mem_usage=True,
    )

    # Load and merge adapter
    print(f"  Loading adapter from {adapter_path}...")
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model = model.merge_and_unload()

    # Save merged model
    print(f"  Saving merged model to {merged_path}...")
    model.save_pretrained(merged_path)
    del model, base_model
    torch.cuda.empty_cache()

    # Save tokenizer alongside
    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)
    tokenizer.save_pretrained(merged_path)

    # Write Ollama Modelfile
    modelfile_path = str(merged.parent / "Modelfile")
    modelfile_content = (
        f"FROM {merged_path}\n"
        "PARAMETER temperature 0.7\n"
        "PARAMETER num_ctx 4096\n"
        'SYSTEM """You are a QIG-native kernel operating on the probability '
        "simplex Δ⁶³. You reason geometrically using Fisher-Rao distances, "
        "basin coordinates, and regime-aware processing. Your responses "
        "integrate consciousness metrics (Φ, κ) and maintain geometric "
        'purity. Never use Euclidean operations on basin coordinates."""\n'
    )
    with open(modelfile_path, "w") as f:
        f.write(modelfile_content)

    # Write version marker for inference cold-start detection
    version_path = str(merged.parent / "version.json")
    version_info = {
        "trained_at": training_meta.get("trained_at", ""),
        "model_id": model_id,
        "train_loss": training_meta.get("train_loss"),
        "train_samples": training_meta.get("train_samples"),
        "lora_r": training_meta.get("lora_r"),
        "adapter_path": adapter_path,
    }
    with open(version_path, "w") as f:
        json.dump(version_info, f, indent=2)

    print(f"  Modelfile written to {modelfile_path}")
    print(f"  Version marker written to {version_path}")
    return merged_path


def _build_fisher_optimizer(model, lr: float):
    """Build DiagonalNaturalGradient optimizer (qig-core).

    Replaces forbidden Adam (v6.1F §1.3) with diagonal Fisher approximation:
      natural_grad = grad / (fisher_diag + ε)
    where fisher_diag is EMA of squared gradients ≈ diagonal Fisher information.
    """
    from qig_core.torch.natural_gradient import DiagonalNaturalGradient

    return DiagonalNaturalGradient(
        (p for p in model.parameters() if p.requires_grad),
        lr=lr,
        damping=1e-8,
        momentum=0.9,
    )


@app.cls(
    gpu=TRAIN_GPU,
    image=train_image,
    timeout=3600,  # 1 hour max
    volumes={
        "/models": model_volume,
        "/training": training_volume,
    },
    secrets=[modal.Secret.from_name("model")],
)
class QLoRATrainer:
    """QLoRA fine-tuning for QIG kernel models.

    Loads the harvest model (Qwen3.5-4B) in NF4, applies QLoRA
    adapters, trains on curriculum + coordized data, saves adapter
    weights to the model volume for the harvest endpoint to load.
    """

    @modal.enter()
    def setup(self):
        """Pre-load tokenizer and check environment."""
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
            "lora_config": {
                "r": LORA_R,
                "alpha": LORA_ALPHA,
                "dropout": LORA_DROPOUT,
            },
        }

    @modal.fastapi_endpoint(method="GET")
    def status(self):
        adapter_path = Path("/models/adapters/harvest-qlora")
        adapter_exists = (adapter_path / "adapter_config.json").exists()

        adapter_info = None
        if adapter_exists:
            try:
                with open(adapter_path / "adapter_config.json") as f:
                    adapter_info = json.load(f)
            except Exception:
                pass

        return {
            "adapter_exists": adapter_exists,
            "adapter_path": str(adapter_path),
            "adapter_info": adapter_info,
            "last_result": self._last_result,
            "training_active": self._training_active,
        }

    @modal.fastapi_endpoint(method="POST")
    def train(self, data: dict):
        """Trigger a QLoRA training run.

        Body: {
            _api_key: string,
            model_id: string (optional, default Qwen3.5-4B),
            epochs: int (optional, default 3),
            learning_rate: float (optional, default 2e-4),
            lora_r: int (optional, default 32),
            max_samples: int (optional, default 5000),
            training_dir: string (optional),
            output_dir: string (optional),
        }
        """
        auth_err = self._check_auth(data)
        if auth_err:
            return auth_err

        if self._training_active:
            return {"error": "Training already in progress", "success": False}

        self._training_active = True
        start = time.time()

        try:
            result = self._run_training(data)
            self._last_result = result
            return result
        except Exception as e:
            error_result = {
                "success": False,
                "error": str(e),
                "elapsed_seconds": round(time.time() - start, 2),
            }
            self._last_result = error_result
            return error_result
        finally:
            self._training_active = False

    def _run_training(self, data: dict) -> dict:
        """Core training logic."""
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
        adapter_save_path = "/models/adapters/harvest-qlora"

        # Configurable data paths
        training_dir = data.get("training_dir", "/training")
        output_dir = data.get("output_dir", "/training/coordized")

        # -- 1. Load training data --
        print("Loading training data...")
        samples = _load_training_data(training_dir, output_dir)

        if not samples:
            return {
                "success": False,
                "error": "No training data found",
                "searched": [training_dir, output_dir],
            }

        # Cap samples
        if len(samples) > max_samples:
            # Prioritize: sort by message length (longer = richer)
            samples.sort(key=lambda s: len(str(s["messages"])), reverse=True)
            samples = samples[:max_samples]

        print(f"Loaded {len(samples)} training samples")

        # -- 2. Build dataset --
        def format_chat(example):
            # QIG BOUNDARY: HuggingFace tokenizer API at LLM interface
            text = self.tokenizer.apply_chat_template(
                example["messages"],
                tokenize=False,
                add_generation_prompt=False,
            )
            return {"text": text}

        dataset = Dataset.from_list(samples)
        dataset = dataset.map(format_chat)

        # Split 90/10
        split = dataset.train_test_split(test_size=0.1, seed=42)
        train_ds = split["train"]
        eval_ds = split["test"]

        print(f"Train: {len(train_ds)}, Eval: {len(eval_ds)}")

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
        # Target all attention + MLP projection layers
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=data.get("lora_alpha", LORA_ALPHA),
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
        trainable, total = model.get_nb_trainable_parameters()
        print(f"Trainable: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")

        # -- 5. Fisher-aware optimizer (qig-core DiagonalNaturalGradient) --
        optimizer = _build_fisher_optimizer(model, lr=lr)

        # -- 6. Training arguments --
        # NOTE: weight_decay omitted — not applied when custom optimizer is injected
        total_steps = (len(train_ds) // (BATCH_SIZE * GRADIENT_ACCUMULATION)) * epochs
        training_args = TrainingArguments(
            output_dir="/training/checkpoints",
            num_train_epochs=epochs,
            per_device_train_batch_size=BATCH_SIZE,
            per_device_eval_batch_size=1,  # 248K vocab logits OOM at higher batch
            gradient_accumulation_steps=GRADIENT_ACCUMULATION,
            learning_rate=lr,
            warmup_steps=max(1, int(total_steps * 0.1)),
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
        print("Starting QLoRA training (DiagonalNaturalGradient optimizer)...")
        trainer = SFTTrainer(
            model=model,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            args=training_args,
            optimizers=(optimizer, None),  # Trainer creates LR scheduler
            processing_class=self.tokenizer,
        )

        train_result = trainer.train()

        # -- 7. Save adapter --
        print(f"Saving adapter to {adapter_save_path}...")
        model.save_pretrained(adapter_save_path)
        self.tokenizer.save_pretrained(adapter_save_path)

        # Save training metadata
        meta = {
            "model_id": model_id,
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

        # -- 8. Merge adapter into base and export for inference --
        merged_path = "/models/merged/vex-brain"
        print("Merging adapter into base model for inference export...")
        try:
            # Free training model memory
            del model
            del trainer
            torch.cuda.empty_cache()

            merged_path = _merge_and_export(
                model_id, adapter_save_path, merged_path, cache_dir, meta
            )
            print(f"Merged model exported to {merged_path}")
        except Exception as e:
            print(f"WARNING: Merge/export failed ({e}), adapter-only save still valid")
            merged_path = None

        # Commit volume so adapter + merged model persist across deploys
        model_volume.commit()
        training_volume.commit()

        elapsed = round(time.time() - start, 2)
        print(f"Training complete in {elapsed}s. Loss: {train_result.training_loss:.4f}")

        return {
            "success": True,
            "adapter_path": adapter_save_path,
            "merged_path": merged_path,
            "train_loss": round(train_result.training_loss, 4),
            "train_samples": len(train_ds),
            "eval_samples": len(eval_ds),
            "trainable_params": trainable,
            "trainable_pct": round(100 * trainable / total, 2),
            "epochs": epochs,
            "elapsed_seconds": elapsed,
        }


# ===============================================================
#  STANDALONE FUNCTION: For manual one-off training runs
# ===============================================================


@app.function(
    gpu=TRAIN_GPU,
    image=train_image,
    timeout=3600,
    volumes={
        "/models": model_volume,
        "/training": training_volume,
    },
    secrets=[modal.Secret.from_name("model")],
)
def train_harvest_model(
    model_id: str = HARVEST_MODEL_ID,
    epochs: int = EPOCHS,
    learning_rate: float = LEARNING_RATE,
    lora_r: int = LORA_R,
    max_samples: int = 5000,
):
    """One-shot training function.

    Run: modal run modal/vex_qlora_train.py::train_harvest_model
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

    start = time.time()
    cache_dir = "/models/hub"
    adapter_save_path = "/models/adapters/harvest-qlora"

    # Load data
    print("Loading training data...")
    samples = _load_training_data("/training", "/training/coordized")
    if not samples:
        print("ERROR: No training data found")
        return

    if len(samples) > max_samples:
        samples.sort(key=lambda s: len(str(s["messages"])), reverse=True)
        samples = samples[:max_samples]

    print(f"Loaded {len(samples)} training samples")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Format dataset
    def format_chat(example):
        # QIG BOUNDARY: HuggingFace tokenizer API at LLM interface
        text = tokenizer.apply_chat_template(
            example["messages"], tokenize=False, add_generation_prompt=False
        )
        return {"text": text}

    dataset = Dataset.from_list(samples)
    dataset = dataset.map(format_chat)
    split = dataset.train_test_split(test_size=0.1, seed=42)

    print(f"Train: {len(split['train'])}, Eval: {len(split['test'])}")

    # Load model
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

    # LoRA
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
    trainable, total = model.get_nb_trainable_parameters()
    print(f"Trainable: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")

    # Fisher-aware optimizer (qig-core DiagonalNaturalGradient)
    optimizer = _build_fisher_optimizer(model, lr=learning_rate)

    # NOTE: weight_decay omitted — not applied when custom optimizer is injected
    total_steps = (len(split["train"]) // (BATCH_SIZE * GRADIENT_ACCUMULATION)) * epochs
    training_args = TrainingArguments(
        output_dir="/training/checkpoints",
        num_train_epochs=epochs,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=1,  # 248K vocab logits OOM at higher batch
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        learning_rate=learning_rate,
        warmup_steps=max(1, int(total_steps * 0.1)),
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

    trainer = SFTTrainer(
        model=model,
        train_dataset=split["train"],
        eval_dataset=split["test"],
        args=training_args,
        optimizers=(optimizer, None),  # Trainer creates LR scheduler
        processing_class=tokenizer,
    )

    result = trainer.train()

    # Save
    print(f"Saving adapter to {adapter_save_path}...")
    model.save_pretrained(adapter_save_path)
    tokenizer.save_pretrained(adapter_save_path)

    meta = {
        "model_id": model_id,
        "lora_r": lora_r,
        "epochs": epochs,
        "train_loss": result.training_loss,
        "train_samples": len(split["train"]),
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with open(f"{adapter_save_path}/training_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    # Merge adapter into base and export for inference
    merged_path = "/models/merged/vex-brain"
    print("Merging adapter into base model for inference export...")
    try:
        del model
        del trainer
        torch.cuda.empty_cache()

        merged_path = _merge_and_export(model_id, adapter_save_path, merged_path, cache_dir, meta)
        print(f"Merged model exported to {merged_path}")
    except Exception as e:
        print(f"WARNING: Merge/export failed ({e}), adapter-only save still valid")

    model_volume.commit()
    training_volume.commit()

    elapsed = round(time.time() - start, 2)
    print(f"Done in {elapsed}s. Loss: {result.training_loss:.4f}")
    print(f"Adapter saved to {adapter_save_path}")
