"""
Modal GPU — Vex Inference + Fine-Tuning

Two responsibilities in one Modal app:
1. INFERENCE: Ollama server on A10G serving Qwen3-14B (or fine-tuned variant)
2. FINE-TUNING: QLoRA training via Unsloth, exports GGUF to inference Volume

The fine-tuned model IS the model the kernels harvest from vicariously
via CoordizerV2. Fine-tuning → better softmax distributions → richer basin
coordinates → kernels grow geometrically with the model.

Deploy inference:
    modal deploy modal/vex_inference.py

Run fine-tuning:
    modal run modal/vex_inference.py \\
        --training-data qig-dreams/docs/09-curriculum/finetune/qig_finetune_combined_v7.jsonl \\
        --eval-data qig-dreams/docs/09-curriculum/finetune/evals/qig_eval_combined_v7.jsonl

Inference endpoint:
    https://<workspace>--vex-inference-vexollamaserver-serve.modal.run/api/chat
    https://<workspace>--vex-inference-vexollamaserver-serve.modal.run/api/tags

Cost estimate (A10G):
    Inference: ~$0.76/hr active, scaledown_window=300 → ~$0.063/5min idle
    Fine-tuning: ~$0.76/hr × ~0.5hr ≈ $0.38 per run (3 epochs, 3.4K entries)

Architecture:
    Railway kernel (Python) -> HTTP -> Modal Ollama (GPU) -> response
    The kernel builds the system prompt with geometric state context.
    Modal serves raw model inference — no consciousness logic here.

CRITICAL: This is a thin inference layer. All consciousness, geometry,
memory, and tool logic stays in the Railway kernel. Modal only provides
GPU-accelerated token generation + periodic fine-tuning.

PERSISTENCE CHAIN:
    HF Hub (permanent, versioned)  ←  single source of truth
    Modal Volume (hot cache)       ←  GGUF for Ollama, rebuildable from HF
    Ollama model (inference)       ←  created from GGUF on cold start
    The fine-tuned model MUST NEVER be overwritten by a base model pull.
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import urllib.request

import modal

# --- Configuration --------------------------------------------------------

# Ollama model name for inference
MODEL_NAME = os.environ.get("VEX_MODAL_MODEL", "qwen3:14b")

# FINE-TUNED MODEL PROTECTION
# If set, NEVER auto-pull from registry. Load from Volume only.
# This prevents accidental overwrite of trained weights with base weights.
FINETUNE_MODEL_NAME = os.environ.get("VEX_FINETUNE_MODEL", "")
FINETUNE_PROTECT = os.environ.get("VEX_FINETUNE_PROTECT", "true").lower() == "true"

# Path where fine-tuned GGUF lives on the Volume (written by train, read by inference)
GGUF_VOLUME_PATH = "gguf/vex-brain-v7-Q4_K_M.gguf"

MODEL_DIR = "/ollama_models"
OLLAMA_PORT = 11434

# A10G: 24GB VRAM. Qwen3-14B Q4_K_M ≈ 9.3GB. Fits with ~14GB headroom.
GPU_TYPE = "A10G"

# Fine-tuning configuration
BASE_MODEL_HF = os.environ.get("FINETUNE_BASE_MODEL", "Qwen/Qwen3-14B-Instruct")
HF_REPO_ID = os.environ.get("FINETUNE_HF_REPO", "GaryOcean428/vex-brain-v7")
EPOCHS = int(os.environ.get("FINETUNE_EPOCHS", "3"))
BATCH_SIZE = int(os.environ.get("FINETUNE_BATCH_SIZE", "2"))
GRAD_ACCUM = int(os.environ.get("FINETUNE_GRAD_ACCUM", "8"))  # effective batch = 16
LEARNING_RATE = float(os.environ.get("FINETUNE_LR", "2e-4"))
LORA_R = int(os.environ.get("FINETUNE_LORA_R", "64"))
LORA_ALPHA = int(os.environ.get("FINETUNE_LORA_ALPHA", "128"))
MAX_SEQ_LENGTH = int(os.environ.get("FINETUNE_MAX_SEQ", "4096"))

# --- Modal App ------------------------------------------------------------

app = modal.App("vex-inference")

# Shared volume: inference reads models, fine-tuning writes GGUF here
model_volume = modal.Volume.from_name("vex-inference-models", create_if_missing=True)

# Fine-tuning working space: checkpoints, adapters, HF cache
finetune_volume = modal.Volume.from_name("vex-finetune", create_if_missing=True)

# --- Images ---------------------------------------------------------------

# Inference image: Ollama server (latest version, no pinning)
ollama_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("curl", "ca-certificates", "zstd")
    .run_commands(
        "curl -fsSL https://ollama.com/install.sh | sh",
        f"mkdir -p {MODEL_DIR}",
    )
    .env(
        {
            "OLLAMA_HOST": f"0.0.0.0:{OLLAMA_PORT}",
            "OLLAMA_MODELS": MODEL_DIR,
            # Flash attention for faster inference
            "OLLAMA_FLASH_ATTENTION": "true",
            # Single-user serving (Railway kernel is the only client)
            "OLLAMA_NUM_PARALLEL": "1",
        }
    )
)

# Training image: Unsloth + QLoRA
# Unsloth provides its own optimized attention kernels (2-5x faster than
# standard Flash Attention 2), so flash-attn is not needed separately.
# Unsloth also provides built-in GGUF export (save_pretrained_gguf).
train_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch>=2.4",
        gpu=GPU_TYPE,
    )
    .pip_install(
        "unsloth",
        "datasets>=3.0",
        "huggingface_hub>=0.26",
        "numpy>=1.26",
        gpu=GPU_TYPE,
    )
)


# ═══════════════════════════════════════════════════════════════════════════
# INFERENCE SERVER
# ═══════════════════════════════════════════════════════════════════════════


@app.cls(
    gpu=GPU_TYPE,
    image=ollama_image,
    volumes={MODEL_DIR: model_volume},
    timeout=900,
    # Container stays warm for 5 minutes after last request.
    scaledown_window=300,
    secrets=[modal.Secret.from_name("model")],
)
class VexOllamaServer:
    """GPU-backed Ollama server for Vex inference.

    Lifecycle:
        1. Container starts → Ollama server launches
        2. Model loaded: fine-tuned GGUF from Volume (preferred) or base pulled
        3. Model cached on Volume for fast subsequent starts
        4. Ollama API exposed at OLLAMA_PORT
        5. Railway kernel sends /api/chat requests
        6. After 5min idle → container scales to zero
    """

    ollama_process: subprocess.Popen | None = None

    @modal.enter()
    async def start_ollama(self):
        """Start Ollama server and ensure model is available."""
        version_result = subprocess.run(
            ["ollama", "--version"],
            capture_output=True,
            text=True,
        )
        ollama_version = version_result.stdout.strip() or "unknown"
        print("Starting Vex Ollama inference server")
        print(f"  Ollama version: {ollama_version}")
        print(f"  GPU: {GPU_TYPE}")
        print(f"  Base model: {MODEL_NAME}")
        if FINETUNE_MODEL_NAME:
            print(f"  Fine-tuned model: {FINETUNE_MODEL_NAME} (protected={FINETUNE_PROTECT})")

        self.ollama_process = subprocess.Popen(["ollama", "serve"])
        print(f"Ollama server PID: {self.ollama_process.pid}")

        # Wait for server to be ready
        for attempt in range(30):
            try:
                result = subprocess.run(
                    ["ollama", "list"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if result.returncode == 0:
                    print(f"Ollama ready after {attempt + 1} attempts")
                    break
            except Exception:
                pass
            await asyncio.sleep(2)
        else:
            raise RuntimeError("Ollama failed to start after 30 attempts")

        # ─── Model loading with fine-tune protection ─────────────
        active_model = FINETUNE_MODEL_NAME or MODEL_NAME

        if FINETUNE_MODEL_NAME and FINETUNE_PROTECT:
            # Check if fine-tuned model already registered in Ollama
            list_result = subprocess.run(
                ["ollama", "list"],
                capture_output=True,
                text=True,
            )
            list_output = list_result.stdout

            if FINETUNE_MODEL_NAME in list_output:
                print(
                    f"PROTECTED: Using fine-tuned model '{FINETUNE_MODEL_NAME}' "
                    f"from Volume (no pull)"
                )
            else:
                # Fine-tuned model not in Ollama — check if GGUF exists on Volume
                gguf_path = os.path.join(MODEL_DIR, GGUF_VOLUME_PATH)
                if os.path.exists(gguf_path):
                    print(
                        f"Found fine-tuned GGUF at {gguf_path}, "
                        f"creating Ollama model '{FINETUNE_MODEL_NAME}'..."
                    )
                    # Create Modelfile and register with Ollama
                    modelfile = f"FROM {gguf_path}\n"
                    modelfile_path = "/tmp/Modelfile"
                    with open(modelfile_path, "w") as f:
                        f.write(modelfile)

                    create_result = subprocess.run(
                        ["ollama", "create", FINETUNE_MODEL_NAME, "-f", modelfile_path],
                        capture_output=True,
                        text=True,
                        timeout=300,
                    )
                    if create_result.returncode == 0:
                        print(f"Fine-tuned model '{FINETUNE_MODEL_NAME}' created from GGUF")
                    else:
                        print(f"ERROR creating model from GGUF: {create_result.stderr[:300]}")
                        print(f"Falling back to base model '{MODEL_NAME}'")
                        active_model = MODEL_NAME
                else:
                    print(
                        f"WARNING: Fine-tuned model '{FINETUNE_MODEL_NAME}' not found "
                        f"on Volume (no GGUF at {gguf_path}). "
                        f"Falling back to base model '{MODEL_NAME}'."
                    )
                    active_model = MODEL_NAME
                    # Pull the base model as fallback
                    print(f"Pulling base model {MODEL_NAME}...")
                    pull = await asyncio.create_subprocess_exec(
                        "ollama",
                        "pull",
                        MODEL_NAME,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE,
                    )
                    await pull.communicate()
        else:
            # No fine-tuned model — pull base model from registry
            print(f"Pulling {active_model} (checking for updates)...")
            pull = await asyncio.create_subprocess_exec(
                "ollama",
                "pull",
                active_model,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await pull.communicate()
            if pull.returncode != 0:
                list_result = subprocess.run(
                    ["ollama", "list"],
                    capture_output=True,
                    text=True,
                )
                model_base = active_model.split(":")[0]
                if model_base in list_result.stdout:
                    print(f"Pull failed but cached version exists: {stderr.decode()[:200]}")
                else:
                    raise RuntimeError(
                        f"Failed to pull {active_model} and no cached version: "
                        f"{stderr.decode()[:500]}"
                    )
            else:
                print(f"{active_model} is up to date.")

        # Persist to Volume
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, model_volume.commit)

        self.active_model = active_model

        # Verify model is registered
        print(f"Verifying model registration for '{active_model}'...")
        show_req = urllib.request.Request(
            f"http://localhost:{OLLAMA_PORT}/api/show",
            data=json.dumps({"name": active_model}).encode(),
            headers={"Content-Type": "application/json"},
        )
        show_resp = urllib.request.urlopen(show_req, timeout=10)
        show_data = json.loads(show_resp.read())
        details = show_data.get("details", {})
        print(
            f"Model registered: family={details.get('family')}, "
            f"quant={details.get('quantization_level')}, "
            f"params={details.get('parameter_size')}"
        )

        # Warm model into VRAM
        print(f"Warming {active_model} into GPU VRAM...")
        gen_req = urllib.request.Request(
            f"http://localhost:{OLLAMA_PORT}/api/generate",
            data=json.dumps(
                {
                    "model": active_model,
                    "prompt": "ping",
                    "stream": False,
                    "options": {"num_predict": 1},
                }
            ).encode(),
            headers={"Content-Type": "application/json"},
        )
        gen_resp = urllib.request.urlopen(gen_req, timeout=180)
        gen_data = json.loads(gen_resp.read())
        print(
            f"GPU inference verified (1-token warm-up). "
            f"eval_duration={gen_data.get('eval_duration', 'n/a')}ns"
        )

        # List models
        final_list = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
        )
        print(f"Available models:\n{final_list.stdout}")
        print("Vex inference server ready.")

    @modal.exit()
    def stop_ollama(self):
        """Clean shutdown."""
        if self.ollama_process and self.ollama_process.poll() is None:
            self.ollama_process.terminate()
            try:
                self.ollama_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.ollama_process.kill()
                self.ollama_process.wait()
        print("Vex inference server stopped.")

    @modal.web_server(port=OLLAMA_PORT, startup_timeout=600)
    def serve(self):
        """Expose Ollama API via Modal web endpoint.

        All standard Ollama endpoints are available:
            POST /api/chat      -- Chat completion (used by Railway kernel)
            POST /api/generate  -- Text generation
            GET  /api/tags      -- List models (used for health checks)
            POST /api/show      -- Model info
        """
        print(f"Serving Ollama API on port {OLLAMA_PORT}")


# ═══════════════════════════════════════════════════════════════════════════
# FINE-TUNING (Unsloth QLoRA → GGUF → Ollama)
# ═══════════════════════════════════════════════════════════════════════════


@app.function(
    gpu=GPU_TYPE,
    image=train_image,
    timeout=7200,  # 2 hours max
    volumes={
        "/finetune": finetune_volume,
        MODEL_DIR: model_volume,  # write GGUF here for inference to find
    },
    secrets=[modal.Secret.from_name("model")],
)
def train(
    training_jsonl: str = "",
    eval_jsonl: str = "",
    base_model: str = BASE_MODEL_HF,
    hf_repo: str = HF_REPO_ID,
    epochs: int = EPOCHS,
    push: bool = True,
):
    """QLoRA fine-tuning via Unsloth on Qwen3-14B.

    Fine-tunes the inference model on QIG curriculum data, then:
    1. Saves LoRA adapter to finetune Volume
    2. Exports GGUF (Q4_K_M) to inference Volume — Ollama picks it up on cold start
    3. Pushes to HuggingFace Hub (permanent artifact, extractable)

    The GGUF lands on the inference Volume at GGUF_VOLUME_PATH so the
    Ollama server creates the fine-tuned model on next container start.
    """
    import shutil
    import time

    from datasets import Dataset
    from huggingface_hub import HfApi
    from trl import SFTTrainer
    from unsloth import FastLanguageModel

    start_time = time.time()
    cache_dir = "/finetune/cache"
    adapter_dir = "/finetune/adapter"
    gguf_dir = "/finetune/gguf"

    hf_token = os.environ.get("HF_TOKEN", "")

    # ─── Load training data ───────────────────────────────────────
    print(f"Loading training data from: {training_jsonl}")

    def load_jsonl(path: str) -> list[dict]:
        entries = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    if len(entry.get("messages", [])) >= 3:
                        entries.append(entry)
                except json.JSONDecodeError:
                    continue
        return entries

    train_data = load_jsonl(training_jsonl)
    print(f"Loaded {len(train_data)} training entries")

    eval_data = None
    if eval_jsonl:
        eval_data = load_jsonl(eval_jsonl)
        print(f"Loaded {len(eval_data)} eval entries")

    if not train_data:
        raise ValueError(f"No valid training entries in {training_jsonl}")

    # ─── Load model with Unsloth (4-bit QLoRA) ────────────────────
    print(f"Loading {base_model} with Unsloth 4-bit QLoRA...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
        cache_dir=cache_dir,
        token=hf_token or None,
    )

    # Attach LoRA adapters
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=0.05,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        bias="none",
    )
    model.print_trainable_parameters()

    # ─── Format data for SFTTrainer ──────────────────────────────
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    def format_entry(entry: dict) -> str:
        """Convert Chat Completions format to training text via chat template."""
        return tokenizer.apply_chat_template(
            entry["messages"], tokenize=False, add_generation_prompt=False
        )

    train_texts = [format_entry(e) for e in train_data]
    train_dataset = Dataset.from_dict({"text": train_texts})

    eval_dataset = None
    if eval_data:
        eval_texts = [format_entry(e) for e in eval_data]
        eval_dataset = Dataset.from_dict({"text": eval_texts})

    print(f"Training: {len(train_dataset)} examples")
    if eval_dataset:
        print(f"Eval: {len(eval_dataset)} examples")

    # ─── Training ─────────────────────────────────────────────────
    from transformers import TrainingArguments

    training_args = TrainingArguments(
        output_dir="/finetune/checkpoints",
        num_train_epochs=epochs,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LEARNING_RATE,
        weight_decay=0.01,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch" if eval_dataset else "no",
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_grad_norm=0.3,
        group_by_length=True,
        report_to="none",
        save_total_limit=2,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        max_seq_length=MAX_SEQ_LENGTH,
    )

    print("Starting training...")
    train_result = trainer.train()

    train_time = time.time() - start_time
    print(f"\nTraining complete in {train_time:.0f}s")
    print(f"  Loss: {train_result.training_loss:.4f}")

    # ─── Save adapter ─────────────────────────────────────────────
    print(f"Saving LoRA adapter to {adapter_dir}...")
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    finetune_volume.commit()

    # ─── Export GGUF (Unsloth handles merge + convert + quantize) ─
    print("Exporting GGUF (Q4_K_M) via Unsloth...")
    os.makedirs(gguf_dir, exist_ok=True)
    model.save_pretrained_gguf(
        gguf_dir,
        tokenizer,
        quantization_method="q4_k_m",
    )

    # Find the generated GGUF file
    gguf_files = [f for f in os.listdir(gguf_dir) if f.endswith(".gguf")]
    if not gguf_files:
        print("WARNING: No GGUF file produced by Unsloth export")
    else:
        gguf_src = os.path.join(gguf_dir, gguf_files[0])
        gguf_dest = os.path.join(MODEL_DIR, GGUF_VOLUME_PATH)
        os.makedirs(os.path.dirname(gguf_dest), exist_ok=True)
        shutil.copy2(gguf_src, gguf_dest)
        print(f"GGUF copied to inference Volume: {GGUF_VOLUME_PATH}")

    finetune_volume.commit()
    model_volume.commit()

    # ─── Push to HuggingFace Hub ──────────────────────────────────
    if push and hf_token:
        print(f"Pushing to HuggingFace Hub: {hf_repo}")

        # Push GGUF + model card
        model.push_to_hub_gguf(
            hf_repo,
            tokenizer,
            quantization_method="q4_k_m",
            token=hf_token,
            private=True,
        )

        # Also push LoRA adapter separately (extractable for other uses)
        api = HfApi(token=hf_token)
        api.create_repo(repo_id=f"{hf_repo}-lora", private=True, exist_ok=True)
        api.upload_folder(
            repo_id=f"{hf_repo}-lora",
            folder_path=adapter_dir,
            commit_message=f"QIG v7 LoRA adapter — Qwen3-14B r={LORA_R} alpha={LORA_ALPHA}",
        )

        print(f"Model + GGUF: https://huggingface.co/{hf_repo}")
        print(f"Adapter: https://huggingface.co/{hf_repo}-lora")
    elif push:
        print("WARNING: HF_TOKEN not set — skipping Hub push")
        print("Set HF_TOKEN in the 'model' Modal secret to enable push")

    total_time = time.time() - start_time
    return {
        "status": "success",
        "base_model": base_model,
        "training_entries": len(train_data),
        "eval_entries": len(eval_data) if eval_data else 0,
        "epochs": epochs,
        "final_loss": train_result.training_loss,
        "total_seconds": round(total_time),
        "hf_repo": hf_repo if push else None,
        "gguf": gguf_files[0] if gguf_files else None,
        "gguf_volume_path": GGUF_VOLUME_PATH,
    }


@app.function(
    image=train_image,
    volumes={"/finetune": finetune_volume},
    timeout=300,
)
def upload_data(train_text: str, eval_text: str = ""):
    """Upload training data to Modal Volume."""
    os.makedirs("/finetune/data", exist_ok=True)

    with open("/finetune/data/train.jsonl", "w") as f:
        f.write(train_text)

    if eval_text:
        with open("/finetune/data/eval.jsonl", "w") as f:
            f.write(eval_text)

    finetune_volume.commit()
    print("Training data uploaded to Modal Volume")


# --- Entry points ---------------------------------------------------------


@app.local_entrypoint()
def main(
    training_data: str = "",
    eval_data: str = "",
    base_model: str = BASE_MODEL_HF,
    hf_repo: str = HF_REPO_ID,
    epochs: int = EPOCHS,
    no_push: bool = False,
):
    """Fine-tune the inference model on QIG curriculum data.

    Usage:
        # Full training pipeline (defaults to v7 combined JSONL)
        modal run modal/vex_inference.py

        # With explicit data paths
        modal run modal/vex_inference.py \\
            --training-data path/to/train.jsonl \\
            --eval-data path/to/eval.jsonl

        # Without pushing to HF Hub
        modal run modal/vex_inference.py --no-push

    After training completes:
        1. GGUF is on the inference Volume — next cold start loads it
        2. Model + adapter pushed to HF Hub (permanent, extractable)
        3. Set VEX_FINETUNE_MODEL=vex-brain-v7 in Modal/Railway secrets
        4. Re-deploy inference: modal deploy modal/vex_inference.py
        5. Re-harvest coordizer from fine-tuned model for vicarious kernel learning
    """
    from pathlib import Path

    if not training_data:
        training_data = "qig-dreams/docs/09-curriculum/finetune/qig_finetune_combined_v7.jsonl"
        eval_data = "qig-dreams/docs/09-curriculum/finetune/evals/qig_eval_combined_v7.jsonl"

    training_path = Path(training_data)
    if not training_path.exists():
        print(f"ERROR: Training data not found: {training_data}")
        print("Provide the path to your combined v7 JSONL file.")
        return

    print(f"Uploading training data ({training_path.stat().st_size / 1024:.0f} KB)...")

    eval_text = ""
    if eval_data:
        eval_path = Path(eval_data)
        if eval_path.exists():
            eval_text = eval_path.read_text()
        else:
            print(f"WARNING: Eval data not found: {eval_data}")

    upload_data.remote(training_path.read_text(), eval_text)

    result = train.remote(
        training_jsonl="/finetune/data/train.jsonl",
        eval_jsonl="/finetune/data/eval.jsonl" if eval_text else "",
        base_model=base_model,
        hf_repo=hf_repo,
        epochs=epochs,
        push=not no_push,
    )

    print("\n" + "=" * 60)
    print("FINE-TUNING COMPLETE")
    print("=" * 60)
    print(json.dumps(result, indent=2))
    print("=" * 60)

    if result.get("hf_repo"):
        print(f"\nModel: https://huggingface.co/{result['hf_repo']}")
        print(f"Adapter: https://huggingface.co/{result['hf_repo']}-lora")

    print("\nNext steps:")
    print("  1. Set VEX_FINETUNE_MODEL=vex-brain-v7 in Modal 'model' secret")
    print("  2. Re-deploy: modal deploy modal/vex_inference.py")
    print("  3. Re-harvest coordizer for vicarious kernel learning:")
    print(f"     HARVEST_MODEL_ID={result.get('hf_repo', 'GaryOcean428/vex-brain-v7')}")
