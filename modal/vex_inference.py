"""
Modal GPU Inference — Vex Ollama Backend

Runs Qwen3.5-27B (or a QLoRA-fine-tuned variant) via Ollama on A10G GPU,
exposing the standard Ollama API via Modal web_server.

Fine-tuned model flow:
    QLoRA training saves merged safetensors to vex-models:/models/merged/vex-brain/
    On cold start, this server detects the merged model, imports it into Ollama
    as "vex-brain", and uses it instead of the base model. The fine-tuned model
    persists in the Ollama volume across restarts — it is NOT re-pulled.

Deploy:
    modal deploy modal/vex_inference.py

Endpoint:
    https://<workspace>--vex-inference-serve.modal.run/api/chat
    (full Ollama API available)

Cost estimate (A10G):
    ~$1.10/hr active, scaledown_window=300

Architecture:
    Railway kernel (Python) -> HTTP -> Modal Ollama (GPU) -> response
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess

import modal

# --- Configuration --------------------------------------------------------
BASE_MODEL = os.environ.get("VEX_MODAL_MODEL", "qwen3.5:27b")
FINE_TUNED_NAME = "vex-brain"  # Ollama model name for fine-tuned variant
MODEL_DIR = "/ollama_models"
FINE_TUNED_DIR = "/fine-tuned"  # Mount point for vex-models volume
MERGED_MODEL_PATH = f"{FINE_TUNED_DIR}/merged/vex-brain"
VERSION_PATH = f"{FINE_TUNED_DIR}/merged/version.json"
MODELFILE_PATH = f"{FINE_TUNED_DIR}/merged/Modelfile"
OLLAMA_PORT = 11434

GPU_TYPE = os.environ.get("VEX_GPU_TYPE", "A10G")

# --- Modal App ------------------------------------------------------------

app = modal.App("vex-inference")

model_volume = modal.Volume.from_name("vex-inference-models", create_if_missing=True)
fine_tuned_volume = modal.Volume.from_name("vex-models", create_if_missing=True)

# Force image rebuild: 2026-03-15 — Qwen3.5 requires Ollama >= 0.9
ollama_image = (
    modal.Image.debian_slim(python_version="3.14")
    .apt_install("curl", "ca-certificates", "zstd")
    .run_commands(
        "curl -fsSL https://ollama.com/install.sh | sh",
        "ollama --version",
        f"mkdir -p {MODEL_DIR}",
    )
    .env(
        {
            "OLLAMA_HOST": f"0.0.0.0:{OLLAMA_PORT}",
            "OLLAMA_MODELS": MODEL_DIR,
            "OLLAMA_FLASH_ATTENTION": "true",
            "OLLAMA_NUM_PARALLEL": "1",
        }
    )
)


@app.cls(
    gpu=GPU_TYPE,
    image=ollama_image,
    volumes={
        MODEL_DIR: model_volume,
        FINE_TUNED_DIR: fine_tuned_volume,
    },
    timeout=900,
    scaledown_window=300,
    secrets=[modal.Secret.from_name("model")],
)
class VexOllamaServer:
    """GPU-backed Ollama server for Vex inference.

    On cold start, checks the shared vex-models volume for a fine-tuned
    model (merged safetensors from QLoRA training). If found, imports it
    into Ollama as 'vex-brain' and uses it for all inference. The imported
    model persists in the Ollama volume — subsequent starts skip import.

    Fallback: if no fine-tuned model exists, pulls the base model.
    """

    ollama_process: subprocess.Popen | None = None
    active_model: str = BASE_MODEL

    async def start_ollama(self):
        """Start Ollama server, import fine-tuned model or pull base."""
        version_result = subprocess.run(
            ["ollama", "--version"],
            capture_output=True,
            text=True,
        )
        ollama_version = version_result.stdout.strip() or "unknown"
        print(f"  Ollama version: {ollama_version}")

        os.environ["OLLAMA_MODELS"] = MODEL_DIR

        self.ollama_process = subprocess.Popen(["ollama", "serve"])
        print(f"Ollama server PID: {self.ollama_process.pid}")

        # Wait for Ollama to be ready
        for _ in range(30):
            try:
                result = subprocess.run(
                    ["ollama", "list"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if result.returncode == 0:
                    break
            except Exception:
                pass
            await asyncio.sleep(1)

        # --- Fine-tuned model detection ---
        self.active_model = self._resolve_model()

        # Ensure the active model is available in Ollama
        check = subprocess.run(
            ["ollama", "show", self.active_model],
            capture_output=True,
            text=True,
        )
        if check.returncode == 0:
            print(f"Model {self.active_model} already cached, skipping pull/import")
        elif self.active_model == FINE_TUNED_NAME:
            # Import fine-tuned model from safetensors
            self._import_fine_tuned()
        else:
            # Pull base model from registry
            print(f"Pulling base model: {self.active_model}")
            pull_result = subprocess.run(
                ["ollama", "pull", self.active_model],
                capture_output=True,
                text=True,
                timeout=600,
            )
            if pull_result.returncode != 0:
                stderr = pull_result.stderr
                raise RuntimeError(f"Failed to pull {self.active_model}: {stderr}")

        final_list = subprocess.run(["ollama", "list"], capture_output=True, text=True)
        print(f"  Models available: {final_list.stdout}")

        model_volume.commit()
        print(f"Active model: {self.active_model}")

    def _resolve_model(self) -> str:
        """Decide which model to use: fine-tuned or base.

        Checks the shared vex-models volume for merged safetensors.
        If a newer version exists than what's already imported, returns
        the fine-tuned model name so it gets re-imported.
        """
        if not os.path.isfile(VERSION_PATH):
            print("No fine-tuned model found, using base model")
            return BASE_MODEL

        try:
            with open(VERSION_PATH) as f:
                version_info = json.load(f)
            trained_at = version_info.get("trained_at", "")
            print(f"Fine-tuned model found (trained: {trained_at})")
        except Exception as e:
            print(f"Error reading version info: {e}, using base model")
            return BASE_MODEL

        # Check if this version is already imported into Ollama
        imported_marker = f"{MODEL_DIR}/.vex-brain-version"
        if os.path.isfile(imported_marker):
            try:
                with open(imported_marker) as f:
                    imported_version = f.read().strip()
                if imported_version == trained_at:
                    print(f"Fine-tuned model already imported (version: {trained_at})")
                    return FINE_TUNED_NAME
            except Exception:
                pass

        # New version available — needs import
        print(f"New fine-tuned model available, will import (version: {trained_at})")
        return FINE_TUNED_NAME

    def _import_fine_tuned(self):
        """Import fine-tuned safetensors into Ollama as 'vex-brain'."""
        if not os.path.isfile(MODELFILE_PATH):
            print("ERROR: Modelfile not found, falling back to base model")
            self.active_model = BASE_MODEL
            return

        print(f"Importing fine-tuned model from {MERGED_MODEL_PATH}...")
        create_result = subprocess.run(
            ["ollama", "create", FINE_TUNED_NAME, "-f", MODELFILE_PATH],
            capture_output=True,
            text=True,
            timeout=600,
        )

        if create_result.returncode != 0:
            print(f"ERROR importing fine-tuned model: {create_result.stderr}")
            print("Falling back to base model")
            self.active_model = BASE_MODEL
            return

        print(f"Fine-tuned model imported as '{FINE_TUNED_NAME}'")

        # Write version marker so we don't re-import on next cold start
        try:
            with open(VERSION_PATH) as f:
                version_info = json.load(f)
            imported_marker = f"{MODEL_DIR}/.vex-brain-version"
            with open(imported_marker, "w") as f:
                f.write(version_info.get("trained_at", "unknown"))
        except Exception as e:
            print(f"Warning: could not write import marker: {e}")

    def stop_ollama(self):
        """Stop Ollama server."""
        if self.ollama_process and self.ollama_process.poll() is None:
            self.ollama_process.terminate()

    @modal.enter()
    async def on_enter(self):
        await self.start_ollama()

    @modal.exit()
    def on_exit(self):
        self.stop_ollama()

    @modal.web_server(port=OLLAMA_PORT, startup_timeout=300)
    def serve(self):
        # Ollama is already running from on_enter, serve just exposes the port
        pass
