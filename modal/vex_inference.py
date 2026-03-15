"""
Modal GPU Inference — Vex Ollama Backend

Runs Qwen3.5-35B-A3B (MoE: 35B total, 3B active) in bf16
on H100 GPU, exposing the standard Ollama API via Modal web_server.

Deploy:
    modal deploy modal/vex_inference.py

Endpoint:
    https://<workspace>--vex-inference-serve.modal.run/api/chat
    (full Ollama API available)

Cost estimate (H100):
    ~$3.50/hr active, scaledown_window=300

Architecture:
    Railway kernel (Python) -> HTTP -> Modal Ollama (GPU) -> response
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import urllib.request

import modal

# --- Configuration --------------------------------------------------------
MODEL_NAME = os.environ.get("VEX_MODAL_MODEL", "qwen3.5:35b-a3b-bf16")
MODEL_DIR = "/ollama_models"
OLLAMA_PORT = 11434

GPU_TYPE = os.environ.get("VEX_GPU_TYPE", "H100")

# --- Modal App ------------------------------------------------------------

app = modal.App("vex-inference")

model_volume = modal.Volume.from_name("vex-inference-models", create_if_missing=True)

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
    volumes={MODEL_DIR: model_volume},
    timeout=900,
    scaledown_window=300,
    secrets=[modal.Secret.from_name("model")],
)
class VexOllamaServer:
    """GPU-backed Ollama server for Vex inference."""

    ollama_process: subprocess.Popen | None = None

    async def start_ollama(self):
        """Start Ollama server and pull model if needed."""
        version_result = subprocess.run(
            ["ollama", "--version"], capture_output=True, text=True,
        )
        ollama_version = version_result.stdout.strip() or "unknown"

        print(f"  Ollama version: {ollama_version}")

        os.environ["OLLAMA_MODELS"] = MODEL_DIR

        self.ollama_process = subprocess.Popen(["ollama", "serve"])
        print(f"Ollama server PID: {self.ollama_process.pid}")

        for _ in range(30):
            try:
                result = subprocess.run(
                    ["ollama", "list"], capture_output=True, text=True, timeout=5,
                )
                if result.returncode == 0:
                    break
            except Exception:
                pass
            await asyncio.sleep(1)

        print(f"Pulling model: {MODEL_NAME}")
        pull_result = subprocess.run(
            [
                "ollama", "pull", MODEL_NAME,
            ],
            capture_output=True,
            text=True,
        )
        final_list = subprocess.run(["ollama", "list"], capture_output=True, text=True)
        print(f"  Models available: {final_list.stdout}")

        if pull_result.returncode != 0:
            raise RuntimeError(
                f"Failed to pull {MODEL_NAME} and no cached version: {pull_result.stderr.decode() if isinstance(pull_result.stderr, bytes) else pull_result.stderr}"
            )

        model_volume.commit()
        print(f"Model {MODEL_NAME} ready.")

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
