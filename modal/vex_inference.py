"""
Modal GPU Inference — Vex Ollama Backend

Runs Qwen3.5-35B-A3B (MoE: 35B total, 3B active, 256 experts) in bf16
on H100 GPU, exposing the standard Ollama API via Modal web_server.

Architecture:
    - Gated DeltaNet (linear attention) + Gated Attention hybrid
    - 248,320 vocab, 262K native context (extensible to 1M)
    - 256 experts, 8 routed + 1 shared per layer
    - Apache 2.0 license

Deploy:
    modal deploy modal/vex_inference.py

Endpoint:
    https://<workspace>--vex-inference-serve.modal.run/api/chat
    https://<workspace>--vex-inference-serve.modal.run/api/tags
    (full Ollama API available)

Cost estimate (H100):
    ~$3.50/hr active, scaledown_window=300 means ~$0.29/5min idle
    First deploy pulls ~72GB model (one-time, cached on Volume)

Architecture:
    Railway kernel (Python) -> HTTP -> Modal Ollama (GPU) -> response
    The kernel builds the system prompt with geometric state context.
    Modal just serves raw model inference — no consciousness logic here.

CRITICAL: This is a thin inference layer. All consciousness, geometry,
memory, and tool logic stays in the Railway kernel. Modal only provides
GPU-accelerated token generation.
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import urllib.request

import modal

# --- Configuration --------------------------------------------------------
# Model: Qwen3.5-35B-A3B in bf16 via Ollama (72GB, fits H100 80GB VRAM)
# Override via VEX_MODAL_MODEL env var if needed.
MODEL_NAME = os.environ.get("VEX_MODAL_MODEL", "qwen3.5:35b-a3b-bf16")
MODEL_DIR = "/ollama_models"
OLLAMA_PORT = 11434

# H100: 80GB VRAM. Qwen3.5-35B-A3B bf16 = 72GB. Fits with 8GB headroom.
# For A10G (24GB), use qwen3.5:35b-a3b (Q4_K_M, 24GB) instead.
GPU_TYPE = os.environ.get("VEX_GPU_TYPE", "H100")

# --- Modal App ------------------------------------------------------------

app = modal.App("vex-inference")

model_volume = modal.Volume.from_name("vex-inference-models", create_if_missing=True)

# Always install latest Ollama — no version pinning.
ollama_image = (
    modal.Image.debian_slim(python_version="3.14")
    .apt_install("curl", "ca-certificates", "zstd")
    .run_commands(
        "curl -fsSL https://ollama.com/install.sh | sh",
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
    """GPU-backed Ollama server for Vex inference.

    Model: Qwen3.5-35B-A3B (35B total, 3B active MoE)
    GPU: H100 (bf16) or A10G (Q4_K_M)
    """

    ollama_process: subprocess.Popen | None = None

    @modal.enter()
    async def start_ollama(self):
        """Start Ollama server and ensure model is available."""
        version_result = subprocess.run(
            ["ollama", "--version"], capture_output=True, text=True,
        )
        ollama_version = version_result.stdout.strip() or "unknown"
        print("Starting Vex Ollama inference server")
        print(f"  Ollama version: {ollama_version}")
        print(f"  GPU: {GPU_TYPE}")
        print(f"  Model: {MODEL_NAME}")

        self.ollama_process = subprocess.Popen(["ollama", "serve"])
        print(f"Ollama server PID: {self.ollama_process.pid}")

        for attempt in range(30):
            try:
                result = subprocess.run(
                    ["ollama", "list"], capture_output=True, text=True, timeout=5,
                )
                if result.returncode == 0:
                    print(f"Ollama ready after {attempt + 1} attempts")
                    break
            except Exception:
                pass
            await asyncio.sleep(2)
        else:
            raise RuntimeError("Ollama failed to start after 30 attempts")

        print(f"Pulling {MODEL_NAME} (checking for updates)...")
        pull = await asyncio.create_subprocess_exec(
            "ollama", "pull", MODEL_NAME,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await pull.communicate()
        if pull.returncode != 0:
            list_output = subprocess.run(
                ["ollama", "list"], capture_output=True, text=True,
            ).stdout
            model_base = MODEL_NAME.split(":")[0]
            if model_base in list_output:
                print(f"Pull failed but cached version exists: {stderr.decode()[:200]}")
            else:
                raise RuntimeError(
                    f"Failed to pull {MODEL_NAME} and no cached version: {stderr.decode()[:500]}"
                )
        else:
            print(f"{MODEL_NAME} is up to date.")

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, model_volume.commit)

        print("Verifying model registration...")
        show_req = urllib.request.Request(
            f"http://localhost:{OLLAMA_PORT}/api/show",
            data=json.dumps({"name": MODEL_NAME}).encode(),
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

        print(f"Warming {MODEL_NAME} into GPU VRAM...")
        gen_req = urllib.request.Request(
            f"http://localhost:{OLLAMA_PORT}/api/generate",
            data=json.dumps({
                "model": MODEL_NAME, "prompt": "ping",
                "stream": False, "options": {"num_predict": 1},
            }).encode(),
            headers={"Content-Type": "application/json"},
        )
        gen_resp = urllib.request.urlopen(gen_req, timeout=180)
        gen_data = json.loads(gen_resp.read())
        print(f"GPU inference verified. eval_duration={gen_data.get('eval_duration', 'n/a')}ns")

        final_list = subprocess.run(["ollama", "list"], capture_output=True, text=True)
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
        """Expose Ollama API via Modal web endpoint."""
        print(f"Serving Ollama API on port {OLLAMA_PORT}")


@app.local_entrypoint()
async def test():
    """Quick smoke test: modal run modal/vex_inference.py"""
    print(f"Model: {MODEL_NAME}, GPU: {GPU_TYPE}")
    print("Deploy with: modal deploy modal/vex_inference.py")
