"""
Modal GPU Inference — Vex Ollama Backend

Runs a configurable model (default: GLM-4.7-Flash 30B-A3B MoE) on
A10G GPU, exposing the standard Ollama API via Modal web_server.

Always installs the latest Ollama version (no version pinning).
Always pulls the model on startup (idempotent — checks digest,
downloads only if a newer version is available).

Deploy:
    modal deploy modal/vex_inference.py

Endpoint:
    https://<workspace>--vex-inference-serve.modal.run/api/chat
    https://<workspace>--vex-inference-serve.modal.run/api/tags
    (full Ollama API available)

Cost estimate (A10G):
    ~$0.76/hr active, scaledown_window=300 means ~$0.063/5min idle
    First deploy pulls ~19GB model (one-time, cached on Volume)

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
# Model is configurable via the VEX_MODAL_MODEL env var (e.g., Modal secret),
# with "glm-4.7-flash" as the hardcoded default if not set.
MODEL_NAME = os.environ.get("VEX_MODAL_MODEL", "glm-4.7-flash")
MODEL_DIR = "/ollama_models"
OLLAMA_PORT = 11434

# A10G: 24GB VRAM. GLM-4.7-Flash Q4_K_M = 19GB. Fits with ~5GB headroom.
# Qwen3:30b Q4_K_M = 19GB also fits.
GPU_TYPE = "A10G"

# --- Modal App ------------------------------------------------------------

app = modal.App("vex-inference")

model_volume = modal.Volume.from_name("vex-inference-models", create_if_missing=True)

# Always install latest Ollama — no version pinning.
# This ensures we get the latest model support, bug fixes, and
# performance improvements on every image rebuild.
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
            # Flash attention for faster inference
            "OLLAMA_FLASH_ATTENTION": "true",
            # Single-user serving (Railway kernel is the only client)
            "OLLAMA_NUM_PARALLEL": "1",
        }
    )
)


# --- Ollama Server --------------------------------------------------------


@app.cls(
    gpu=GPU_TYPE,
    image=ollama_image,
    volumes={MODEL_DIR: model_volume},
    # First boot pulls ~19GB model; subsequent boots use cached volume (~90s)
    timeout=900,
    # Container stays warm for 5 minutes after last request.
    # At A10G pricing this costs ~$0.063 per idle window.
    scaledown_window=300,
    secrets=[modal.Secret.from_name("model")],
)
class VexOllamaServer:
    """GPU-backed Ollama server for Vex inference.

    Lifecycle:
        1. Container starts -> Ollama server launches (latest version)
        2. Model pulled from registry (always pulled to get latest digest)
        3. Model cached on Volume for fast subsequent starts
        4. Ollama API exposed at OLLAMA_PORT
        5. Railway kernel sends /api/chat requests
        6. After 5min idle -> container scales to zero
    """

    ollama_process: subprocess.Popen | None = None

    @modal.enter()
    async def start_ollama(self):
        """Start Ollama server and ensure model is available."""
        # Log Ollama version for debugging
        version_result = subprocess.run(
            ["ollama", "--version"],
            capture_output=True,
            text=True,
        )
        ollama_version = version_result.stdout.strip() or "unknown"
        print("Starting Vex Ollama inference server")
        print(f"  Ollama version: {ollama_version}")
        print(f"  GPU: {GPU_TYPE}")
        print(f"  Model: {MODEL_NAME}")

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

        # Always pull model — idempotent, checks digest, downloads
        # only changed layers if a newer version is available.
        # This ensures every container start gets the latest model.
        print(f"Pulling {MODEL_NAME} (checking for updates)...")
        pull = await asyncio.create_subprocess_exec(
            "ollama",
            "pull",
            MODEL_NAME,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await pull.communicate()
        if pull.returncode != 0:
            # Pull failed — check if cached version exists
            list_output = subprocess.run(
                ["ollama", "list"],
                capture_output=True,
                text=True,
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

        # Persist to Volume
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, model_volume.commit)

        # Verify model is registered
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

        # Warm model into VRAM with 1-token generate.
        # 19GB model onto A10G (24GB VRAM) takes ~90-120s on cold start.
        print(f"Warming {MODEL_NAME} into GPU VRAM...")
        gen_req = urllib.request.Request(
            f"http://localhost:{OLLAMA_PORT}/api/generate",
            data=json.dumps(
                {
                    "model": MODEL_NAME,
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

    # First boot pulls ~19GB model; subsequent boots use cached volume (~90s)
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


# --- Local testing --------------------------------------------------------


@app.local_entrypoint()
async def test():
    """Quick smoke test: modal run modal/vex_inference.py"""
    print("Testing Vex inference endpoint...")
    print(f"Model: {MODEL_NAME}")
    print(f"GPU: {GPU_TYPE}")
    print("Deploy with: modal deploy modal/vex_inference.py")
    print("Then test with:")
    print("  curl -X POST <URL>/api/chat \\")
    print('    -H "Content-Type: application/json" \\')
    print(
        f'    -d \'{{"model": "{MODEL_NAME}", "messages": '
        f'[{{"role": "user", "content": "Hello"}}], "stream": false}}\''
    )
