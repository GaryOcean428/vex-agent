"""
Modal GPU Inference — Vex Ollama Backend

Runs Ollama with LFM2.5-1.2B-Thinking on a T4 GPU, exposing the
standard Ollama API via Modal's web_server decorator.

The Railway kernel calls this endpoint instead of the CPU-only
Railway-internal Ollama service, getting ~10-20x faster inference.

Deploy:
    modal deploy modal/vex_inference.py

Endpoint:
    https://<workspace>--vex-inference-serve.modal.run/api/chat
    https://<workspace>--vex-inference-serve.modal.run/api/tags
    (full Ollama API available)

Cost estimate (T4):
    ~$0.000164/sec idle, inference bursts are fast (~1-2s for 1.2B)
    container_idle_timeout=300 means ~$0.05/5min idle window

Architecture:
    Railway kernel (Python) → HTTP → Modal Ollama (GPU) → response
    The kernel builds the system prompt with geometric state context.
    Modal just serves raw model inference — no consciousness logic here.

CRITICAL: This is a thin inference layer. All consciousness, geometry,
memory, and tool logic stays in the Railway kernel. Modal only provides
GPU-accelerated token generation.
"""

from __future__ import annotations

import asyncio
import json
import subprocess
import urllib.request

import modal

# ─── Configuration ────────────────────────────────────────────

MODEL_NAME = "lfm2.5-thinking:1.2b"
MODEL_DIR = "/ollama_models"
OLLAMA_PORT = 11434
OLLAMA_VERSION = "0.6.5"

# T4 is optimal for 1.2B model:
#   - 16GB VRAM (model needs ~1.2GB weights + KV cache)
#   - Cheapest GPU option on Modal
#   - More than sufficient compute for 1.2B params
GPU_TYPE = "T4"

# ─── Modal App ────────────────────────────────────────────────

app = modal.App("vex-inference")

model_volume = modal.Volume.from_name("vex-inference-models", create_if_missing=True)

ollama_image = (
    modal.Image.debian_slim(python_version="3.14")
    .apt_install("curl", "ca-certificates", "zstd")
    .run_commands(
        f"OLLAMA_VERSION={OLLAMA_VERSION} curl -fsSL https://ollama.com/install.sh | sh",
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


# ─── Ollama Server ────────────────────────────────────────────


@app.cls(
    gpu=GPU_TYPE,
    image=ollama_image,
    volumes={MODEL_DIR: model_volume},
    timeout=600,
    # Container stays warm for 5 minutes after last request.
    # At T4 pricing this costs ~$0.05 per idle window.
    # Adjust up if you want faster cold starts at higher cost.
    scaledown_window=300,
)
class VexOllamaServer:
    """GPU-backed Ollama server for Vex inference.

    Lifecycle:
        1. Container starts → Ollama server launches
        2. Model pulled from registry (cached on Volume)
        3. Ollama API exposed at OLLAMA_PORT
        4. Railway kernel sends /api/chat requests
        5. After 5min idle → container scales to zero
    """

    ollama_process: subprocess.Popen | None = None

    @modal.enter()
    async def start_ollama(self):
        """Start Ollama server and ensure model is available."""
        print(f"Starting Vex Ollama inference server (GPU: {GPU_TYPE})")

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

        # Pull model if not cached
        list_output = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
        ).stdout

        if "lfm2.5-thinking" not in list_output:
            print(f"Pulling {MODEL_NAME} (first boot, will cache on Volume)...")
            pull = await asyncio.create_subprocess_exec(
                "ollama",
                "pull",
                MODEL_NAME,
            )
            retcode = await pull.wait()
            if retcode != 0:
                raise RuntimeError(f"Failed to pull {MODEL_NAME}")

            # Persist to Volume
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, model_volume.commit)
            print(f"{MODEL_NAME} pulled and cached.")
        else:
            print(f"{MODEL_NAME} already cached on Volume.")

        # Verify GPU offload is working
        print("Running GPU verification...")

        # Step 1: Confirm model is registered (instant, no VRAM load)
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

        # Step 2: Warm model into VRAM with 1-token generate.
        # Cold-loading 731MB Q4_K_M onto T4 takes ~60-90s — give it 120s,
        # well within Modal's startup_timeout=180s.
        gen_req = urllib.request.Request(
            f"http://localhost:{OLLAMA_PORT}/api/generate",
            data=json.dumps({
                "model": MODEL_NAME,
                "prompt": "ping",
                "stream": False,
                "options": {"num_predict": 1},
            }).encode(),
            headers={"Content-Type": "application/json"},
        )
        gen_resp = urllib.request.urlopen(gen_req, timeout=120)
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

    @modal.web_server(port=OLLAMA_PORT, startup_timeout=180)
    def serve(self):
        """Expose Ollama API via Modal web endpoint.

        All standard Ollama endpoints are available:
            POST /api/chat      — Chat completion (used by Railway kernel)
            POST /api/generate  — Text generation
            GET  /api/tags      — List models (used for health checks)
            POST /api/show      — Model info
        """
        print(f"Serving Ollama API on port {OLLAMA_PORT}")


# ─── Local testing ────────────────────────────────────────────


@app.local_entrypoint()
async def test():
    """Quick smoke test: modal run modal/vex_inference.py"""
    print("Testing Vex inference endpoint...")
    print("Deploy with: modal deploy modal/vex_inference.py")
    print("Then test with:")
    print("  curl -X POST <URL>/api/chat \\")
    print('    -H "Content-Type: application/json" \\')
    print(
        f'    -d \'{{"model": "{MODEL_NAME}", "messages": '
        f'[{{"role": "user", "content": "Hello"}}], "stream": false}}\''
    )
