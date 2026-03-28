"""
Modal Deployment Audit Script — vex-qlora-train
================================================
Research only — does NOT modify anything.

Checks:
1. Keys present in the "model" Modal secret (masked)
2. GPU type configured for QLoRATrainer / train_all_kernels
3. Whether Qwen/Qwen3.5-35B-A3B exists in /models/hub/ on vex-models volume
4. Stale adapter/checkpoint directories under /models/
5. Top-level vex-models volume contents and /models/hub/ subdirectories

Run:
    modal run modal/audit_deployment.py
"""

import os
from pathlib import Path

import modal

app = modal.App("vex-qlora-audit")

model_volume = modal.Volume.from_name("vex-models", create_if_missing=False)

# Minimal image — no CUDA compile needed, just Python + basic tools
audit_image = modal.Image.debian_slim(python_version="3.11")


# ── 1. Secret key audit ──────────────────────────────────────────────────────────


@app.function(secrets=[modal.Secret.from_name("model")])
def audit_secret_keys():
    """Read all env vars injected by the 'model' secret; report SET/NOT SET only."""
    print("\n" + "=" * 60)
    print("SECRET AUDIT: 'model'")
    print("=" * 60)

    # Modal injects secret keys as environment variables.
    # We can't enumerate which vars came from the secret vs the image,
    # but we can report ALL env vars that look like they belong to the
    # 'model' secret by checking the known key names.
    known_secret_keys = [
        "HARVEST_MODEL_ID",
        "HF_TOKEN",
        "HUGGING_FACE_HUB_TOKEN",
        "TRAIN_GPU",
        "KERNEL_API_KEY",
        "KERNEL_CALLBACK_URL",
        "WANDB_API_KEY",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
    ]

    found = {}
    for key in known_secret_keys:
        val = os.environ.get(key)
        if val is None:
            found[key] = "NOT SET"
        elif val == "":
            found[key] = "SET (empty string)"
        else:
            found[key] = "SET"

    print(f"\n{'Key':<30} {'Status'}")
    print("-" * 55)
    for k, v in found.items():
        print(f"  {k:<28} {v}")

    # Also dump ALL env var names (no values) so we catch any non-standard keys
    print("\n  All env var NAMES in container (no values shown):")
    all_keys = sorted(os.environ.keys())
    for k in all_keys:
        marker = "  <-- from secret (known)" if k in known_secret_keys else ""
        print(f"    {k}{marker}")

    return found


# ── 2. GPU type in deployed script ─────────────────────────────────────────


@app.function()
def audit_gpu_config():
    """Parse vex_qlora_train.py to report the baked-in GPU type."""
    print("\n" + "=" * 60)
    print("GPU CONFIGURATION AUDIT")
    print("=" * 60)

    # The GPU type is baked in at deploy time via TRAIN_GPU env var
    # (evaluated locally, NOT from Modal secret).
    # We can inspect the deployed source to see what was baked in,
    # but the most reliable approach is to look at the local source file.

    note = (
        "NOTE: TRAIN_GPU is evaluated at LOCAL deploy time, not from Modal secrets.\n"
        "The value below reflects what TRAIN_GPU was set to when `modal deploy` was last run.\n"
        "If TRAIN_GPU was not set, it defaults to 'a100-80gb' (updated from a10g).\n"
    )
    print(note)

    train_gpu_env = os.environ.get("TRAIN_GPU", "<not in this container env>")
    print(f"  TRAIN_GPU in this container: {train_gpu_env}")
    print(
        "\n  For deployed QLoRATrainer / train_all_kernels, read the source directly:\n"
        "  @app.cls(gpu=TRAIN_GPU, ...) — TRAIN_GPU default is 'a100-80gb'\n"
        "  35B-A3B requires a100-80gb minimum for training."
    )

    return {"train_gpu_env": train_gpu_env}


# ── 3 + 4 + 5. Volume content audit ───────────────────────────────────────


@app.function(
    image=audit_image,
    volumes={"/models": model_volume},
    timeout=300,
)
def audit_volume():
    """Inspect vex-models volume: model cache, adapters, checkpoints."""

    print("\n" + "=" * 60)
    print("VOLUME AUDIT: vex-models")
    print("=" * 60)

    models_root = Path("/models")

    # ── 5a. Top-level contents ──
    print("\n[5] TOP-LEVEL /models/ CONTENTS:")
    print("-" * 40)
    try:
        top_items = sorted(models_root.iterdir())
        for item in top_items:
            kind = "DIR " if item.is_dir() else "FILE"
            try:
                size = sum(f.stat().st_size for f in item.rglob("*") if f.is_file())
                size_str = f"{size / 1e9:.2f} GB" if size >= 1e9 else f"{size / 1e6:.1f} MB"
            except Exception:
                size_str = "?"
            print(f"  {kind}  {item.name:<40} {size_str}")
    except Exception as e:
        print(f"  ERROR listing /models/: {e}")

    # ── 5b. /models/hub/ subdirectories ──
    hub_path = models_root / "hub"
    print("\n[5b] /models/hub/ SUBDIRECTORIES:")
    print("-" * 40)
    if hub_path.exists():
        hub_items = sorted(hub_path.iterdir())
        if hub_items:
            for item in hub_items:
                kind = "DIR " if item.is_dir() else "FILE"
                try:
                    files = list(item.rglob("*"))
                    file_count = sum(1 for f in files if f.is_file())
                    size = sum(f.stat().st_size for f in files if f.is_file())
                    size_str = f"{size / 1e9:.2f} GB" if size >= 1e9 else f"{size / 1e6:.1f} MB"
                except Exception:
                    file_count = 0
                    size_str = "?"
                print(f"  {kind}  {item.name:<50} {file_count:>5} files  {size_str}")
        else:
            print("  (empty)")
    else:
        print("  /models/hub/ does NOT EXIST")

    # ── 3. Qwen3.5-35B-A3B cache check ──
    target_model = "Qwen/Qwen3.5-35B-A3B"
    cache_name = "models--" + target_model.replace("/", "--")
    model_cache_path = hub_path / cache_name
    print(f"\n[3] MODEL CACHE CHECK: {target_model}")
    print("-" * 40)
    if model_cache_path.exists():
        # Check for actual weight files
        safetensors = list(model_cache_path.rglob("*.safetensors"))
        bins = list(model_cache_path.rglob("*.bin"))
        total_size = sum(f.stat().st_size for f in safetensors + bins if f.is_file())
        print(f"  Cache directory EXISTS: {model_cache_path}")
        print(f"  .safetensors files: {len(safetensors)}")
        print(f"  .bin files:         {len(bins)}")
        print(f"  Total weight size:  {total_size / 1e9:.2f} GB")
        if safetensors or bins:
            print("  STATUS: WEIGHTS PRESENT — model is cached")
        else:
            print(
                "  STATUS: WARNING — directory exists but NO weight files found (incomplete download?)"
            )

        # List snapshots
        snapshots_dir = model_cache_path / "snapshots"
        if snapshots_dir.exists():
            snaps = sorted(snapshots_dir.iterdir())
            print(f"\n  Snapshots ({len(snaps)}):")
            for s in snaps:
                sfiles = list(s.rglob("*"))
                scount = sum(1 for f in sfiles if f.is_file())
                print(f"    {s.name}  ({scount} files)")
        else:
            print("  No snapshots/ subdirectory found.")
    else:
        print(f"  Cache directory NOT FOUND: {model_cache_path}")
        print("  STATUS: MODEL NOT CACHED — run download_model first")
        # Check for 4B as fallback
        cache_4b = hub_path / "models--Qwen--Qwen3.5-4B"
        if cache_4b.exists():
            safetensors_4b = list(cache_4b.rglob("*.safetensors"))
            print(f"\n  NOTE: Qwen3.5-4B cache found at {cache_4b}")
            print(f"        ({len(safetensors_4b)} safetensors files)")

    # ── 4. Stale adapters / checkpoints ──
    print("\n[4] ADAPTER / CHECKPOINT DIRECTORY AUDIT:")
    print("-" * 40)

    adapters_path = models_root / "adapters"
    merged_path = models_root / "merged"
    genesis_eggs_path = models_root / "genesis_eggs"
    checkpoints_extra = [
        models_root / "checkpoints",
        models_root / "tmp",
        models_root / "temp",
    ]

    def describe_dir(p: Path, label: str):
        if not p.exists():
            print(f"  {label}: NOT PRESENT")
            return
        subdirs = [d for d in p.iterdir() if d.is_dir()]
        files = [f for f in p.iterdir() if f.is_file()]
        print(f"\n  {label} ({p}):")
        print(f"    subdirs: {len(subdirs)}, top-level files: {len(files)}")
        for d in sorted(subdirs):
            # Look for checkpoint-specific files
            weight_files = list(d.rglob("*.safetensors")) + list(d.rglob("*.bin"))
            total_size = sum(f.stat().st_size for f in weight_files if f.is_file())
            size_str = (
                f"{total_size / 1e9:.2f} GB" if total_size >= 1e9 else f"{total_size / 1e6:.1f} MB"
            )
            has_config = any(d.rglob("adapter_config.json"))
            complete_marker = (
                "OK (adapter_config.json present)"
                if has_config
                else "WARNING: no adapter_config.json"
            )
            print(f"    + {d.name:<30} {size_str:>10}  {complete_marker}")

            # Check for in-progress checkpoint sub-dirs (checkpoint-NNN from SFTTrainer)
            trl_ckpts = [x for x in d.iterdir() if x.is_dir() and x.name.startswith("checkpoint-")]
            if trl_ckpts:
                for ckpt in sorted(trl_ckpts):
                    ckpt_size = sum(f.stat().st_size for f in ckpt.rglob("*") if f.is_file())
                    ckpt_str = (
                        f"{ckpt_size / 1e9:.2f} GB"
                        if ckpt_size >= 1e9
                        else f"{ckpt_size / 1e6:.1f} MB"
                    )
                    print(f"      checkpoint subdir: {ckpt.name}  ({ckpt_str}) — may be stale")

    describe_dir(adapters_path, "/models/adapters (per-kernel LoRA adapters)")
    describe_dir(merged_path, "/models/merged (merged Ollama fallback models)")
    describe_dir(genesis_eggs_path, "/models/genesis_eggs (Genesis Egg snapshots)")
    for p in checkpoints_extra:
        if p.exists():
            describe_dir(p, f"EXTRA: {p.name}")

    # Top-level stale files
    print("\n  Top-level /models/ loose files (could be noise):")
    try:
        loose = [f for f in models_root.iterdir() if f.is_file()]
        if loose:
            for f in loose:
                print(f"    FILE: {f.name}  ({f.stat().st_size} bytes)")
        else:
            print("    (none)")
    except Exception as e:
        print(f"    ERROR: {e}")

    print("\n" + "=" * 60)
    print("VOLUME AUDIT COMPLETE")
    print("=" * 60)

    return {"status": "done"}


# ── Local source audit (no Modal container needed) ───────────────────────────


def _local_source_audit():
    """Read the local vex_qlora_train.py to report baked-in GPU and secret config."""
    src = Path(__file__).parent / "vex_qlora_train.py"
    if not src.exists():
        print("  vex_qlora_train.py not found next to this script")
        return

    print("\n" + "=" * 60)
    print("LOCAL SOURCE AUDIT: vex_qlora_train.py")
    print("=" * 60)

    lines = src.read_text().splitlines()
    relevant = [
        "TRAIN_GPU",
        "HARVEST_MODEL_ID",
        "gpu=TRAIN_GPU",
        'Secret.from_name("model")',
        "modal.Secret",
        "modal.Volume",
        "@app.cls",
        "@app.function",
    ]
    print("\n  Lines containing GPU / secret config:")
    for i, line in enumerate(lines, 1):
        if any(r in line for r in relevant):
            print(f"  L{i:>4}: {line.rstrip()}")


@app.local_entrypoint()
def main():
    print("\n" + "#" * 60)
    print("# VEX-QLORA-TRAIN DEPLOYMENT AUDIT")
    print("# Research only — no modifications made")
    print("#" * 60)

    # ── Local source audit (instant, no Modal spin-up) ──
    _local_source_audit()

    # ── GPU config from source (evaluated locally) ──
    print("\n" + "=" * 60)
    print("[2] GPU TYPE ANALYSIS")
    print("=" * 60)
    train_gpu_local = os.environ.get("TRAIN_GPU", "a100-80gb (DEFAULT — TRAIN_GPU not set in local env)")
    harvest_model_local = os.environ.get("HARVEST_MODEL_ID", "Qwen/Qwen3.5-4B (DEFAULT — not set)")
    print(f"\n  Local env TRAIN_GPU:       {train_gpu_local}")
    print(f"  Local env HARVEST_MODEL_ID:{harvest_model_local}")
    print(
        "\n  CRITICAL NOTE: TRAIN_GPU is evaluated at deploy time from LOCAL env.\n"
        "  If TRAIN_GPU was not set when 'modal deploy' was last run, the deployed\n"
        "  QLoRATrainer.gpu and train_all_kernels.gpu are BOTH 'a100-80gb'.\n"
        "  Qwen3.5-35B-A3B requires 'a100' (40 GB) minimum for QLoRA — 'a10g' WILL OOM.\n"
        "  Verify deployed GPU: modal app info vex-qlora-train (or check Modal dashboard)"
    )

    # ── Secret key audit (spins up a Modal container) ──
    print("\n[1] Running secret key audit on Modal...")
    audit_secret_keys.remote()

    # ── Volume audit ──
    print("\n[3-5] Running volume audit on Modal...")
    audit_volume.remote()

    print("\n" + "#" * 60)
    print("# AUDIT COMPLETE")
    print("#" * 60)
