"""
Send corpus through Modal /coordize to get Δ⁶³ basin coordinates.

Splits corpus into batches and sends to the 35B model on A100.
Saves raw harvest results + compressed basin bank.
"""

import json
import os
import sys
import time

import numpy as np
import requests

MODAL_URL = "https://archelon--vex-coordizer-harvest-coordizerharvester-web.modal.run"
API_KEY = os.environ.get("KERNEL_API_KEY", "")
CORPUS_PATH = "scripts/harvest_corpus.json"
OUTPUT_DIR = "harvest_data"
BATCH_SIZE = 50  # texts per request (keep reasonable for 35B model)
MIN_CONTEXTS = 1  # accept tokens seen even once (we want broad coverage)


def main():
    with open(CORPUS_PATH) as f:
        all_texts = json.load(f)

    print(f"Corpus: {len(all_texts)} texts")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Split into batches
    batches = [all_texts[i : i + BATCH_SIZE] for i in range(0, len(all_texts), BATCH_SIZE)]
    print(f"Batches: {len(batches)} × {BATCH_SIZE}")

    all_results = []
    total_elapsed = 0

    for i, batch in enumerate(batches):
        print(f"\n--- Batch {i + 1}/{len(batches)} ({len(batch)} texts) ---")
        payload = {
            "texts": batch,
            "batch_size": 16,
            "max_length": 512,
            "min_contexts": MIN_CONTEXTS,
            "target_resonances": 0,
            "lens_dim": 32,
            "compute_curvature": True,
        }
        if API_KEY:
            payload["_api_key"] = API_KEY

        headers = {"Content-Type": "application/json"}
        if API_KEY:
            headers["X-Api-Key"] = API_KEY

        t0 = time.time()
        try:
            resp = requests.post(
                f"{MODAL_URL}/coordize",
                json=payload,
                headers=headers,
                timeout=300,
            )
            elapsed = time.time() - t0
            total_elapsed += elapsed

            if resp.status_code != 200:
                print(f"  ERROR {resp.status_code}: {resp.text[:500]}")
                continue

            result = resp.json()
            if not result.get("success"):
                print(f"  FAILED: {result.get('error', 'unknown')}")
                continue

            basin = result.get("basin_coords", [])
            lens = result.get("lens_coords", [])
            meta = result.get("harvest_meta", {})
            curvature = result.get("curvature", {})

            print(f"  OK in {elapsed:.1f}s — basin={len(basin)}D, lens={len(lens)}D")
            print(
                f"  harvest: {meta.get('n_resonances_harvested', '?')} resonances from {meta.get('total_tokens_processed', '?')} tokens"
            )
            if curvature:
                print(
                    f"  curvature: mean={curvature.get('mean', '?'):.4f}, high={curvature.get('high_curvature_count', '?')}, low={curvature.get('low_curvature_count', '?')}"
                )

            all_results.append(
                {
                    "batch_idx": i,
                    "basin_coords": basin,
                    "lens_coords": lens,
                    "eigenvalues": result.get("eigenvalues", []),
                    "harvest_meta": meta,
                    "curvature": curvature,
                    "elapsed_seconds": elapsed,
                    "pga_dim": result.get("pga_dim"),
                }
            )

        except requests.exceptions.Timeout:
            print("  TIMEOUT after 300s")
            continue
        except Exception as e:
            print(f"  ERROR: {e}")
            continue

    print("\n=== HARVEST COMPLETE ===")
    print(f"Successful batches: {len(all_results)}/{len(batches)}")
    print(f"Total time: {total_elapsed:.1f}s")

    if not all_results:
        print("No results — nothing to save.")
        sys.exit(1)

    # Save raw results
    with open(f"{OUTPUT_DIR}/coordize_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Saved raw results to {OUTPUT_DIR}/coordize_results.json")

    # Aggregate basin coordinates across batches (average on simplex via sqrt-space)
    basin_arrays = [
        np.array(r["basin_coords"]) for r in all_results if len(r["basin_coords"]) == 64
    ]
    if basin_arrays:
        # Fréchet mean approximation in sqrt-space
        sqrt_basins = [np.sqrt(np.maximum(b, 1e-30)) for b in basin_arrays]
        mean_sqrt = np.mean(sqrt_basins, axis=0)
        mean_basin = mean_sqrt**2
        mean_basin = mean_basin / mean_basin.sum()  # normalize to simplex

        np.save(f"{OUTPUT_DIR}/aggregate_basin.npy", mean_basin)
        print(
            f"Saved aggregate basin ({len(basin_arrays)} batches averaged) to {OUTPUT_DIR}/aggregate_basin.npy"
        )
        print(f"Basin entropy: {-np.sum(mean_basin * np.log(mean_basin + 1e-30)):.4f}")
        print(f"Basin max: {mean_basin.max():.6f}, min: {mean_basin.min():.6f}")

    # Save all eigenvalues for analysis
    all_eigs = [r["eigenvalues"] for r in all_results if r["eigenvalues"]]
    if all_eigs:
        np.save(f"{OUTPUT_DIR}/eigenvalues.npy", np.array(all_eigs))
        print(f"Saved eigenvalues ({len(all_eigs)} sets) to {OUTPUT_DIR}/eigenvalues.npy")

    # Save harvest metadata summary
    summary = {
        "total_batches": len(all_results),
        "total_texts": len(all_texts),
        "total_elapsed_seconds": total_elapsed,
        "model_id": "Qwen/Qwen3.5-35B-A3B",
        "basin_dim": 64,
        "lens_dim": 32,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "curvature_stats": [r["curvature"] for r in all_results if r.get("curvature")],
    }
    with open(f"{OUTPUT_DIR}/harvest_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary to {OUTPUT_DIR}/harvest_summary.json")


if __name__ == "__main__":
    main()
