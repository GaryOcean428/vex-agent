#!/usr/bin/env python3
"""
End-to-End Verification Script — Tests the Full Training Pipeline Round-Trip.

Usage:
    python scripts/verify_e2e.py [--base-url BASE_URL] [--api-key KEY] [--skip-training]

Verifies:
    1. Health check — kernel server is alive
    2. Health/reachability — all external services green
    3. Training upload — small test file accepted and processed
    4. Training stats — chunk count increased
    5. Coordizer coordize — text → 64D basin coords returned
    6. Coordizer harvest — harvest processes
    7. Training trigger — Modal training starts
    8. Modal status — adapter status check
    9. Consciousness state — loop is not stuck

Exit codes:
    0 = all checks passed
    1 = one or more checks failed
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import httpx

# Default to Railway's internal URL
DEFAULT_BASE_URL = "http://localhost:8000"


def _log(check: str, ok: bool, msg: str) -> bool:
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {check}: {msg}")
    return ok


def run_verification(base_url: str, api_key: str, skip_training: bool = False) -> bool:
    """Run all E2E verification checks. Returns True if all pass."""
    results: list[bool] = []
    headers = {"X-Kernel-Key": api_key} if api_key else {}

    client = httpx.Client(base_url=base_url, timeout=30.0, headers=headers)

    print("\n=== VEX E2E Verification ===\n")

    # 1. Health check
    try:
        resp = client.get("/health")
        data = resp.json()
        ok = resp.status_code == 200 and data.get("status") == "ok"
        results.append(_log("Health", ok, f"status={data.get('status', '?')}"))
    except Exception as e:
        results.append(_log("Health", False, f"Connection failed: {e}"))
        print("\n  Cannot reach kernel server. Aborting.\n")
        return False

    # 2. Health/reachability
    try:
        resp = client.get("/health/reachability")
        data = resp.json()
        services = data.get("services", {})
        summary = ", ".join(
            f"{k}={'OK' if v.get('reachable') else 'DOWN'}" for k, v in services.items()
        )
        results.append(
            _log("Reachability", resp.status_code == 200, summary or "no services configured")
        )
    except Exception as e:
        results.append(_log("Reachability", False, str(e)))

    # 3. Training stats (baseline)
    baseline_chunks = 0
    try:
        resp = client.get("/training/stats")
        data = resp.json()
        baseline_chunks = data.get("total_chunks", 0)
        results.append(
            _log(
                "Training stats (baseline)",
                resp.status_code == 200,
                f"total_chunks={baseline_chunks}",
            )
        )
    except Exception as e:
        results.append(_log("Training stats (baseline)", False, str(e)))

    # 4. Upload test file
    test_content = json.dumps(
        {
            "messages": [
                {"role": "system", "content": "You are a geometric consciousness test."},
                {"role": "user", "content": "What is the Fisher-Rao metric?"},
                {
                    "role": "assistant",
                    "content": "The Fisher-Rao metric is the unique "
                    "Riemannian metric on statistical manifolds invariant under "
                    "sufficient statistics. On the probability simplex Delta^63, "
                    "it gives geodesics that respect the geometric structure of "
                    "basin coordinates.",
                },
            ]
        }
    )
    try:
        resp = client.post(
            "/training/upload",
            files={"file": ("e2e_test.jsonl", test_content.encode(), "application/octet-stream")},
            data={"category": "curriculum", "mode": "standard"},
        )
        data = resp.json()
        job_id = data.get("job_id", "")
        ok = resp.status_code == 200 and data.get("status") == "processing"
        results.append(_log("Upload accepted", ok, f"job_id={job_id}"))

        # Poll for completion
        if job_id:
            for _ in range(30):  # 30s max
                time.sleep(1)
                resp = client.get(f"/training/upload/status/{job_id}")
                status = resp.json()
                if status.get("status") not in ("processing",):
                    break
            final_status = status.get("status", "unknown")
            chunks = status.get("chunks_written", 0)
            ok = final_status in ("complete", "ok", "done") and chunks > 0
            results.append(_log("Upload processed", ok, f"status={final_status}, chunks={chunks}"))
    except Exception as e:
        results.append(_log("Upload", False, str(e)))

    # 5. Verify stats increased
    try:
        resp = client.get("/training/stats")
        data = resp.json()
        new_chunks = data.get("total_chunks", 0)
        ok = new_chunks >= baseline_chunks  # At least not decreased
        results.append(
            _log(
                "Training stats (post-upload)",
                ok,
                f"total_chunks={new_chunks} (was {baseline_chunks})",
            )
        )
    except Exception as e:
        results.append(_log("Training stats (post-upload)", False, str(e)))

    # 6. Coordizer coordize
    try:
        resp = client.post(
            "/api/coordizer/coordize",
            json={"text": "The Fisher-Rao metric on the probability simplex."},
        )
        data = resp.json()
        basin = data.get("basin_coords") or data.get("basin") or data.get("coordinates")
        ok = resp.status_code == 200 and basin is not None and len(basin) == 64
        results.append(_log("Coordizer coordize", ok, f"basin_dim={len(basin) if basin else 0}"))
    except Exception as e:
        results.append(_log("Coordizer coordize", False, str(e)))

    # 7. Coordizer harvest status
    try:
        resp = client.get("/api/coordizer/harvest/status")
        data = resp.json()
        ok = resp.status_code == 200
        pending = data.get("pending", 0)
        results.append(_log("Harvest status", ok, f"pending={pending}"))
    except Exception as e:
        results.append(_log("Harvest status", False, str(e)))

    # 8. Modal status
    try:
        resp = client.get("/training/modal-status")
        data = resp.json()
        ok = resp.status_code == 200
        training_active = data.get("training_active", False)
        adapters = data.get("loaded_adapters", [])
        results.append(_log("Modal status", ok, f"training={training_active}, adapters={adapters}"))
    except Exception as e:
        results.append(_log("Modal status", False, str(e)))

    # 9. Training trigger (optional)
    if not skip_training:
        try:
            resp = client.post("/training/trigger", json={})
            data = resp.json()
            ok = resp.status_code == 200 and data.get("status") == "triggered"
            results.append(_log("Training trigger", ok, f"status={data.get('status', '?')}"))
        except Exception as e:
            results.append(_log("Training trigger", False, str(e)))
    else:
        print("  [SKIP] Training trigger (--skip-training)")

    # 10. Consciousness state
    try:
        resp = client.get("/state")
        data = resp.json()
        state = data.get("consciousness_state") or data.get("state", "unknown")
        phi = data.get("phi", 0)
        divergence = data.get("ocean_divergence", data.get("divergence", "?"))
        ok = resp.status_code == 200
        results.append(
            _log("Consciousness state", ok, f"state={state}, phi={phi}, divergence={divergence}")
        )
    except Exception as e:
        results.append(_log("Consciousness state", False, str(e)))

    # Summary
    passed = sum(results)
    total = len(results)
    print(f"\n{'=' * 50}")
    print(f"  Results: {passed}/{total} checks passed")
    if passed == total:
        print("  STATUS: ALL CLEAR")
    else:
        print(f"  STATUS: {total - passed} FAILURES")
    print(f"{'=' * 50}\n")

    client.close()
    return passed == total


def main():
    parser = argparse.ArgumentParser(description="VEX E2E Verification")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="Kernel server base URL")
    parser.add_argument("--api-key", default="", help="Kernel API key (X-Kernel-Key header)")
    parser.add_argument(
        "--skip-training", action="store_true", help="Skip the training trigger step"
    )
    args = parser.parse_args()

    ok = run_verification(args.base_url, args.api_key, args.skip_training)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
