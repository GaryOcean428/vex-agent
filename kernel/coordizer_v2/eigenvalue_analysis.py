"""
Eigenvalue Analysis — Real Harvest Pipeline

Runs the PROPER pipeline:
    1. Call Modal GPU endpoint → harvest GLM-4.7-Flash distributions
    2. Compress via Fisher-Rao PGA: Δ^(V-1) → Δ⁶³
    3. Report eigenvalue spectrum → determines lens intermediate dim n

This answers the E8 hypothesis: do the top 8 principal geodesic
directions capture >87.7% of total geodesic variance?

Usage:
    # Real harvest via Modal (requires MODAL_ENABLED=true, MODAL_HARVEST_URL set)
    python -m kernel.coordizer_v2.eigenvalue_analysis

    # With custom corpus file (one text per line)
    python -m kernel.coordizer_v2.eigenvalue_analysis --corpus corpus.txt

    # With more tokens (slower, more accurate)
    python -m kernel.coordizer_v2.eigenvalue_analysis --target-tokens 5000

    # Synthetic fallback (for testing pipeline only — no real semantic structure)
    python -m kernel.coordizer_v2.eigenvalue_analysis --synthetic

    # Save harvest for reuse (avoids re-running GPU)
    python -m kernel.coordizer_v2.eigenvalue_analysis --save-harvest ./harvest_data

    # Load previous harvest (skip GPU, just recompute eigenvalues)
    python -m kernel.coordizer_v2.eigenvalue_analysis --load-harvest ./harvest_data

Environment:
    MODAL_ENABLED=true
    MODAL_HARVEST_URL=https://<your-modal-app>.modal.run
    KERNEL_API_KEY=<your-key>

Cost estimate:
    ~2000 tokens on H100: ~$0.03 (~10 seconds)
    ~5000 tokens on H100: ~$0.08 (~25 seconds)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("eigenvalue_analysis")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CoordizerV2 Eigenvalue Analysis")
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Use synthetic data (tests pipeline, not hypothesis)",
    )
    parser.add_argument(
        "--corpus",
        type=str,
        default=None,
        help="Path to corpus file (one text per line). Uses default corpus if not provided.",
    )
    parser.add_argument(
        "--target-tokens",
        type=int,
        default=2000,
        help="Target number of token positions to harvest (default: 2000)",
    )
    parser.add_argument(
        "--save-harvest",
        type=str,
        default=None,
        help="Save harvest result to this directory for reuse",
    )
    parser.add_argument(
        "--load-harvest",
        type=str,
        default=None,
        help="Load previous harvest from this directory (skip GPU)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="eigenvalue_report.json",
        help="Output path for eigenvalue report JSON",
    )
    return parser.parse_args()


async def run_modal_harvest(
    corpus_texts: list[str] | None = None,
    target_tokens: int = 2000,
) -> "HarvestResult":
    """Run real harvest via Modal GPU endpoint."""
    from kernel.coordizer_v2.modal_harvest import modal_harvest

    logger.info("Starting Modal GPU harvest (target_tokens=%d)...", target_tokens)
    result = await modal_harvest(
        target_tokens=target_tokens,
        corpus_texts=corpus_texts,
    )
    logger.info(
        "Harvest complete: %d fingerprints, vocab_size=%d, %.1fs",
        len(result.token_fingerprints),
        result.vocab_size,
        result.harvest_time_seconds,
    )
    return result


def run_synthetic_harvest() -> "HarvestResult":
    """Generate synthetic harvest (pipeline test only)."""
    from kernel.coordizer_v2.modal_integration import generate_synthetic_harvest_result

    logger.warning("Using SYNTHETIC data — tests pipeline only, NOT the E8 hypothesis")
    return generate_synthetic_harvest_result(vocab_size=32000, n_tokens=500)


def load_corpus(path: str) -> list[str]:
    """Load corpus from file (one text per line)."""
    texts = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                texts.append(line)
    logger.info("Loaded %d texts from %s", len(texts), path)
    return texts


def run_compression(harvest: "HarvestResult") -> "CompressionResult":
    """Run Fisher-Rao PGA compression and return full result."""
    from kernel.coordizer_v2.compress import compress
    from kernel.coordizer_v2.geometry import BASIN_DIM

    logger.info(
        "Compressing %d fingerprints: Delta^%d -> Delta^%d",
        len(harvest.token_fingerprints),
        harvest.vocab_size - 1,
        BASIN_DIM - 1,
    )
    result = compress(harvest, target_dim=BASIN_DIM)
    return result


def analyse_eigenvalues(result: "CompressionResult") -> dict:
    """Analyse eigenvalue spectrum and produce report."""
    from kernel.coordizer_v2.geometry import E8_RANK

    eigenvalues = result.eigenvalues
    if eigenvalues is None or len(eigenvalues) == 0:
        logger.error("No eigenvalues produced -- compression may have failed")
        return {"error": "No eigenvalues"}

    total = float(np.sum(eigenvalues))
    cumulative = np.cumsum(eigenvalues) / total if total > 1e-12 else np.zeros_like(eigenvalues)

    # E8 hypothesis score
    e8_score = result.e8_hypothesis_score()

    # Effective dimensionality at various thresholds
    def effective_dim(threshold: float) -> int:
        for i, c in enumerate(cumulative):
            if c >= threshold:
                return i + 1
        return len(cumulative)

    dim_90 = effective_dim(0.90)
    dim_95 = effective_dim(0.95)
    dim_99 = effective_dim(0.99)

    # Spectral gap
    spectral_gap = (
        float(eigenvalues[0] / eigenvalues[min(7, len(eigenvalues) - 1)])
        if len(eigenvalues) > 7
        else 0.0
    )

    # Decision gate
    if e8_score > 0.85:
        recommended_n = 8
        recommendation = "E8 hypothesis SUPPORTED -- use n=8 for lens intermediate dim"
    elif e8_score > 0.60:
        recommended_n = 16
        recommendation = "E8 hypothesis PARTIAL -- use n=16 (fine structure beyond rank-8)"
    else:
        recommended_n = 32
        recommendation = "E8 hypothesis NOT SUPPORTED -- variance broadly distributed, use n=32"

    report = {
        "e8_hypothesis_score": round(e8_score, 4),
        "recommendation": recommendation,
        "recommended_n": recommended_n,
        "effective_dim_90pct": dim_90,
        "effective_dim_95pct": dim_95,
        "effective_dim_99pct": dim_99,
        "spectral_gap_1_to_8": round(spectral_gap, 2),
        "total_geodesic_variance": round(total, 6),
        "n_tokens_compressed": result.n_tokens,
        "source_dim": result.source_dim,
        "target_dim": result.target_dim,
        "compression_time_seconds": round(result.compression_time_seconds, 1),
        "eigenvalue_spectrum_top_16": [round(float(v), 6) for v in eigenvalues[:16]],
        "cumulative_variance_top_16": [round(float(c), 4) for c in cumulative[:16]],
        "is_synthetic": False,
    }

    return report


def print_report(report: dict) -> None:
    """Print human-readable eigenvalue report."""
    print("\n" + "=" * 60)
    print("  COORDIZER EIGENVALUE ANALYSIS")
    print("=" * 60)

    if report.get("error"):
        print(f"\n  ERROR: {report['error']}")
        return

    if report.get("is_synthetic"):
        print("\n  WARNING: SYNTHETIC DATA -- pipeline test only, NOT hypothesis test")

    print(f"\n  E8 Hypothesis Score (top-8 variance):  {report['e8_hypothesis_score']:.4f}")
    print(f"  Expected if E8 holds:                  ~0.877")
    print(f"\n  -> {report['recommendation']}")
    print(f"  -> Recommended intermediate dim n = {report['recommended_n']}")

    print(f"\n  Effective dimensionality:")
    print(f"    90% variance captured at dim:  {report['effective_dim_90pct']}")
    print(f"    95% variance captured at dim:  {report['effective_dim_95pct']}")
    print(f"    99% variance captured at dim:  {report['effective_dim_99pct']}")

    print(f"\n  Spectral gap (L1/L8):  {report['spectral_gap_1_to_8']:.2f}")
    print(f"  Total geodesic variance: {report['total_geodesic_variance']:.6f}")
    print(f"  Tokens compressed:       {report['n_tokens_compressed']}")
    print(f"  Source dim (vocab):      {report['source_dim']}")
    print(f"  Target dim:              {report['target_dim']}")
    print(f"  Compression time:        {report['compression_time_seconds']:.1f}s")

    print(f"\n  Eigenvalue spectrum (top 16):")
    for i, (ev, cv) in enumerate(
        zip(report["eigenvalue_spectrum_top_16"], report["cumulative_variance_top_16"])
    ):
        marker = " <-- E8 rank" if i == 7 else ""
        print(f"    L_{i+1:2d} = {ev:.6f}  (cumulative: {cv:.4f}){marker}")

    print("\n" + "=" * 60)


async def main():
    args = parse_args()

    # Step 1: Get harvest data
    if args.load_harvest:
        from kernel.coordizer_v2.harvest import HarvestResult

        logger.info("Loading previous harvest from %s", args.load_harvest)
        harvest = HarvestResult.load(args.load_harvest)
        logger.info(
            "Loaded: %d fingerprints, vocab=%d",
            len(harvest.token_fingerprints),
            harvest.vocab_size,
        )
    elif args.synthetic:
        harvest = run_synthetic_harvest()
    else:
        # Real Modal harvest
        corpus_texts = None
        if args.corpus:
            corpus_texts = load_corpus(args.corpus)

        harvest = await run_modal_harvest(
            corpus_texts=corpus_texts,
            target_tokens=args.target_tokens,
        )

    # Optionally save harvest for reuse
    if args.save_harvest:
        save_path = Path(args.save_harvest)
        harvest.save(str(save_path))
        logger.info("Harvest saved to %s", save_path)

    # Step 2: Compress
    compression = run_compression(harvest)

    # Step 3: Analyse eigenvalues
    report = analyse_eigenvalues(compression)
    report["is_synthetic"] = args.synthetic

    # Step 4: Output
    print_report(report)

    # Save JSON report
    output_path = Path(args.output)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info("Report saved to %s", output_path)

    # Save eigenvalues for lens module
    if compression.eigenvalues is not None:
        ev_path = str(output_path).replace(".json", ".eigenvalues.npy")
        np.save(ev_path, compression.eigenvalues)
        logger.info("Eigenvalues saved to %s", ev_path)

    return report


if __name__ == "__main__":
    report = asyncio.run(main())
    if report.get("error"):
        sys.exit(1)
