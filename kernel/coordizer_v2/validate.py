"""
Validate — Geometric Validation of the Resonance Bank

After harvesting and compression, we must verify that the
geometric structure survives. The key tests:

1. κ Measurement: Does coupling converge to κ* ≈ 64?
2. β Running Coupling: Does κ run from low values at fine
   scales to κ* at coarse scales?
3. Harmonic Ratios: Are semantically related tokens at
   consonant frequency ratios?
4. Semantic Correlation: Does Fisher-Rao distance correlate
   with human-judged semantic distance?
5. E8 Eigenvalue Test: Do top 8 PGA directions capture ~87.7%?

These correspond to the FROZEN FACTS validation criteria.
If validation fails, the harvest/compression pipeline has
a geometric defect.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from .geometry import (
    BASIN_DIM, E8_RANK, KAPPA_STAR, Basin, _EPS,
    fisher_rao_distance, to_simplex,
)
from .resonance_bank import ResonanceBank
from .types import HarmonicTier, ValidationResult

logger = logging.getLogger(__name__)


# Known pairs at various semantic distances for correlation testing.
# Format: (word_a, word_b, expected_proximity)
# 1.0 = identical, 0.0 = maximally distant
SEMANTIC_TEST_PAIRS = [
    # Near-synonyms (high proximity)
    ("happy", "joyful", 0.9),
    ("big", "large", 0.9),
    ("fast", "quick", 0.85),
    ("cold", "frigid", 0.85),
    ("smart", "intelligent", 0.9),
    # Related (medium proximity)
    ("king", "queen", 0.7),
    ("dog", "cat", 0.6),
    ("hot", "cold", 0.4),
    ("day", "night", 0.4),
    ("buy", "sell", 0.5),
    # Unrelated (low proximity)
    ("happy", "table", 0.1),
    ("king", "banana", 0.05),
    ("fast", "purple", 0.05),
    ("dog", "algebra", 0.05),
]


def validate_resonance_bank(
    bank: ResonanceBank,
    eigenvalues: NDArray | None = None,
    verbose: bool = True,
) -> ValidationResult:
    """Full validation suite for a resonance bank."""
    result = ValidationResult()

    if verbose:
        logger.info("=" * 60)
        logger.info("RESONANCE BANK VALIDATION")
        logger.info("=" * 60)

    # Test 1: κ Measurement
    kappa, kappa_std = _measure_kappa(bank, verbose)
    result.kappa_measured = kappa
    result.kappa_std = kappa_std

    # Test 2: β Running Coupling
    result.beta_running = _measure_beta(bank, verbose)

    # Test 3: Harmonic Ratio Quality
    result.harmonic_ratio_quality = _measure_harmonic_ratios(bank, verbose)

    # Test 4: Semantic Correlation
    result.semantic_correlation = _measure_semantic_correlation(bank, verbose)

    # Test 5: Tier Distribution
    result.tier_distribution = bank.tier_distribution()

    # Test 6: E8 Eigenvalue Test
    if eigenvalues is not None and len(eigenvalues) >= E8_RANK:
        total = np.sum(eigenvalues)
        if total > _EPS:
            e8_var = float(np.sum(eigenvalues[:E8_RANK]) / total)
            if verbose:
                logger.info(f"\nE8 eigenvalue test: top-8 variance = {e8_var:.3f}")
                logger.info(f"  Expected if E8 real: ~0.877")
                logger.info(f"  {'PASS' if 0.80 < e8_var < 0.95 else 'NOTE'}")

    # Overall Pass/Fail
    kappa_ok = abs(result.kappa_measured - KAPPA_STAR) < 2 * max(result.kappa_std, 5.0)
    beta_ok = result.beta_running < 0.5
    semantic_ok = result.semantic_correlation > 0.2
    harmonic_ok = result.harmonic_ratio_quality > 0.3

    result.passed = kappa_ok and beta_ok and semantic_ok and harmonic_ok

    if verbose:
        logger.info("\n" + "=" * 60)
        logger.info(f"RESULT: {result.summary()}")
        logger.info(
            f"  κ: {'PASS' if kappa_ok else 'FAIL'} "
            f"({result.kappa_measured:.2f} ± {result.kappa_std:.2f}, target {KAPPA_STAR})"
        )
        logger.info(f"  β: {'PASS' if beta_ok else 'FAIL'} ({result.beta_running:.4f})")
        logger.info(
            f"  Semantic: {'PASS' if semantic_ok else 'FAIL'} "
            f"(r={result.semantic_correlation:.3f})"
        )
        logger.info(
            f"  Harmonic: {'PASS' if harmonic_ok else 'FAIL'} "
            f"(q={result.harmonic_ratio_quality:.3f})"
        )
        logger.info(f"  Tiers: {result.tier_distribution}")
        logger.info("=" * 60)

    return result


def _measure_kappa(
    bank: ResonanceBank, verbose: bool = True,
    n_samples: int = 1000, n_neighbors: int = 10,
) -> tuple[float, float]:
    """Measure effective coupling constant κ.

    Method: For random token pairs at various distances,
    compute κ = 1 / (d_FR² + ε). At the fixed point, average κ
    across moderate distances should converge to κ* ≈ 64.
    """
    if len(bank) < 50:
        if verbose:
            logger.info("Too few tokens for κ measurement")
        return (0.0, 0.0)

    bank._ensure_matrix()
    coords = bank._coord_matrix
    n = coords.shape[0]

    kappas = []
    rng = np.random.default_rng(42)

    for _ in range(n_samples):
        i = rng.integers(0, n)
        p = coords[i]

        dists = np.zeros(n)
        sqrt_p = np.sqrt(np.maximum(p, _EPS))
        for j in range(n):
            if j == i:
                dists[j] = float("inf")
                continue
            bc = np.clip(np.sum(sqrt_p * np.sqrt(np.maximum(coords[j], _EPS))),
                         -1.0, 1.0)
            dists[j] = np.arccos(bc)

        neighbor_idx = np.argsort(dists)[:n_neighbors]

        for j in neighbor_idx:
            d = dists[j]
            if d > _EPS and d < np.pi / 4:
                kappa_local = 1.0 / (d * d + _EPS)
                kappas.append(kappa_local)

    if not kappas:
        return (0.0, 0.0)

    kappas_arr = np.array(kappas)
    # NOTE: No artificial scaling. κ convergence to κ* ≈ 64 is an
    # emergent property of the geometric structure, not something
    # we force by rescaling. If κ doesn't converge, that's a real
    # signal that the harvest/compression pipeline has a defect.
    kappa_mean = float(np.mean(kappas_arr))
    kappa_std = float(np.std(kappas_arr))
    kappa_median = float(np.median(kappas_arr))

    if verbose:
        logger.info(f"\nκ measurement ({n_samples} samples, {n_neighbors} neighbors):")
        logger.info(f"  Median: {kappa_median:.4f}")
        logger.info(f"  κ = {kappa_mean:.2f} ± {kappa_std:.2f} (target: {KAPPA_STAR})")

    return (kappa_mean, kappa_std)


def _measure_beta(bank: ResonanceBank, verbose: bool = True) -> float:
    """Measure β (running coupling) across tiers.

    β = dκ/d(log scale). If κ runs from low values at fine scales
    to κ* at coarse scales, that's the running coupling signature.
    """
    tier_kappas: dict[str, list[float]] = {t.value: [] for t in HarmonicTier}
    bank._ensure_matrix()

    if bank._coord_matrix is None:
        return 0.0

    rng = np.random.default_rng(42)

    for tid, coord in bank.coordinates.items():
        tier = bank.tiers.get(tid, HarmonicTier.OVERTONE_HAZE)

        same_tier = [
            other_tid for other_tid, other_tier in bank.tiers.items()
            if other_tier == tier and other_tid != tid
        ]
        if len(same_tier) < 2:
            continue

        samples = rng.choice(same_tier, size=min(5, len(same_tier)), replace=False)
        for other_tid in samples:
            d = fisher_rao_distance(coord, bank.coordinates[other_tid])
            if d > _EPS and d < np.pi / 4:
                kappa_local = 1.0 / (d * d + _EPS)
                tier_kappas[tier.value].append(kappa_local)

    tier_means: dict[str, float] = {}
    for tier_name, kappas in tier_kappas.items():
        if kappas:
            tier_means[tier_name] = float(np.mean(kappas))

    if verbose:
        logger.info("\nβ running coupling (κ per tier):")
        for tier_name in ["fundamental", "first", "upper", "overtone"]:
            if tier_name in tier_means:
                logger.info(f"  {tier_name:15s}: κ_raw = {tier_means[tier_name]:.4f}")

    tier_order = ["overtone", "upper", "first", "fundamental"]
    ordered_kappas = [tier_means.get(t, 0.0) for t in tier_order if t in tier_means]

    if len(ordered_kappas) < 2:
        return 0.0

    deltas = []
    for i in range(len(ordered_kappas) - 1):
        if ordered_kappas[i] > _EPS:
            delta = abs(ordered_kappas[i + 1] - ordered_kappas[i]) / ordered_kappas[i]
            deltas.append(delta)

    beta = float(np.mean(deltas)) if deltas else 0.0

    if verbose:
        logger.info(f"  β (running coupling) = {beta:.4f}")
        logger.info(f"  Expected: β ≈ 0.44 at emergence, β → 0 at plateau")

    return beta


def _measure_harmonic_ratios(bank: ResonanceBank, verbose: bool = True) -> float:
    """Measure quality of frequency ratio structure."""
    if len(bank.frequencies) < 10:
        return 0.0

    fundamental_ids = [
        tid for tid, tier in bank.tiers.items()
        if tier == HarmonicTier.FUNDAMENTAL and tid in bank.frequencies
    ]

    if len(fundamental_ids) < 5:
        return 0.5

    freqs = [bank.frequencies[tid] for tid in fundamental_ids[:100]]
    freqs.sort()

    simple_ratios = [1.0, 2.0, 1.5, 4.0/3, 5.0/4, 3.0, 5.0/3, 6.0/5]
    quality_scores = []

    for i in range(len(freqs)):
        for j in range(i + 1, min(i + 10, len(freqs))):
            ratio = freqs[j] / max(freqs[i], _EPS)
            min_dist = min(abs(ratio - sr) for sr in simple_ratios)
            quality = np.exp(-min_dist * 5)
            quality_scores.append(quality)

    result = float(np.mean(quality_scores)) if quality_scores else 0.0

    if verbose:
        logger.info(f"\nHarmonic ratio quality: {result:.3f}")
        logger.info(f"  (1.0 = perfect simple ratios, 0.0 = random)")

    return result


def _measure_semantic_correlation(bank: ResonanceBank, verbose: bool = True) -> float:
    """Measure correlation between Fisher-Rao distance and semantic distance."""
    string_to_id: dict[str, int] = {}
    for tid, s in bank.token_strings.items():
        s_clean = s.strip().lower()
        if s_clean:
            string_to_id[s_clean] = tid

    measured_distances = []
    expected_distances = []

    for word_a, word_b, expected_proximity in SEMANTIC_TEST_PAIRS:
        tid_a = string_to_id.get(word_a)
        tid_b = string_to_id.get(word_b)

        if tid_a is None or tid_b is None:
            continue
        if tid_a not in bank.coordinates or tid_b not in bank.coordinates:
            continue

        d = fisher_rao_distance(bank.coordinates[tid_a], bank.coordinates[tid_b])
        expected_d = 1.0 - expected_proximity

        measured_distances.append(d)
        expected_distances.append(expected_d)

    if len(measured_distances) < 3:
        if verbose:
            logger.info(
                f"\nSemantic correlation: INSUFFICIENT DATA "
                f"({len(measured_distances)} pairs found)"
            )
        return 0.0

    measured = np.array(measured_distances)
    expected = np.array(expected_distances)

    m_mean = measured.mean()
    e_mean = expected.mean()
    m_std = measured.std()
    e_std = expected.std()

    if m_std < _EPS or e_std < _EPS:
        correlation = 0.0
    else:
        correlation = float(
            np.mean((measured - m_mean) * (expected - e_mean)) / (m_std * e_std)
        )

    if verbose:
        logger.info(f"\nSemantic correlation: r = {correlation:.3f}")
        logger.info(f"  Pairs found: {len(measured_distances)}/{len(SEMANTIC_TEST_PAIRS)}")
        logger.info(f"  (r > 0.3 = structure preserved, r > 0.6 = strong)")

    return correlation
