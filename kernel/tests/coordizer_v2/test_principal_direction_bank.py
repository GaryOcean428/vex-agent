"""
Tests for coordizer_v2.principal_direction_bank — PGA direction persistence & projection.

All operations are on the Fisher-Rao manifold. The principal directions
live in tangent space (which IS Euclidean), and projection results are
normalized back to the simplex.
"""

import json

import numpy as np
import pytest

from kernel.coordizer_v2.geometry import BASIN_DIM, E8_RANK
from kernel.coordizer_v2.principal_direction_bank import (
    _DIRECTIONS_FILE,
    _EIGENVALUES_FILE,
    _FRECHET_MEAN_FILE,
    _META_FILE,
    PrincipalDirectionBank,
)

# ═══════════════════════════════════════════════════════════════
#  FIXTURES
# ═══════════════════════════════════════════════════════════════


@pytest.fixture
def rng():
    return np.random.RandomState(42)


@pytest.fixture
def source_dim():
    return 128


@pytest.fixture
def frechet_mean(rng, source_dim):
    """Synthetic Fréchet mean on Δ^(source_dim-1)."""
    raw = rng.dirichlet(np.ones(source_dim))
    return raw.astype(np.float64)


@pytest.fixture
def principal_directions(rng, source_dim):
    """Synthetic principal directions in tangent space (source_dim × BASIN_DIM)."""
    dirs = rng.randn(source_dim, BASIN_DIM)
    # Gram-Schmidt to orthogonalize (valid in tangent space which is Euclidean)
    for j in range(BASIN_DIM):
        for k in range(j):
            proj = np.sum(dirs[:, j] * dirs[:, k])
            dirs[:, j] -= proj * dirs[:, k]
        norm = np.sqrt(np.sum(dirs[:, j] ** 2))
        if norm > 1e-12:
            dirs[:, j] /= norm
    return dirs.astype(np.float64)


@pytest.fixture
def eigenvalues():
    """E8-like eigenvalue spectrum."""
    eigs = np.zeros(BASIN_DIM)
    eigs[:E8_RANK] = np.array([20.0, 15.0, 12.0, 10.0, 8.0, 6.0, 5.0, 4.0])
    eigs[E8_RANK:16] = np.linspace(3.0, 0.5, 8)
    eigs[16:] = np.linspace(0.4, 0.01, BASIN_DIM - 16)
    return eigs.astype(np.float64)


@pytest.fixture
def bank(principal_directions, frechet_mean, eigenvalues):
    return PrincipalDirectionBank.from_compression(
        principal_directions=principal_directions,
        frechet_mean=frechet_mean,
        eigenvalues=eigenvalues,
        meta={"n_tokens": 500, "source_dim": 128},
    )


@pytest.fixture
def fingerprints(rng, source_dim):
    """Synthetic fingerprints on Δ^(source_dim-1)."""
    n = 10
    fps = rng.dirichlet(np.ones(source_dim), size=n)
    return fps.astype(np.float64)


# ═══════════════════════════════════════════════════════════════
#  CONSTRUCTION
# ═══════════════════════════════════════════════════════════════


def test_from_compression_sets_dims(bank, source_dim):
    assert bank.source_dim == source_dim
    assert bank.target_dim == BASIN_DIM


def test_from_compression_stores_components(bank, principal_directions, frechet_mean, eigenvalues):
    np.testing.assert_array_equal(bank.directions, principal_directions)
    np.testing.assert_array_equal(bank.frechet_mean, frechet_mean)
    np.testing.assert_array_equal(bank.eigenvalues, eigenvalues)


def test_is_complete(bank):
    assert bank.is_complete is True


def test_not_complete_without_mean(principal_directions):
    bank = PrincipalDirectionBank(directions=principal_directions)
    assert bank.is_complete is False


# ═══════════════════════════════════════════════════════════════
#  PERSISTENCE (save/load round-trip)
# ═══════════════════════════════════════════════════════════════


def test_save_creates_artifacts(bank, tmp_path):
    out = bank.save(tmp_path / "artifacts")
    assert (out / _DIRECTIONS_FILE).exists()
    assert (out / _FRECHET_MEAN_FILE).exists()
    assert (out / _EIGENVALUES_FILE).exists()
    assert (out / _META_FILE).exists()


def test_save_load_roundtrip(bank, tmp_path):
    out = bank.save(tmp_path / "artifacts")
    loaded = PrincipalDirectionBank.load(out)

    np.testing.assert_allclose(loaded.directions, bank.directions)
    np.testing.assert_allclose(loaded.frechet_mean, bank.frechet_mean)
    np.testing.assert_allclose(loaded.eigenvalues, bank.eigenvalues)
    assert loaded.source_dim == bank.source_dim
    assert loaded.target_dim == bank.target_dim
    assert loaded.is_complete


def test_load_directions_only(bank, tmp_path):
    """Load works even if only directions .npy exists (no mean/eigenvalues)."""
    out = tmp_path / "minimal"
    out.mkdir()
    np.save(out / _DIRECTIONS_FILE, bank.directions)

    loaded = PrincipalDirectionBank.load(out)
    assert loaded.frechet_mean is None
    assert loaded.eigenvalues is None
    assert loaded.is_complete is False
    np.testing.assert_array_equal(loaded.directions, bank.directions)


def test_load_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="Principal direction bank not found"):
        PrincipalDirectionBank.load(tmp_path / "nonexistent")


def test_meta_roundtrip(bank, tmp_path):
    out = bank.save(tmp_path / "artifacts")
    meta = json.loads((out / _META_FILE).read_text(encoding="utf-8"))
    assert meta["n_tokens"] == 500
    assert meta["source_dim"] == 128
    assert meta["target_dim"] == BASIN_DIM


# ═══════════════════════════════════════════════════════════════
#  PROJECTION
# ═══════════════════════════════════════════════════════════════


def test_project_output_shape(bank, fingerprints):
    basins = bank.project(fingerprints)
    assert basins.shape == (10, BASIN_DIM)


def test_project_output_on_simplex(bank, fingerprints):
    """Every projected basin must be on Δ⁶³: non-negative, sum ≈ 1."""
    basins = bank.project(fingerprints)
    assert np.all(basins >= 0), "Basin coordinates must be non-negative"
    sums = basins.sum(axis=1)
    np.testing.assert_allclose(sums, 1.0, atol=1e-10)


def test_project_single(bank, fingerprints):
    basin = bank.project_single(fingerprints[0])
    assert basin.shape == (BASIN_DIM,)
    assert np.all(basin >= 0)
    np.testing.assert_allclose(basin.sum(), 1.0, atol=1e-10)


def test_project_requires_frechet_mean(principal_directions, fingerprints):
    bank = PrincipalDirectionBank(directions=principal_directions)
    with pytest.raises(ValueError, match="Cannot project without a Fréchet mean"):
        bank.project(fingerprints)


def test_project_deterministic(bank, fingerprints):
    """Same input → same output (no randomness in projection)."""
    basins1 = bank.project(fingerprints)
    basins2 = bank.project(fingerprints)
    np.testing.assert_array_equal(basins1, basins2)


def test_project_after_roundtrip(bank, fingerprints, tmp_path):
    """Projection gives identical results after save/load."""
    basins_before = bank.project(fingerprints)

    out = bank.save(tmp_path / "rt")
    loaded = PrincipalDirectionBank.load(out)
    basins_after = loaded.project(fingerprints)

    np.testing.assert_allclose(basins_after, basins_before, atol=1e-12)


# ═══════════════════════════════════════════════════════════════
#  DIAGNOSTICS
# ═══════════════════════════════════════════════════════════════


def test_e8_hypothesis_score(bank):
    score = bank.e8_hypothesis_score()
    # Top 8 eigenvalues: 20+15+12+10+8+6+5+4 = 80
    # Total: 80 + sum(linspace) — score should be substantial
    assert 0.0 < score < 1.0


def test_e8_hypothesis_no_eigenvalues():
    dirs = np.eye(64)
    bank = PrincipalDirectionBank(directions=dirs)
    assert bank.e8_hypothesis_score() == 0.0


def test_explained_variance_ratio(bank):
    evr = bank.explained_variance_ratio()
    assert evr is not None
    assert evr[-1] == pytest.approx(1.0, abs=1e-10)
    # Monotonically increasing
    assert np.all(np.diff(evr) >= 0)


def test_repr(bank):
    r = repr(bank)
    assert "128→64" in r
    assert "complete" in r
    assert "e8=" in r


# ═══════════════════════════════════════════════════════════════
#  LOAD FROM EXISTING shared_artifacts/
# ═══════════════════════════════════════════════════════════════


def test_load_existing_shared_artifacts():
    """Smoke test: load the actual pre-computed artifact if present."""
    from pathlib import Path

    artifact_dir = Path(__file__).resolve().parents[3] / "shared_artifacts"
    if not (artifact_dir / _DIRECTIONS_FILE).exists():
        pytest.skip("shared_artifacts/principal_direction_bank.npy not found")

    bank = PrincipalDirectionBank.load(artifact_dir)
    assert bank.directions.shape == (64, 64)
    # Without frechet_mean, projection won't work — but loading should succeed
    assert bank.directions is not None
