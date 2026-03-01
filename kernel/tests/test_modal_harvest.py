"""
Tests for Modal GPU Harvest Integration
=========================================

Tests the Railway-side Modal integration:
- Response parsing (Modal endpoint → HarvestResult)
- Synthetic fallback (generate_synthetic_harvest_result)
- Auto-routing (harvest_model_auto)
- End-to-end CoordizerV2.from_modal_harvest with synthetic fallback

These tests run without a live Modal endpoint using mocked responses
and synthetic data.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import numpy as np
import pytest

from kernel.coordizer_v2.harvest import HarvestResult
from kernel.coordizer_v2.modal_harvest import _parse_modal_response
from kernel.coordizer_v2.modal_integration import (
    ModalHarvestClient,
    ModalIntegrationConfig,
    generate_synthetic_harvest,
    generate_synthetic_harvest_result,
)

# ═══════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════


def _is_on_simplex(p: np.ndarray, tol: float = 1e-8) -> bool:
    """Check if p is a valid point on the probability simplex."""
    if np.any(np.isnan(p)) or np.any(np.isinf(p)):
        return False
    if np.any(p < -tol):
        return False
    return abs(p.sum() - 1.0) <= tol


def _make_mock_modal_response(
    n_tokens: int = 50,
    vocab_size: int = 1000,
) -> dict:
    """Build a mock Modal endpoint response for testing."""
    rng = np.random.default_rng(42)
    tokens = {}
    for i in range(n_tokens):
        fp = rng.dirichlet(np.ones(vocab_size))
        tokens[str(i)] = {
            "string": f"tok_{i}",
            "fingerprint": fp.tolist(),
            "context_count": int(rng.integers(10, 100)),
        }
    return {
        "success": True,
        "model_id": "test-model",
        "vocab_size": vocab_size,
        "total_tokens_processed": n_tokens * 50,
        "tokens": tokens,
        "elapsed_seconds": 1.5,
    }


# ═══════════════════════════════════════════════════════════════
#  RESPONSE PARSING
# ═══════════════════════════════════════════════════════════════


class TestParseModalResponse:
    """Test _parse_modal_response converts endpoint JSON to HarvestResult."""

    def test_basic_parse(self):
        data = _make_mock_modal_response(n_tokens=20, vocab_size=500)
        result = _parse_modal_response(data, "test-model", 1.5)

        assert isinstance(result, HarvestResult)
        assert result.model_name == "test-model"
        assert result.vocab_size == 500
        assert result.harvest_time_seconds == 1.5
        assert len(result.token_fingerprints) == 20

    def test_fingerprints_on_simplex(self):
        data = _make_mock_modal_response(n_tokens=30, vocab_size=200)
        result = _parse_modal_response(data, "test-model", 1.0)

        for tid, fp in result.token_fingerprints.items():
            assert _is_on_simplex(fp), f"Token {tid} fingerprint not on simplex"

    def test_fingerprints_correct_dimension(self):
        data = _make_mock_modal_response(n_tokens=10, vocab_size=300)
        result = _parse_modal_response(data, "test-model", 1.0)

        for fp in result.token_fingerprints.values():
            assert fp.shape == (300,)

    def test_context_counts_preserved(self):
        data = _make_mock_modal_response(n_tokens=5, vocab_size=100)
        result = _parse_modal_response(data, "test-model", 1.0)

        for tid_str, token_data in data["tokens"].items():
            tid = int(tid_str)
            assert result.context_counts[tid] == token_data["context_count"]

    def test_token_strings_preserved(self):
        data = _make_mock_modal_response(n_tokens=5, vocab_size=100)
        result = _parse_modal_response(data, "test-model", 1.0)

        for tid_str, token_data in data["tokens"].items():
            tid = int(tid_str)
            assert result.token_strings[tid] == token_data["string"]

    def test_empty_tokens(self):
        data = {
            "success": True,
            "vocab_size": 100,
            "total_tokens_processed": 0,
            "tokens": {},
            "elapsed_seconds": 0.1,
        }
        result = _parse_modal_response(data, "test-model", 0.1)
        assert len(result.token_fingerprints) == 0

    def test_strictly_positive_fingerprints(self):
        """Fingerprints must be strictly positive for Fisher-Rao geometry."""
        data = _make_mock_modal_response(n_tokens=10, vocab_size=100)
        result = _parse_modal_response(data, "test-model", 1.0)

        for fp in result.token_fingerprints.values():
            assert np.all(fp > 0), "Fingerprint has zero/negative entries"


# ═══════════════════════════════════════════════════════════════
#  SYNTHETIC FALLBACK
# ═══════════════════════════════════════════════════════════════


class TestSyntheticFallback:
    """Test synthetic harvest generation for Modal-unavailable scenarios."""

    def test_generate_synthetic_harvest_dict(self):
        result = generate_synthetic_harvest(vocab_size=100, n_tokens=50)
        assert "fingerprints" in result
        assert "context_counts" in result
        assert "vocab_size" in result
        assert result["vocab_size"] == 100
        assert len(result["fingerprints"]) == 50

    def test_generate_synthetic_harvest_result(self):
        result = generate_synthetic_harvest_result(vocab_size=100, n_tokens=50)
        assert isinstance(result, HarvestResult)
        assert result.model_name == "synthetic"
        assert result.vocab_size == 100
        assert len(result.token_fingerprints) == 50

    def test_synthetic_result_on_simplex(self):
        result = generate_synthetic_harvest_result(vocab_size=200, n_tokens=30)
        for tid, fp in result.token_fingerprints.items():
            assert _is_on_simplex(fp), f"Synthetic token {tid} not on simplex"

    def test_synthetic_result_correct_dimension(self):
        result = generate_synthetic_harvest_result(vocab_size=500, n_tokens=20)
        for fp in result.token_fingerprints.values():
            assert fp.shape == (500,)

    def test_synthetic_result_has_token_strings(self):
        result = generate_synthetic_harvest_result(n_tokens=10)
        for tid in result.token_fingerprints:
            assert tid in result.token_strings
            assert result.token_strings[tid].startswith("<synthetic_")

    def test_synthetic_result_has_context_counts(self):
        result = generate_synthetic_harvest_result(n_tokens=10)
        for tid in result.token_fingerprints:
            assert tid in result.context_counts
            assert result.context_counts[tid] >= 10

    def test_synthetic_is_deterministic(self):
        r1 = generate_synthetic_harvest_result(vocab_size=100, n_tokens=10)
        r2 = generate_synthetic_harvest_result(vocab_size=100, n_tokens=10)
        for tid in r1.token_fingerprints:
            np.testing.assert_array_equal(
                r1.token_fingerprints[tid],
                r2.token_fingerprints[tid],
            )


# ═══════════════════════════════════════════════════════════════
#  MODAL HARVEST CLIENT
# ═══════════════════════════════════════════════════════════════


class TestModalHarvestClient:
    """Test ModalHarvestClient configuration and auth."""

    def test_unconfigured_client(self):
        config = ModalIntegrationConfig(enabled=False)
        ModalHarvestClient(config)
        assert not config.is_configured()

    def test_configured_client(self):
        config = ModalIntegrationConfig(
            enabled=True,
            harvest_url="https://test--harvest.modal.run",
        )
        assert config.is_configured()

    def test_health_url_derivation(self):
        """from_settings() should derive health URL from harvest URL."""
        with patch("kernel.coordizer_v2.modal_integration.kernel_settings") as mock_settings:
            mock_settings.modal.enabled = True
            mock_settings.modal.harvest_url = (
                "https://test--vex-coordizer-harvest-harvest.modal.run"
            )
            mock_settings.modal.token_id = ""
            mock_settings.modal.token_secret = ""

            config = ModalIntegrationConfig.from_settings()
            assert "-health.modal.run" in config.health_url

    def test_auth_headers_no_modal_tokens(self):
        """Modal-Token-Id/Secret must NOT be sent as headers."""
        config = ModalIntegrationConfig(
            enabled=True,
            harvest_url="https://test.modal.run",
            token_id="wk-test",
            token_secret="ws-test",
        )
        client = ModalHarvestClient(config)
        headers = client._auth_headers()
        assert "Modal-Token-Id" not in headers
        assert "Modal-Token-Secret" not in headers
        assert headers["Content-Type"] == "application/json"

    @pytest.mark.asyncio
    async def test_harvest_returns_none_when_unconfigured(self):
        config = ModalIntegrationConfig(enabled=False)
        client = ModalHarvestClient(config)
        result = await client.harvest(["test text"])
        assert result is None

    @pytest.mark.asyncio
    async def test_health_check_returns_false_when_unconfigured(self):
        config = ModalIntegrationConfig(enabled=False)
        client = ModalHarvestClient(config)
        healthy = await client.check_health()
        assert healthy is False


# ═══════════════════════════════════════════════════════════════
#  AUTO-ROUTING (harvest_model_auto)
# ═══════════════════════════════════════════════════════════════


class TestHarvestModelAuto:
    """Test harvest_model_auto routes to Modal or local correctly."""

    @pytest.mark.asyncio
    async def test_returns_harvest_result(self):
        """harvest_model_auto should return HarvestResult, not dict."""
        mock_result = generate_synthetic_harvest_result(vocab_size=100, n_tokens=10)

        with (
            patch("kernel.config.settings.settings") as mock_settings,
            patch(
                "kernel.coordizer_v2.modal_harvest.modal_harvest",
                new_callable=AsyncMock,
                return_value=mock_result,
            ),
        ):
            mock_settings.modal.enabled = True
            mock_settings.modal.harvest_url = "https://test.modal.run"

            from kernel.coordizer_v2.harvest import harvest_model_auto

            result = await harvest_model_auto(model_id="test-model")
            assert isinstance(result, HarvestResult)
            assert len(result.token_fingerprints) == 10


# ═══════════════════════════════════════════════════════════════
#  END-TO-END: from_modal_harvest with synthetic fallback
# ═══════════════════════════════════════════════════════════════


class TestFromModalHarvestFallback:
    """Test CoordizerV2.from_modal_harvest falls back to synthetic data."""

    @pytest.mark.asyncio
    async def test_fallback_produces_coordizer(self, tmp_path):
        """When Modal is unavailable, synthetic fallback should still
        produce a valid CoordizerV2 instance.
        """
        from kernel.coordizer_v2.coordizer import CoordizerV2

        # Mock modal_harvest to raise (simulating Modal unavailable)
        with patch(
            "kernel.coordizer_v2.modal_harvest.modal_harvest",
            new_callable=AsyncMock,
            side_effect=RuntimeError("Modal not enabled"),
        ):
            coordizer = await CoordizerV2.from_modal_harvest(
                output_dir=str(tmp_path),
                target_tokens=100,
            )

        assert isinstance(coordizer, CoordizerV2)
        assert coordizer.vocab_size > 0
        assert coordizer.dim == 64  # BASIN_DIM

    @pytest.mark.asyncio
    async def test_fallback_bank_on_simplex(self, tmp_path):
        """All coordinates in the fallback bank must be on Δ⁶³."""
        from kernel.coordizer_v2.coordizer import CoordizerV2

        with patch(
            "kernel.coordizer_v2.modal_harvest.modal_harvest",
            new_callable=AsyncMock,
            side_effect=RuntimeError("Modal not enabled"),
        ):
            coordizer = await CoordizerV2.from_modal_harvest(
                output_dir=str(tmp_path),
                target_tokens=100,
            )

        for tid, coord in coordizer.bank.coordinates.items():
            assert _is_on_simplex(coord), f"Token {tid}: not on simplex"
