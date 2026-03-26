"""
Tests for modal_url() — Modal endpoint URL builder
====================================================

The modal_url(base_url, path) helper centralises Modal URL derivation
that was previously done inline with string manipulation across 6+ files.

Covers all routing patterns:
1. ASGI base URL + path append
2. ASGI base with existing route (replace path)
3. Legacy hostname pattern (swap method segment in hostname)
4. Legacy hostname exact match (return as-is)
5. No double-path duplication
Plus edge cases: trailing slashes, empty path, unknown routes.
"""

from __future__ import annotations

import pytest

from kernel.config.settings import modal_url

# ═══════════════════════════════════════════════════════════════
#  CONSTANTS — realistic Modal URLs used throughout
# ═══════════════════════════════════════════════════════════════

ASGI_BASE = "https://archelon--vex-qlora-train-qloratrainer-web.modal.run"
HARVEST_ASGI = "https://archelon--vex-coordizer-harvest-coordizerharvester-web.modal.run"

LEGACY_TRAIN = "https://archelon--vex-qlora-train.modal.run"
LEGACY_HARVEST = "https://archelon--vex-coordizer-harvest.modal.run"
LEGACY_HEALTH = "https://archelon--vex-coordizer-health.modal.run"
LEGACY_INFER = "https://archelon--vex-inference-infer.modal.run"


# ═══════════════════════════════════════════════════════════════
#  1. ASGI base URL + path append
# ═══════════════════════════════════════════════════════════════


class TestASGIAppend:
    """ASGI base URLs ending in web.modal.run — append path."""

    @pytest.mark.parametrize(
        "path, expected_suffix",
        [
            ("health", "/health"),
            ("train", "/train"),
            ("infer", "/infer"),
            ("status", "/status"),
            ("harvest", "/harvest"),
            ("coordize", "/coordize"),
            ("data-receive", "/data-receive"),
            ("data-stats", "/data-stats"),
            ("export-image", "/export-image"),
        ],
    )
    def test_asgi_base_appends_path(self, path: str, expected_suffix: str) -> None:
        result = modal_url(ASGI_BASE, path)
        assert result == f"{ASGI_BASE}{expected_suffix}"

    def test_asgi_harvest_base_appends_health(self) -> None:
        result = modal_url(HARVEST_ASGI, "health")
        assert result == f"{HARVEST_ASGI}/health"


# ═══════════════════════════════════════════════════════════════
#  2. ASGI base with existing route — replace path
# ═══════════════════════════════════════════════════════════════


class TestASGIReplacePath:
    """When base already has a known route suffix, strip it and append new path."""

    @pytest.mark.parametrize(
        "existing_route, new_path",
        [
            ("/train", "health"),
            ("/train", "infer"),
            ("/train", "status"),
            ("/harvest", "health"),
            ("/harvest", "coordize"),
            ("/infer", "health"),
            ("/health", "train"),
            ("/status", "health"),
            ("/coordize", "harvest"),
            ("/data-receive", "health"),
            ("/data-stats", "train"),
            ("/export-image", "status"),
        ],
    )
    def test_replaces_existing_route(self, existing_route: str, new_path: str) -> None:
        base_with_route = f"{ASGI_BASE}{existing_route}"
        result = modal_url(base_with_route, new_path)
        assert result == f"{ASGI_BASE}/{new_path}"

    def test_train_to_health(self) -> None:
        """Concrete example from docstring: .../train + health -> .../health."""
        base = f"{ASGI_BASE}/train"
        result = modal_url(base, "health")
        assert result == f"{ASGI_BASE}/health"


# ═══════════════════════════════════════════════════════════════
#  3. Legacy hostname pattern — swap method in hostname
# ═══════════════════════════════════════════════════════════════


class TestLegacyHostnameSwap:
    """Legacy per-method hostnames: ...-train.modal.run -> ...-health.modal.run."""

    @pytest.mark.parametrize(
        "base, new_path, expected",
        [
            # train -> health
            (
                LEGACY_TRAIN,
                "health",
                "https://archelon--vex-qlora-health.modal.run",
            ),
            # train -> infer
            (
                LEGACY_TRAIN,
                "infer",
                "https://archelon--vex-qlora-infer.modal.run",
            ),
            # harvest -> health
            (
                LEGACY_HARVEST,
                "health",
                "https://archelon--vex-coordizer-health.modal.run",
            ),
            # health -> train
            (
                LEGACY_HEALTH,
                "train",
                "https://archelon--vex-coordizer-train.modal.run",
            ),
            # infer -> status
            (
                LEGACY_INFER,
                "status",
                "https://archelon--vex-inference-status.modal.run",
            ),
        ],
    )
    def test_swaps_method_in_hostname(self, base: str, new_path: str, expected: str) -> None:
        result = modal_url(base, new_path)
        assert result == expected

    @pytest.mark.parametrize(
        "method",
        ["train", "infer", "health", "status", "harvest", "coordize"],
    )
    def test_all_known_methods_are_swappable(self, method: str) -> None:
        """Every known method suffix in the legacy hostname can be swapped out."""
        base = f"https://app--svc-{method}.modal.run"
        result = modal_url(base, "health" if method != "health" else "train")
        target = "health" if method != "health" else "train"
        expected = f"https://app--svc-{target}.modal.run"
        assert result == expected


# ═══════════════════════════════════════════════════════════════
#  4. Legacy hostname exact match — return as-is
# ═══════════════════════════════════════════════════════════════


class TestLegacyExactMatch:
    """When base already ends with -<path>.modal.run, return unchanged."""

    @pytest.mark.parametrize(
        "base, path",
        [
            (LEGACY_TRAIN, "train"),
            (LEGACY_HARVEST, "harvest"),
            (LEGACY_HEALTH, "health"),
            (LEGACY_INFER, "infer"),
            ("https://x--svc-status.modal.run", "status"),
            ("https://x--svc-coordize.modal.run", "coordize"),
        ],
    )
    def test_returns_as_is(self, base: str, path: str) -> None:
        result = modal_url(base, path)
        assert result == base


# ═══════════════════════════════════════════════════════════════
#  5. No double path — same route already in ASGI base
# ═══════════════════════════════════════════════════════════════


class TestNoDoublePath:
    """ASGI base already has the requested route — should not duplicate."""

    @pytest.mark.parametrize(
        "route",
        [
            "train",
            "infer",
            "health",
            "status",
            "harvest",
            "coordize",
            "data-receive",
            "data-stats",
            "export-image",
        ],
    )
    def test_no_duplication(self, route: str) -> None:
        base_with_route = f"{ASGI_BASE}/{route}"
        result = modal_url(base_with_route, route)
        assert result == f"{ASGI_BASE}/{route}"
        # Must NOT produce .../route/route
        assert not result.endswith(f"/{route}/{route}")


# ═══════════════════════════════════════════════════════════════
#  EDGE CASES
# ═══════════════════════════════════════════════════════════════


class TestEdgeCases:
    """Trailing slashes, leading slashes on path, empty path, etc."""

    def test_trailing_slash_on_base(self) -> None:
        result = modal_url(f"{ASGI_BASE}/", "health")
        assert result == f"{ASGI_BASE}/health"

    def test_multiple_trailing_slashes(self) -> None:
        result = modal_url(f"{ASGI_BASE}///", "health")
        # rstrip("/") removes all trailing slashes
        assert result == f"{ASGI_BASE}/health"

    def test_leading_slash_on_path(self) -> None:
        result = modal_url(ASGI_BASE, "/health")
        assert result == f"{ASGI_BASE}/health"

    def test_both_slashes(self) -> None:
        result = modal_url(f"{ASGI_BASE}/", "/health")
        assert result == f"{ASGI_BASE}/health"

    def test_empty_path(self) -> None:
        result = modal_url(ASGI_BASE, "")
        assert result == f"{ASGI_BASE}/"

    def test_path_is_just_slash(self) -> None:
        result = modal_url(ASGI_BASE, "/")
        assert result == f"{ASGI_BASE}/"

    def test_unknown_route_in_path(self) -> None:
        """Unknown route names are appended verbatim (ASGI fallback)."""
        result = modal_url(ASGI_BASE, "custom-endpoint")
        assert result == f"{ASGI_BASE}/custom-endpoint"

    def test_unknown_route_on_asgi_base_is_stripped(self) -> None:
        """ASGI base detection strips ANY path after -web.modal.run, not just known routes."""
        base = f"{ASGI_BASE}/my-custom-path"
        result = modal_url(base, "health")
        assert result == f"{ASGI_BASE}/health"

    def test_unknown_route_on_non_asgi_base_is_kept(self) -> None:
        """Non-ASGI, non-legacy URLs preserve existing path segments."""
        base = "https://example.com/my-custom-path"
        result = modal_url(base, "health")
        assert result == "https://example.com/my-custom-path/health"

    def test_trailing_slash_on_base_with_route(self) -> None:
        """Base has /train/ (trailing slash) — should still strip and replace."""
        base = f"{ASGI_BASE}/train/"
        result = modal_url(base, "health")
        # rstrip("/") removes trailing slash first, then /train is detected
        assert result == f"{ASGI_BASE}/health"

    def test_legacy_base_with_trailing_slash(self) -> None:
        result = modal_url(f"{LEGACY_TRAIN}/", "health")
        # After rstrip("/"), base is LEGACY_TRAIN → legacy hostname swap
        assert result == "https://archelon--vex-qlora-health.modal.run"

    def test_legacy_exact_match_with_trailing_slash(self) -> None:
        result = modal_url(f"{LEGACY_TRAIN}/", "train")
        # After rstrip("/"), base ends with -train.modal.run → exact match
        assert result == LEGACY_TRAIN


# ═══════════════════════════════════════════════════════════════
#  INTEGRATION-STYLE — realistic caller patterns
# ═══════════════════════════════════════════════════════════════


class TestRealisticCallerPatterns:
    """Patterns that match actual call sites in the codebase."""

    def test_training_url_to_infer(self) -> None:
        """ModalConfig.training_url → /infer for PEFT inference."""
        training = ASGI_BASE
        result = modal_url(training, "infer")
        assert "/infer" in result
        assert result.endswith("/infer")

    def test_training_url_to_health(self) -> None:
        """Health check from training URL."""
        training = ASGI_BASE
        result = modal_url(training, "health")
        assert result.endswith("/health")

    def test_harvest_url_to_health(self) -> None:
        """ModalConfig.harvest_url → /health for harvest health check."""
        result = modal_url(HARVEST_ASGI, "health")
        assert result.endswith("/health")
        assert "web.modal.run" in result

    def test_harvest_url_to_coordize(self) -> None:
        """Harvest base → /coordize endpoint."""
        result = modal_url(HARVEST_ASGI, "coordize")
        assert result.endswith("/coordize")

    def test_training_url_with_train_path_to_status(self) -> None:
        """Caller stored base as .../train, wants /status."""
        base = f"{ASGI_BASE}/train"
        result = modal_url(base, "status")
        assert result.endswith("/status")
        assert "/train" not in result

    def test_chained_calls_are_idempotent(self) -> None:
        """Calling modal_url on its own output with the same path returns the same URL."""
        first = modal_url(ASGI_BASE, "health")
        second = modal_url(first, "health")
        assert first == second


# ═══════════════════════════════════════════════════════════════
#  PRODUCTION URLs — exact env var values from Railway
# ═══════════════════════════════════════════════════════════════

# These are the ACTUAL MODAL_TRAINING_URL and MODAL_HARVEST_URL values.
PROD_TRAINING_URL = "https://archelon--vex-qlora-train-qloratrainer-web.modal.run"
PROD_HARVEST_URL = "https://archelon--vex-coordizer-harvest-coordizerharvester-web.modal.run"


class TestProductionURLs:
    """Verify all critical call sites produce correct URLs with production env vars."""

    # ── Training URL call sites ──────────────────────────────────

    def test_training_to_train(self) -> None:
        """modal_url(training_url, 'train') — server.py line 314."""
        result = modal_url(PROD_TRAINING_URL, "train")
        assert result == f"{PROD_TRAINING_URL}/train"

    def test_training_to_health(self) -> None:
        """modal_url(training_url, 'health') — server.py line 630."""
        result = modal_url(PROD_TRAINING_URL, "health")
        assert result == f"{PROD_TRAINING_URL}/health"

    def test_training_to_infer(self) -> None:
        """modal_url(training_url, 'infer') — client.py line 236."""
        result = modal_url(PROD_TRAINING_URL, "infer")
        assert result == f"{PROD_TRAINING_URL}/infer"

    def test_training_to_data_receive(self) -> None:
        """modal_url(training_url, 'data-receive') — ingest.py."""
        result = modal_url(PROD_TRAINING_URL, "data-receive")
        assert result == f"{PROD_TRAINING_URL}/data-receive"

    def test_training_to_status(self) -> None:
        """modal_url(training_url, 'status') — ingest.py line 1568."""
        result = modal_url(PROD_TRAINING_URL, "status")
        assert result == f"{PROD_TRAINING_URL}/status"

    def test_training_to_data_stats(self) -> None:
        """modal_url(training_url, 'data-stats') — ingest.py line 1826."""
        result = modal_url(PROD_TRAINING_URL, "data-stats")
        assert result == f"{PROD_TRAINING_URL}/data-stats"

    def test_training_to_export_image(self) -> None:
        """modal_url(training_url, 'export-image')."""
        result = modal_url(PROD_TRAINING_URL, "export-image")
        assert result == f"{PROD_TRAINING_URL}/export-image"

    # ── Harvest URL call sites ───────────────────────────────────

    def test_harvest_to_harvest(self) -> None:
        """modal_url(harvest_url, 'harvest') — modal_integration.py line 72."""
        result = modal_url(PROD_HARVEST_URL, "harvest")
        assert result == f"{PROD_HARVEST_URL}/harvest"

    def test_harvest_to_coordize(self) -> None:
        """modal_url(harvest_url, 'coordize')."""
        result = modal_url(PROD_HARVEST_URL, "coordize")
        assert result == f"{PROD_HARVEST_URL}/coordize"

    def test_harvest_to_health(self) -> None:
        """modal_url(harvest_url, 'health') — modal_integration.py line 68."""
        result = modal_url(PROD_HARVEST_URL, "health")
        assert result == f"{PROD_HARVEST_URL}/health"

    # ── Chained call patterns (peft_client.py) ───────────────────

    def test_infer_url_to_health(self) -> None:
        """peft_client stores infer_url = modal_url(training, 'infer'), then calls
        modal_url(infer_url, 'health'). Must produce .../web.modal.run/health."""
        infer_url = modal_url(PROD_TRAINING_URL, "infer")
        health_url = modal_url(infer_url, "health")
        assert health_url == f"{PROD_TRAINING_URL}/health"

    def test_harvest_url_stored_then_requeried(self) -> None:
        """modal_integration stores harvest_url = modal_url(base, 'harvest'), then
        a caller might pass it back. Must not double to /harvest/harvest."""
        stored = modal_url(PROD_HARVEST_URL, "harvest")
        again = modal_url(stored, "harvest")
        assert again == f"{PROD_HARVEST_URL}/harvest"
        assert "/harvest/harvest" not in again

    # ── Idempotency (requirement #5) ────────────────────────────

    @pytest.mark.parametrize("path", ["health", "train", "infer", "harvest", "coordize"])
    def test_idempotent_training(self, path: str) -> None:
        first = modal_url(PROD_TRAINING_URL, path)
        second = modal_url(first, path)
        assert first == second

    @pytest.mark.parametrize("path", ["health", "harvest", "coordize"])
    def test_idempotent_harvest(self, path: str) -> None:
        first = modal_url(PROD_HARVEST_URL, path)
        second = modal_url(first, path)
        assert first == second

    # ── ASGI base detection strips unknown routes too ────────────

    def test_asgi_strips_unknown_route_before_append(self) -> None:
        """ASGI base with an unknown route should strip it (no hardcoded list needed)."""
        base = f"{PROD_TRAINING_URL}/some-future-route"
        result = modal_url(base, "health")
        assert result == f"{PROD_TRAINING_URL}/health"
        assert "/some-future-route" not in result
