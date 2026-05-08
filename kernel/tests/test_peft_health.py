"""Tests for PEFT health check and availability gating in peft_client.py."""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from kernel.llm.peft_client import PeftInferenceClient


def _make_client(base_url: str = "https://example--app-web.modal.run/infer") -> PeftInferenceClient:
    return PeftInferenceClient(base_url=base_url, timeout_ms=5000, api_key="test-key")


class TestHealthCheck:
    """Tests for check_health() status matching."""

    def test_healthy_status_accepted(self) -> None:
        client = _make_client()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"status": "healthy", "specializations": ["genesis"]}

        async def _run():
            client._http = AsyncMock()
            client._http.get = AsyncMock(return_value=mock_resp)
            result = await client.check_health()
            assert result is True
            assert client.available is True

        asyncio.get_event_loop().run_until_complete(_run())

    def test_ok_status_rejected(self) -> None:
        """The old bug: Modal returned 'ok', client expected 'healthy'."""
        client = _make_client()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"status": "ok"}

        async def _run():
            client._http = AsyncMock()
            client._http.get = AsyncMock(return_value=mock_resp)
            result = await client.check_health()
            assert result is False
            assert client.available is False

        asyncio.get_event_loop().run_until_complete(_run())

    def test_non_200_marks_unavailable(self) -> None:
        client = _make_client()
        mock_resp = MagicMock()
        mock_resp.status_code = 503

        async def _run():
            client._http = AsyncMock()
            client._http.get = AsyncMock(return_value=mock_resp)
            result = await client.check_health()
            assert result is False
            assert client.available is False

        asyncio.get_event_loop().run_until_complete(_run())

    def test_connection_error_marks_unavailable(self) -> None:
        import httpx

        client = _make_client()

        async def _run():
            client._http = AsyncMock()
            client._http.get = AsyncMock(side_effect=httpx.ConnectError("refused"))
            result = await client.check_health()
            assert result is False
            assert client.available is False

        asyncio.get_event_loop().run_until_complete(_run())


class TestEnsureAvailable:
    """Tests for _ensure_available() caching and retry logic."""

    def test_caches_healthy_result(self) -> None:
        client = _make_client()
        client._available = True
        client._last_check = time.monotonic()

        async def _run():
            result = await client._ensure_available()
            assert result is True

        asyncio.get_event_loop().run_until_complete(_run())

    def test_caches_unhealthy_within_retry_interval(self) -> None:
        client = _make_client()
        client._available = False
        client._last_check = time.monotonic()

        async def _run():
            result = await client._ensure_available()
            assert result is False

        asyncio.get_event_loop().run_until_complete(_run())

    def test_rechecks_after_retry_interval(self) -> None:
        client = _make_client()
        client._available = False
        client._last_check = time.monotonic() - 20.0  # Past the 15s retry interval

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"status": "healthy", "specializations": []}

        async def _run():
            client._http = AsyncMock()
            client._http.get = AsyncMock(return_value=mock_resp)
            result = await client._ensure_available()
            assert result is True
            assert client.available is True

        asyncio.get_event_loop().run_until_complete(_run())


class TestCompleteAvailabilityGate:
    """Tests that complete() checks availability before hitting the endpoint."""

    def test_skips_inference_when_unavailable(self) -> None:
        client = _make_client()
        client._available = False
        client._last_check = time.monotonic()

        async def _run():
            result = await client.complete(
                system_prompt="test",
                user_message="hello",
                specialization="genesis",
            )
            assert result.success is False
            assert "unavailable" in result.error

        asyncio.get_event_loop().run_until_complete(_run())

    def test_proceeds_when_available(self) -> None:
        client = _make_client()
        client._available = True
        client._last_check = time.monotonic()

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "text": "response text",
            "specialization": "genesis",
            "adapter_loaded": "genesis",
            "inference_tier": 2,
            "tokens_generated": 10,
            "latency_ms": 50.0,
        }

        async def _run():
            client._http = AsyncMock()
            client._http.post = AsyncMock(return_value=mock_resp)
            result = await client.complete(
                system_prompt="test",
                user_message="hello",
                specialization="genesis",
            )
            assert result.success is True
            assert result.text == "response text"

        asyncio.get_event_loop().run_until_complete(_run())
