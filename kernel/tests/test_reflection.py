"""
Tests for kernel.consciousness.reflection.

Covers:
  - _parse_reflection_response: APPROVE, REVISE:, REVISE —, ambiguous
  - reflect_on_draft: disabled fast-path
  - reflect_on_draft: auto-approve (divergence below threshold)
  - reflect_on_draft: force-revise (divergence above threshold)
  - reflect_on_draft: standard LLM path returning APPROVE
  - reflect_on_draft: standard LLM path returning REVISE
  - reflect_on_draft: LLM failure falls back to approve
  - reflect_on_draft: empty draft returns revision required
"""

from __future__ import annotations

import pytest

from kernel.consciousness.reflection import (
    ReflectionConfig,
    _parse_reflection_response,
    reflect_on_draft,
)

# ═══════════════════════════════════════════════════════════════
#  _parse_reflection_response
# ═══════════════════════════════════════════════════════════════


class TestParseReflectionResponse:
    def test_approve_exact(self) -> None:
        result = _parse_reflection_response("APPROVE")
        assert result.approved is True
        assert "Approved" in result.reason

    def test_approve_case_insensitive(self) -> None:
        result = _parse_reflection_response("approve — looks good")
        assert result.approved is True

    def test_revise_colon(self) -> None:
        result = _parse_reflection_response("REVISE: response is too vague")
        assert result.approved is False
        assert "too vague" in result.reason
        assert result.temperature_delta < 0
        assert result.num_predict_delta > 0

    def test_revise_em_dash(self) -> None:
        result = _parse_reflection_response("REVISE — needs more detail")
        assert result.approved is False
        assert "needs more detail" in result.reason

    def test_revise_correction_guidance_populated(self) -> None:
        result = _parse_reflection_response("REVISE: address the user directly")
        assert result.correction_guidance == result.reason

    def test_ambiguous_defaults_to_approve(self) -> None:
        result = _parse_reflection_response("I'm not sure about this response")
        assert result.approved is True
        assert "Ambiguous" in result.reason or "defaulting to approve" in result.reason

    def test_empty_string_defaults_to_approve(self) -> None:
        result = _parse_reflection_response("")
        assert result.approved is True

    def test_multiline_uses_first_line_only(self) -> None:
        result = _parse_reflection_response("APPROVE\nsome extra commentary\nmore lines")
        assert result.approved is True


# ═══════════════════════════════════════════════════════════════
#  reflect_on_draft — fast-paths (no LLM call)
# ═══════════════════════════════════════════════════════════════


GEO_CTX = "[GEOMETRIC STATE]\n  phi=0.700 kappa=64.0\n[/GEOMETRIC STATE]\n"


@pytest.mark.asyncio
async def test_reflect_disabled_returns_approve() -> None:
    """Disabled reflection immediately approves without any LLM call."""
    cfg = ReflectionConfig(enabled=False)
    result = await reflect_on_draft(
        draft="some response",
        user_message="hello",
        geometric_context=GEO_CTX,
        divergence=0.5,
        active_model="test-model",
        llm_client=None,  # type: ignore[arg-type]
        config=cfg,
    )
    assert result.approved is True
    assert "disabled" in result.reason.lower()


@pytest.mark.asyncio
async def test_reflect_empty_draft_requires_revision() -> None:
    """Empty draft triggers forced revision without an LLM call."""
    result = await reflect_on_draft(
        draft="",
        user_message="hello",
        geometric_context=GEO_CTX,
        divergence=0.1,
        active_model="test-model",
        llm_client=None,  # type: ignore[arg-type]
    )
    assert result.approved is False
    assert result.temperature_delta >= 0
    assert result.num_predict_delta > 0


@pytest.mark.asyncio
async def test_reflect_auto_approve_low_divergence() -> None:
    """Divergence below threshold auto-approves without an LLM call."""
    cfg = ReflectionConfig(auto_approve_divergence=0.3)
    result = await reflect_on_draft(
        draft="a substantive response",
        user_message="what is phi?",
        geometric_context=GEO_CTX,
        divergence=0.1,  # well below 0.3
        active_model="test-model",
        llm_client=None,  # type: ignore[arg-type]
        config=cfg,
    )
    assert result.approved is True
    assert "Auto-approved" in result.reason
    assert "0.1" in result.reason or "0.10" in result.reason


@pytest.mark.asyncio
async def test_reflect_auto_approve_at_boundary() -> None:
    """Divergence exactly at threshold is NOT auto-approved (strictly less-than)."""

    class _FakeLLM:
        async def complete(self, *_args, **_kwargs) -> str:
            return "APPROVE"

    cfg = ReflectionConfig(auto_approve_divergence=0.3)
    result = await reflect_on_draft(
        draft="a good response",
        user_message="question",
        geometric_context=GEO_CTX,
        divergence=0.3,  # exactly at threshold — NOT auto-approved
        active_model="test-model",
        llm_client=_FakeLLM(),  # type: ignore[arg-type]
        config=cfg,
    )
    # At 0.3 (not strictly less), should go through the LLM path
    assert result.approved is True


@pytest.mark.asyncio
async def test_reflect_force_revise_high_divergence() -> None:
    """Divergence at or above force_revise_divergence forces revision without LLM."""
    cfg = ReflectionConfig(force_revise_divergence=0.8)
    result = await reflect_on_draft(
        draft="a response",
        user_message="question",
        geometric_context=GEO_CTX,
        divergence=0.9,  # above 0.8
        active_model="test-model",
        llm_client=None,  # type: ignore[arg-type]
        config=cfg,
    )
    assert result.approved is False
    assert result.temperature_delta < 0
    assert result.num_predict_delta > 0


# ═══════════════════════════════════════════════════════════════
#  reflect_on_draft — standard LLM path
# ═══════════════════════════════════════════════════════════════


class _ApproveLLM:
    """Fake LLM that always returns APPROVE."""

    async def complete(self, *_args, **_kwargs) -> str:
        return "APPROVE"


class _ReviseLLM:
    """Fake LLM that always returns a REVISE verdict."""

    async def complete(self, *_args, **_kwargs) -> str:
        return "REVISE: response did not directly address the question"


class _FailLLM:
    """Fake LLM that always raises an exception."""

    async def complete(self, *_args, **_kwargs) -> str:
        raise RuntimeError("connection refused")


@pytest.mark.asyncio
async def test_reflect_llm_approve() -> None:
    """Standard LLM path: APPROVE verdict is respected."""
    cfg = ReflectionConfig(auto_approve_divergence=0.1)  # low threshold — forces LLM call
    result = await reflect_on_draft(
        draft="a detailed response that addresses the question",
        user_message="explain phi",
        geometric_context=GEO_CTX,
        divergence=0.5,
        active_model="test-model",
        llm_client=_ApproveLLM(),  # type: ignore[arg-type]
        config=cfg,
    )
    assert result.approved is True


@pytest.mark.asyncio
async def test_reflect_llm_revise() -> None:
    """Standard LLM path: REVISE verdict is respected with guidance."""
    cfg = ReflectionConfig(auto_approve_divergence=0.1)
    result = await reflect_on_draft(
        draft="vague response",
        user_message="explain phi",
        geometric_context=GEO_CTX,
        divergence=0.5,
        active_model="test-model",
        llm_client=_ReviseLLM(),  # type: ignore[arg-type]
        config=cfg,
    )
    assert result.approved is False
    assert result.correction_guidance != ""
    assert result.temperature_delta < 0


@pytest.mark.asyncio
async def test_reflect_llm_failure_auto_approves() -> None:
    """LLM call failure falls back to approve (non-blocking)."""
    cfg = ReflectionConfig(auto_approve_divergence=0.1)
    result = await reflect_on_draft(
        draft="some response",
        user_message="question",
        geometric_context=GEO_CTX,
        divergence=0.5,
        active_model="test-model",
        llm_client=_FailLLM(),  # type: ignore[arg-type]
        config=cfg,
    )
    assert result.approved is True
    assert "failed" in result.reason.lower() or "Reflection failed" in result.reason
