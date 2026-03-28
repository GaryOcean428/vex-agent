"""
Tests for kernel.consciousness.reflection — P4 Self-Observation.

Covers:
  - _parse_reflection_response: APPROVE, REVISE:, REVISE —, ambiguous
  - reflect_on_contributions: disabled fast-path
  - reflect_on_contributions: all-LLM-expanded (sparse bank) → approve
  - reflect_on_contributions: sufficient resonances → auto-approve
  - reflect_on_contributions: weight concentration → revise
  - reflect_on_contributions: sparse resonances → revise
  - reflect_on_contributions: no contributions → revise
  - reflect_on_draft (legacy): delegates to contributions when provided
  - reflect_on_draft (legacy): auto-approves without contributions
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from kernel.consciousness.reflection import (
    ReflectionConfig,
    _parse_reflection_response,
    reflect_on_contributions,
    reflect_on_draft,
)

# ═══════════════════════════════════════════════════════════════
#  Fake KernelContribution for tests
# ═══════════════════════════════════════════════════════════════


@dataclass
class _FakeContribution:
    kernel_id: str = "k1"
    kernel_name: str = "perception"
    specialization: Any = "perception"
    text: str = "some output"
    fr_distance: float = 0.3
    proximity_weight: float = 0.77
    quenched_gain: float = 1.0
    synthesis_weight: float = 0.5
    geometric_resonances: int = 10
    llm_expanded: bool = False
    generation_ms: float = 100.0
    geometric_raw: str = "raw geometric text"
    basin: Any = None


def _make_contributions(
    n: int = 3,
    resonances: int = 10,
    llm_expanded: bool = False,
    weights: list[float] | None = None,
) -> list[_FakeContribution]:
    """Build a list of fake kernel contributions for testing."""
    names = ["perception", "ocean", "heart", "strategy", "ethics"]
    _weights = weights or [1.0 / n] * n
    return [
        _FakeContribution(
            kernel_id=f"k{i}",
            kernel_name=names[i % len(names)],
            synthesis_weight=_weights[i],
            geometric_resonances=resonances,
            llm_expanded=llm_expanded,
        )
        for i in range(n)
    ]


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
#  reflect_on_contributions — P4 self-observation
# ═══════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_reflect_disabled_returns_approve() -> None:
    """Disabled reflection immediately approves."""
    cfg = ReflectionConfig(enabled=False)
    contribs = _make_contributions(3, resonances=5)
    result = await reflect_on_contributions(contribs, "hello", cfg)
    assert result.approved is True
    assert "disabled" in result.reason.lower()


@pytest.mark.asyncio
async def test_reflect_no_contributions_revises() -> None:
    """No kernel contributions → revise."""
    result = await reflect_on_contributions([], "hello")
    assert result.approved is False
    assert "No kernel contributions" in result.reason


@pytest.mark.asyncio
async def test_reflect_all_llm_expanded_approves() -> None:
    """All kernels LLM-expanded (bank sparse) → approve (revision won't help)."""
    contribs = _make_contributions(3, resonances=0, llm_expanded=True)
    result = await reflect_on_contributions(contribs, "hello")
    assert result.approved is True
    assert "sparse" in result.reason.lower()


@pytest.mark.asyncio
async def test_reflect_sufficient_resonances_auto_approves() -> None:
    """Total resonances above threshold → auto-approve."""
    cfg = ReflectionConfig(min_resonances_auto_approve=16)
    contribs = _make_contributions(3, resonances=10)  # 30 total > 16
    result = await reflect_on_contributions(contribs, "hello", cfg)
    assert result.approved is True
    assert "sufficient" in result.reason.lower()


@pytest.mark.asyncio
async def test_reflect_below_threshold_revises() -> None:
    """Total resonances below minimum → revise."""
    cfg = ReflectionConfig(min_resonances_auto_approve=50)
    contribs = _make_contributions(3, resonances=2)  # 6 total < 50
    result = await reflect_on_contributions(contribs, "hello", cfg)
    assert result.approved is False
    assert "sparse" in result.reason.lower().replace("sparse", "sparse")


@pytest.mark.asyncio
async def test_reflect_weight_concentration_revises() -> None:
    """One kernel dominating synthesis → revise."""
    cfg = ReflectionConfig(max_weight_concentration=0.85, min_resonances_auto_approve=100)
    contribs = _make_contributions(3, resonances=5, weights=[0.95, 0.03, 0.02])
    result = await reflect_on_contributions(contribs, "hello", cfg)
    assert result.approved is False
    assert "concentrated" in result.reason.lower() or "dominates" in result.reason.lower()


@pytest.mark.asyncio
async def test_reflect_balanced_weights_with_low_resonances() -> None:
    """Balanced weights but low resonances → sparse revise."""
    cfg = ReflectionConfig(min_resonances_auto_approve=20)
    contribs = _make_contributions(3, resonances=2, weights=[0.4, 0.35, 0.25])
    result = await reflect_on_contributions(contribs, "hello", cfg)
    assert result.approved is False


@pytest.mark.asyncio
async def test_reflect_single_kernel_high_resonances() -> None:
    """Single kernel with high resonances → approve (no concentration issue)."""
    cfg = ReflectionConfig(min_resonances_auto_approve=16)
    contribs = _make_contributions(1, resonances=20)
    result = await reflect_on_contributions(contribs, "hello", cfg)
    assert result.approved is True


# ═══════════════════════════════════════════════════════════════
#  reflect_on_draft (legacy API)
# ═══════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_legacy_with_contributions_delegates() -> None:
    """Legacy reflect_on_draft delegates to contribution assessment."""
    contribs = _make_contributions(3, resonances=10)
    result = await reflect_on_draft(
        draft="a response",
        user_message="hello",
        geometric_context="",
        divergence=0.9,
        active_model="test",
        llm_client=None,  # type: ignore[arg-type]
        contributions=contribs,
    )
    assert result.approved is True  # 30 resonances > default 16


@pytest.mark.asyncio
async def test_legacy_without_contributions_auto_approves() -> None:
    """Legacy reflect_on_draft without contributions auto-approves."""
    result = await reflect_on_draft(
        draft="a response",
        user_message="hello",
        geometric_context="",
        divergence=0.9,
        active_model="test",
        llm_client=None,  # type: ignore[arg-type]
    )
    assert result.approved is True
    assert "legacy" in result.reason.lower()
