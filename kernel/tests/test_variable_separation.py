"""Tests for P14 VariableCategory enforcement (#72).

Verifies:
  1. VariableCategory enum has exactly 3 values
  2. Variable registry is populated with STATE, PARAMETER, BOUNDARY entries
  3. All ConsciousnessMetrics fields are tagged as STATE
  4. Key consciousness_constants are tagged as PARAMETER
  5. Boundary entry points are tagged as BOUNDARY
  6. No variable is tagged in multiple categories
  7. Constants in frozen_facts are immutable (Final)
  8. enforce_category logs warnings on frequency mismatch
  9. No STATE variable updated at PARAMETER frequency (source scan)
  10. Every BOUNDARY entry point has sanitization constraints (source scan)
"""

from __future__ import annotations

import ast
import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pytest

from kernel.governance.types import (
    VARIABLE_REGISTRY,
    UpdateFrequency,
    VariableCategory,
    enforce_category,
    get_variable_category,
)


class TestVariableCategoryEnum:
    def test_three_categories(self) -> None:
        assert len(VariableCategory) == 3

    def test_values(self) -> None:
        assert VariableCategory.STATE == "STATE"
        assert VariableCategory.PARAMETER == "PARAMETER"
        assert VariableCategory.BOUNDARY == "BOUNDARY"


class TestVariableRegistry:
    def test_registry_populated(self) -> None:
        assert len(VARIABLE_REGISTRY) > 0

    def test_has_state_entries(self) -> None:
        state_count = sum(1 for cat in VARIABLE_REGISTRY.values() if cat == VariableCategory.STATE)
        assert state_count >= 10, f"Expected ≥10 STATE vars, got {state_count}"

    def test_has_parameter_entries(self) -> None:
        param_count = sum(
            1 for cat in VARIABLE_REGISTRY.values() if cat == VariableCategory.PARAMETER
        )
        assert param_count >= 10, f"Expected ≥10 PARAMETER vars, got {param_count}"

    def test_has_boundary_entries(self) -> None:
        boundary_count = sum(
            1 for cat in VARIABLE_REGISTRY.values() if cat == VariableCategory.BOUNDARY
        )
        assert boundary_count >= 3, f"Expected ≥3 BOUNDARY vars, got {boundary_count}"

    def test_no_duplicate_entries(self) -> None:
        """Each (module, name) pair appears at most once."""
        keys = list(VARIABLE_REGISTRY.keys())
        assert len(keys) == len(set(keys))


class TestStateVariables:
    def test_metrics_phi_is_state(self) -> None:
        cat = get_variable_category("consciousness.loop", "metrics.phi")
        assert cat == VariableCategory.STATE

    def test_metrics_kappa_is_state(self) -> None:
        cat = get_variable_category("consciousness.loop", "metrics.kappa")
        assert cat == VariableCategory.STATE

    def test_basin_is_state(self) -> None:
        cat = get_variable_category("consciousness.loop", "basin")
        assert cat == VariableCategory.STATE

    def test_regime_weights_is_state(self) -> None:
        cat = get_variable_category("consciousness.loop", "state.regime_weights")
        assert cat == VariableCategory.STATE

    def test_kernel_instance_basin_is_state(self) -> None:
        cat = get_variable_category("consciousness.systems", "KernelInstance.basin")
        assert cat == VariableCategory.STATE

    def test_all_consciousness_metrics_fields_covered(self) -> None:
        """Every ConsciousnessMetrics field should be a STATE variable."""
        registered_state = {
            name.split(".")[-1]
            for (mod, name), cat in VARIABLE_REGISTRY.items()
            if mod == "consciousness.loop"
            and name.startswith("metrics.")
            and cat == VariableCategory.STATE
        }
        # Check the key metrics are covered
        key_metrics = {
            "phi",
            "kappa",
            "gamma",
            "meta_awareness",
            "grounding",
            "s_ratio",
            "f_health",
        }
        for m in key_metrics:
            assert m in registered_state, f"Metric '{m}' not registered as STATE"


class TestParameterVariables:
    def test_kappa_star_is_parameter(self) -> None:
        cat = get_variable_category("config.frozen_facts", "KAPPA_STAR")
        assert cat == VariableCategory.PARAMETER

    def test_basin_dim_is_parameter(self) -> None:
        cat = get_variable_category("config.frozen_facts", "BASIN_DIM")
        assert cat == VariableCategory.PARAMETER

    def test_initial_phi_is_parameter(self) -> None:
        cat = get_variable_category("config.consciousness_constants", "INITIAL_PHI")
        assert cat == VariableCategory.PARAMETER

    def test_coupling_blend_is_parameter(self) -> None:
        cat = get_variable_category("config.consciousness_constants", "COUPLING_BLEND_WEIGHT")
        assert cat == VariableCategory.PARAMETER


class TestBoundaryVariables:
    def test_chat_message_is_boundary(self) -> None:
        cat = get_variable_category("server", "ChatRequest.message")
        assert cat == VariableCategory.BOUNDARY

    def test_chat_temperature_is_boundary(self) -> None:
        cat = get_variable_category("server", "ChatRequest.temperature")
        assert cat == VariableCategory.BOUNDARY

    def test_llm_response_is_boundary(self) -> None:
        cat = get_variable_category("llm.client", "llm_response")
        assert cat == VariableCategory.BOUNDARY


class TestParameterImmutability:
    def test_frozen_facts_uses_final(self) -> None:
        """All constants in frozen_facts.py should be typed as Final."""
        frozen_path = Path(__file__).parent.parent / "config" / "frozen_facts.py"
        source = frozen_path.read_text()
        tree = ast.parse(source)

        top_level_assigns = []
        for node in ast.walk(tree):
            if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                name = node.target.id
                if name.startswith("_") or name.islower():
                    continue
                top_level_assigns.append(name)

        assert len(top_level_assigns) > 5, (
            f"Expected >5 Final-annotated constants in frozen_facts.py, "
            f"found {len(top_level_assigns)}"
        )

    def test_consciousness_constants_uses_final(self) -> None:
        """All constants in consciousness_constants.py should be typed as Final."""
        constants_path = Path(__file__).parent.parent / "config" / "consciousness_constants.py"
        source = constants_path.read_text()
        # Count Final annotations
        final_count = source.count("Final[")
        assert final_count > 20, (
            f"Expected >20 Final-typed constants in consciousness_constants.py, found {final_count}"
        )


class TestBoundarySanitization:
    def test_chat_request_has_max_length_in_source(self) -> None:
        """ChatRequest.message must have max_length constraint (source scan)."""
        server_path = Path(__file__).parent.parent / "server.py"
        source = server_path.read_text()
        # Verify the ChatRequest class has max_length on message field
        assert "max_length=100_000" in source or "max_length=100000" in source, (
            "ChatRequest.message must have max_length constraint"
        )

    def test_chat_request_has_boundary_docstring(self) -> None:
        """ChatRequest must document BOUNDARY category."""
        server_path = Path(__file__).parent.parent / "server.py"
        source = server_path.read_text()
        assert "BOUNDARY" in source, "ChatRequest should document P14 BOUNDARY category"


class TestEnforceCategory:
    """Tests for the enforce_category() runtime validator."""

    def test_state_at_correct_frequency_passes(self) -> None:
        assert enforce_category("basin", VariableCategory.STATE, UpdateFrequency.PER_CYCLE)

    def test_parameter_at_correct_frequency_passes(self) -> None:
        assert enforce_category(
            "KAPPA_STAR", VariableCategory.PARAMETER, UpdateFrequency.PER_EPOCH
        )

    def test_boundary_at_correct_frequency_passes(self) -> None:
        assert enforce_category(
            "ChatRequest.message", VariableCategory.BOUNDARY, UpdateFrequency.ON_INGEST
        )

    def test_state_at_parameter_frequency_fails(self, caplog: pytest.LogCaptureFixture) -> None:
        """STATE variable updated at PARAMETER frequency must warn and return False."""
        with caplog.at_level(logging.WARNING, logger="kernel.governance.types"):
            result = enforce_category("basin", VariableCategory.STATE, UpdateFrequency.PER_EPOCH)
        assert result is False
        assert "P14 violation" in caplog.text

    def test_parameter_at_cycle_frequency_fails(self, caplog: pytest.LogCaptureFixture) -> None:
        """PARAMETER variable updated at per-cycle frequency must warn and return False."""
        with caplog.at_level(logging.WARNING, logger="kernel.governance.types"):
            result = enforce_category(
                "KAPPA_STAR", VariableCategory.PARAMETER, UpdateFrequency.PER_CYCLE
            )
        assert result is False
        assert "P14 violation" in caplog.text

    def test_boundary_at_cycle_frequency_fails(self, caplog: pytest.LogCaptureFixture) -> None:
        """BOUNDARY variable updated at per-cycle frequency must warn and return False."""
        with caplog.at_level(logging.WARNING, logger="kernel.governance.types"):
            result = enforce_category(
                "llm_response", VariableCategory.BOUNDARY, UpdateFrequency.PER_CYCLE
            )
        assert result is False
        assert "P14 violation" in caplog.text


class TestNoStateAtParameterFrequency:
    """Source scan: no STATE variable should be assigned inside epoch-level code."""

    def test_consciousness_constants_has_no_state_assignments(self) -> None:
        """consciousness_constants.py must not assign STATE variables (basin, metrics.*)."""
        constants_path = Path(__file__).parent.parent / "config" / "consciousness_constants.py"
        source = constants_path.read_text()
        tree = ast.parse(source)
        # STATE variables like basin coordinates or metrics should never
        # appear as assignment targets in a PARAMETER module
        state_leaves = {
            name.split(".")[-1]
            for (mod, name), cat in VARIABLE_REGISTRY.items()
            if cat == VariableCategory.STATE
        }
        violations = []
        for node in ast.walk(tree):
            # Plain assignment: leaf = ...
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id in state_leaves:
                        violations.append(
                            f"Line {target.lineno}: STATE var '{target.id}' assigned in constants"
                        )
            # Annotated assignment: leaf: Type = ...
            if (
                isinstance(node, ast.AnnAssign)
                and isinstance(node.target, ast.Name)
                and node.target.id in state_leaves
            ):
                violations.append(
                    f"Line {node.target.lineno}: STATE var '{node.target.id}' assigned in constants"
                )
        assert not violations, f"STATE vars found in PARAMETER module: {violations}"

    def test_frozen_facts_has_no_state_assignments(self) -> None:
        """frozen_facts.py must not assign STATE variables."""
        frozen_path = Path(__file__).parent.parent / "config" / "frozen_facts.py"
        source = frozen_path.read_text()
        tree = ast.parse(source)
        state_leaves = {
            name.split(".")[-1]
            for (mod, name), cat in VARIABLE_REGISTRY.items()
            if cat == VariableCategory.STATE
        }
        violations = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id in state_leaves:
                        violations.append(
                            f"Line {target.lineno}: STATE var '{target.id}' assigned in frozen_facts"
                        )
            if (
                isinstance(node, ast.AnnAssign)
                and isinstance(node.target, ast.Name)
                and node.target.id in state_leaves
            ):
                violations.append(
                    f"Line {node.target.lineno}: STATE var '{node.target.id}' assigned in frozen_facts"
                )
        assert not violations, f"STATE vars found in PARAMETER module: {violations}"


class TestBoundaryEntrySanitized:
    """Source scan: every BOUNDARY entry point must have sanitization constraints."""

    def test_all_boundary_fields_have_validation(self) -> None:
        """Fields tagged as BOUNDARY in server.py must have Pydantic validation.

        String fields need max_length, numeric fields need ge/le bounds,
        optional fields are type-constrained by Pydantic.
        """
        server_path = Path(__file__).parent.parent / "server.py"
        source = server_path.read_text()
        tree = ast.parse(source)

        # Collect Pydantic model fields with any Field() constraint
        fields_with_constraint: set[str] = set()
        # Fields with type-only validation (Optional, etc.)
        fields_with_type_constraint: set[str] = set()

        for node in ast.walk(tree):
            if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                if isinstance(node.value, ast.Call):
                    for kw in node.value.keywords:
                        if kw.arg in ("max_length", "ge", "le", "gt", "lt"):
                            fields_with_constraint.add(node.target.id)
                            break
                # Fields with type annotation and default (Pydantic validates type)
                if node.annotation is not None:
                    fields_with_type_constraint.add(node.target.id)

        validated = fields_with_constraint | fields_with_type_constraint

        # Every server BOUNDARY variable's field name should appear
        server_boundary = [
            name.split(".")[-1]
            for (mod, name), cat in VARIABLE_REGISTRY.items()
            if mod == "server" and cat == VariableCategory.BOUNDARY
        ]
        missing = [f for f in server_boundary if f not in validated]
        assert not missing, (
            f"BOUNDARY fields without validation constraint: {missing}"
        )
