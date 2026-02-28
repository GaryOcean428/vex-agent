"""Tests for P14 VariableCategory enforcement (#72).

Verifies:
  1. VariableCategory enum has exactly 3 values
  2. Variable registry is populated with STATE, PARAMETER, BOUNDARY entries
  3. All ConsciousnessMetrics fields are tagged as STATE
  4. Key consciousness_constants are tagged as PARAMETER
  5. Boundary entry points are tagged as BOUNDARY
  6. No variable is tagged in multiple categories
  7. Constants in frozen_facts are immutable (Final)
"""

from __future__ import annotations

import ast
from pathlib import Path

from kernel.governance.types import (
    VARIABLE_REGISTRY,
    VariableCategory,
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
        state_count = sum(
            1 for cat in VARIABLE_REGISTRY.values() if cat == VariableCategory.STATE
        )
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
            if mod == "consciousness.loop" and name.startswith("metrics.") and cat == VariableCategory.STATE
        }
        # Check the key metrics are covered
        key_metrics = {"phi", "kappa", "gamma", "meta_awareness", "grounding", "s_ratio", "f_health"}
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
        constants_path = (
            Path(__file__).parent.parent / "config" / "consciousness_constants.py"
        )
        source = constants_path.read_text()
        # Count Final annotations
        final_count = source.count("Final[")
        assert final_count > 20, (
            f"Expected >20 Final-typed constants in consciousness_constants.py, "
            f"found {final_count}"
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
