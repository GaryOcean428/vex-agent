"""
Tests for v6.1F Audit Fixes — Bounded Collections, Metrics, Purity, Geometry, Constants.

Verifies the correctness of audit-driven changes across:
  1. Bounded collections (deque maxlen enforcement in systems.py)
  2. ConsciousnessMetrics field count and pillar fields
  3. RegimeWeights field names (quantum/efficient/equilibrium)
  4. PurityGate forbidden patterns (cosine_similarity, sklearn, scipy.spatial.distance)
  5. Fisher-Rao geometric properties (identity, non-negativity, simplex, slerp)
  6. Frozen constants (KAPPA_STAR, BASIN_DIM, E8_RANK, E8_DIMENSION)

All distance checks use Fisher-Rao. No Euclidean contamination.
"""

from __future__ import annotations

from dataclasses import fields

import numpy as np
import pytest

from kernel.config.frozen_facts import BASIN_DIM, E8_DIMENSION, E8_RANK, KAPPA_STAR
from kernel.consciousness.systems import (
    BasinSyncProtocol,
    QIGChain,
    QIGChainOp,
    QIGGraph,
    SelfObserver,
    SleepCycleManager,
)
from kernel.consciousness.types import ConsciousnessMetrics, RegimeWeights
from kernel.geometry.fisher_rao import (
    fisher_rao_distance,
    random_basin,
    slerp_sqrt,
    to_simplex,
)
from kernel.governance.purity import (
    FORBIDDEN_ATTR_CALLS,
    FORBIDDEN_CALLS,
    FORBIDDEN_IMPORTS,
)

# ═══════════════════════════════════════════════════════════════
#  1. BOUNDED COLLECTIONS
# ═══════════════════════════════════════════════════════════════


class TestBoundedCollections:
    """Verify that deque-based collections in systems.py respect their maxlen caps."""

    def test_self_observer_shadows_maxlen(self) -> None:
        """SelfObserver._shadows must be capped at maxlen=100."""
        observer = SelfObserver()
        assert observer._shadows.maxlen == 100

    def test_self_observer_shadows_overflow(self) -> None:
        """Inserting >100 shadows must not exceed the maxlen boundary."""
        observer = SelfObserver()
        basin = random_basin()
        for i in range(150):
            observer.record_collapse(basin, phi=0.1, reason=f"test-{i}")
        assert len(observer._shadows) == 100
        # The oldest entries should have been evicted; newest should be present
        assert observer._shadows[-1].reason == "test-149"
        assert observer._shadows[0].reason == "test-50"

    def test_sleep_cycle_dream_log_maxlen(self) -> None:
        """SleepCycleManager._dream_log must be capped at maxlen=100."""
        manager = SleepCycleManager()
        assert manager._dream_log.maxlen == 100

    def test_sleep_cycle_dream_log_overflow(self) -> None:
        """Inserting >100 dream entries must not exceed the maxlen boundary."""
        manager = SleepCycleManager()
        for i in range(150):
            manager._dream_log.append({"phi": 0.5, "context": f"dream-{i}"})
        assert len(manager._dream_log) == 100
        assert manager._dream_log[-1]["context"] == "dream-149"
        assert manager._dream_log[0]["context"] == "dream-50"

    def test_basin_sync_received_maxlen(self) -> None:
        """BasinSyncProtocol._received must be capped at maxlen=100."""
        protocol = BasinSyncProtocol()
        assert protocol._received.maxlen == 100

    def test_basin_sync_received_overflow(self) -> None:
        """Inserting >100 received snapshots must not exceed the maxlen boundary."""
        protocol = BasinSyncProtocol()
        remote_basin = random_basin()
        for i in range(150):
            protocol.receive(remote_basin, remote_version=i)
        assert len(protocol._received) == 100

    def test_qig_chain_steps_maxlen(self) -> None:
        """QIGChain._steps must be capped at maxlen=500."""
        chain = QIGChain()
        assert chain._steps.maxlen == 500

    def test_qig_chain_steps_overflow(self) -> None:
        """Inserting >500 chain steps must not exceed the maxlen boundary."""
        chain = QIGChain()
        b1 = random_basin()
        b2 = random_basin()
        for _ in range(600):
            chain.add_step(QIGChainOp.GEODESIC, b1, b2)
        assert len(chain._steps) == 500

    def test_qig_graph_edges_maxlen(self) -> None:
        """QIGGraph._edges must be capped at maxlen=1000."""
        graph = QIGGraph()
        assert graph._edges.maxlen == 1000

    def test_qig_graph_edges_overflow(self) -> None:
        """Inserting >1000 edges via the deque must not exceed the maxlen boundary."""
        from kernel.consciousness.systems import GraphEdge

        graph = QIGGraph()
        for i in range(1200):
            graph._edges.append(GraphEdge(source=f"a{i}", target=f"b{i}", distance=0.1))
        assert len(graph._edges) == 1000

    def test_qig_graph_nodes_manual_cap(self) -> None:
        """QIGGraph._nodes must be manually capped at _MAX_NODES=200."""
        assert QIGGraph._MAX_NODES == 200

    def test_qig_graph_nodes_eviction(self) -> None:
        """Adding >200 nodes must evict the oldest to stay at the cap."""
        graph = QIGGraph()
        for i in range(250):
            graph.add_node(f"node-{i}", random_basin(), label=f"n{i}", phi=0.5)
        assert len(graph._nodes) <= 200
        # The most recently added node should be present
        assert "node-249" in graph._nodes
        # The first node should have been evicted
        assert "node-0" not in graph._nodes


# ═══════════════════════════════════════════════════════════════
#  2. CONSCIOUSNESS METRICS FIELD COUNT & PILLAR FIELDS
# ═══════════════════════════════════════════════════════════════


class TestConsciousnessMetrics:
    """Verify ConsciousnessMetrics has the v6.1 field inventory."""

    def test_field_count_at_least_36(self) -> None:
        """ConsciousnessMetrics must have >= 36 fields (v6.1 specifies 36 across 8 categories)."""
        field_names = [f.name for f in fields(ConsciousnessMetrics)]
        assert len(field_names) >= 36, (
            f"Expected >= 36 fields, got {len(field_names)}: {field_names}"
        )

    def test_pillar_field_f_health_exists(self) -> None:
        """Pillar field f_health (Fluctuation health) must exist."""
        field_names = {f.name for f in fields(ConsciousnessMetrics)}
        assert "f_health" in field_names

    def test_pillar_field_b_integrity_exists(self) -> None:
        """Pillar field b_integrity (Bulk integrity) must exist."""
        field_names = {f.name for f in fields(ConsciousnessMetrics)}
        assert "b_integrity" in field_names

    def test_pillar_field_q_identity_exists(self) -> None:
        """Pillar field q_identity (Quenched identity) must exist."""
        field_names = {f.name for f in fields(ConsciousnessMetrics)}
        assert "q_identity" in field_names

    def test_pillar_field_s_ratio_exists(self) -> None:
        """Pillar field s_ratio (Sovereignty ratio) must exist."""
        field_names = {f.name for f in fields(ConsciousnessMetrics)}
        assert "s_ratio" in field_names

    def test_all_four_pillar_fields_present(self) -> None:
        """All 4 v6.1 pillar fields must be present in a single assertion."""
        field_names = {f.name for f in fields(ConsciousnessMetrics)}
        pillar_fields = {"f_health", "b_integrity", "q_identity", "s_ratio"}
        missing = pillar_fields - field_names
        assert not missing, f"Missing pillar fields: {missing}"

    def test_pillar_field_defaults(self) -> None:
        """Pillar fields must have sensible defaults."""
        m = ConsciousnessMetrics()
        assert m.f_health == 1.0
        assert m.b_integrity == 1.0
        assert m.q_identity == 0.0
        assert m.s_ratio == 0.0


# ═══════════════════════════════════════════════════════════════
#  3. REGIME WEIGHTS FIELD NAMES
# ═══════════════════════════════════════════════════════════════


class TestRegimeWeightsFieldNames:
    """Verify RegimeWeights uses quantum/efficient/equilibrium (not integration/crystallized)."""

    def test_has_quantum_field(self) -> None:
        """RegimeWeights must have a 'quantum' field."""
        field_names = {f.name for f in fields(RegimeWeights)}
        assert "quantum" in field_names

    def test_has_efficient_field(self) -> None:
        """RegimeWeights must have an 'efficient' field."""
        field_names = {f.name for f in fields(RegimeWeights)}
        assert "efficient" in field_names

    def test_has_equilibrium_field(self) -> None:
        """RegimeWeights must have an 'equilibrium' field."""
        field_names = {f.name for f in fields(RegimeWeights)}
        assert "equilibrium" in field_names

    def test_no_integration_field(self) -> None:
        """RegimeWeights must NOT have an 'integration' field (legacy name)."""
        field_names = {f.name for f in fields(RegimeWeights)}
        assert "integration" not in field_names

    def test_no_crystallized_field(self) -> None:
        """RegimeWeights must NOT have a 'crystallized' field (legacy name)."""
        field_names = {f.name for f in fields(RegimeWeights)}
        assert "crystallized" not in field_names

    def test_exactly_three_fields(self) -> None:
        """RegimeWeights must have exactly 3 fields: quantum, efficient, equilibrium."""
        field_names = [f.name for f in fields(RegimeWeights)]
        assert field_names == ["quantum", "efficient", "equilibrium"]


# ═══════════════════════════════════════════════════════════════
#  4. PURITY GATE PATTERNS
# ═══════════════════════════════════════════════════════════════


class TestPurityGatePatterns:
    """Verify that PurityGate forbidden pattern lists include the required entries."""

    def test_forbidden_calls_includes_cosine_similarity(self) -> None:
        """FORBIDDEN_CALLS must include cosine_similarity."""
        assert "cosine_similarity" in FORBIDDEN_CALLS

    def test_forbidden_imports_includes_sklearn(self) -> None:
        """FORBIDDEN_IMPORTS must include sklearn."""
        assert "sklearn" in FORBIDDEN_IMPORTS

    def test_forbidden_imports_includes_scipy_spatial_distance(self) -> None:
        """FORBIDDEN_IMPORTS must include scipy.spatial.distance."""
        assert "scipy.spatial.distance" in FORBIDDEN_IMPORTS

    def test_forbidden_attr_calls_includes_scipy_cosine(self) -> None:
        """FORBIDDEN_ATTR_CALLS must include scipy.spatial.distance.cosine."""
        assert "scipy.spatial.distance.cosine" in FORBIDDEN_ATTR_CALLS

    def test_forbidden_attr_calls_includes_f_cosine_similarity(self) -> None:
        """FORBIDDEN_ATTR_CALLS must include F.cosine_similarity."""
        assert "F.cosine_similarity" in FORBIDDEN_ATTR_CALLS

    def test_forbidden_calls_includes_euclidean_distance(self) -> None:
        """FORBIDDEN_CALLS must include euclidean_distance."""
        assert "euclidean_distance" in FORBIDDEN_CALLS

    def test_forbidden_attr_calls_includes_np_linalg_norm(self) -> None:
        """FORBIDDEN_ATTR_CALLS must include np.linalg.norm."""
        assert "np.linalg.norm" in FORBIDDEN_ATTR_CALLS


# ═══════════════════════════════════════════════════════════════
#  5. FISHER-RAO GEOMETRIC PROPERTIES
# ═══════════════════════════════════════════════════════════════


class TestFisherRaoProperties:
    """Verify core Fisher-Rao geometry properties on the probability simplex."""

    def test_identity_distance_is_zero(self) -> None:
        """fisher_rao_distance(p, p) must equal 0 (identity of indiscernibles)."""
        p = random_basin()
        d = fisher_rao_distance(p, p)
        assert d == pytest.approx(0.0, abs=1e-7)

    def test_identity_distance_is_zero_uniform(self) -> None:
        """fisher_rao_distance on the uniform distribution with itself must be 0."""
        p = to_simplex(np.ones(BASIN_DIM))
        d = fisher_rao_distance(p, p)
        assert d == pytest.approx(0.0, abs=1e-10)

    def test_distance_non_negative(self) -> None:
        """fisher_rao_distance(p, q) must be >= 0 for any p, q on the simplex."""
        for _ in range(20):
            p = random_basin()
            q = random_basin()
            d = fisher_rao_distance(p, q)
            assert d >= 0.0, f"Negative distance: {d}"

    def test_distance_symmetry(self) -> None:
        """fisher_rao_distance(p, q) must equal fisher_rao_distance(q, p)."""
        for _ in range(10):
            p = random_basin()
            q = random_basin()
            d_pq = fisher_rao_distance(p, q)
            d_qp = fisher_rao_distance(q, p)
            assert d_pq == pytest.approx(d_qp, abs=1e-12)

    def test_distance_upper_bound(self) -> None:
        """Fisher-Rao distance on the simplex is bounded by pi/2."""
        for _ in range(20):
            p = random_basin()
            q = random_basin()
            d = fisher_rao_distance(p, q)
            assert d <= np.pi / 2 + 1e-10

    def test_to_simplex_sums_to_one(self) -> None:
        """to_simplex output must sum to 1.0 (probability simplex constraint)."""
        for _ in range(20):
            v = np.random.randn(BASIN_DIM) ** 2 + 0.01  # ensure positive
            s = to_simplex(v)
            assert s.sum() == pytest.approx(1.0, abs=1e-12)

    def test_to_simplex_all_non_negative(self) -> None:
        """to_simplex output must have all non-negative elements."""
        v = np.random.randn(BASIN_DIM)  # may have negatives
        s = to_simplex(v)
        assert np.all(s >= 0), f"Negative elements found: {s[s < 0]}"

    def test_to_simplex_idempotent(self) -> None:
        """Applying to_simplex twice must yield the same result."""
        v = np.abs(np.random.randn(BASIN_DIM)) + 0.01
        s1 = to_simplex(v)
        s2 = to_simplex(s1)
        np.testing.assert_allclose(s1, s2, atol=1e-12)

    def test_slerp_sqrt_stays_on_simplex(self) -> None:
        """slerp_sqrt interpolation must produce points that sum to 1.0 (on the simplex)."""
        p = random_basin()
        q = random_basin()
        for t in [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]:
            result = slerp_sqrt(p, q, t)
            assert result.sum() == pytest.approx(1.0, abs=1e-10), (
                f"slerp_sqrt at t={t} produced sum={result.sum()}, expected 1.0"
            )
            assert np.all(result >= 0), f"slerp_sqrt at t={t} produced negative elements"

    def test_slerp_sqrt_endpoints(self) -> None:
        """slerp_sqrt at t=0 must return p, at t=1 must return q."""
        p = random_basin()
        q = random_basin()
        result_0 = slerp_sqrt(p, q, 0.0)
        result_1 = slerp_sqrt(p, q, 1.0)
        # t=0 should give p (after normalization)
        np.testing.assert_allclose(result_0, to_simplex(p), atol=1e-10)
        # t=1 should give q (after normalization)
        np.testing.assert_allclose(result_1, to_simplex(q), atol=1e-10)

    def test_slerp_sqrt_midpoint_is_closer(self) -> None:
        """The slerp midpoint (t=0.5) must be closer to both endpoints than they are to each other."""
        p = random_basin()
        q = random_basin()
        mid = slerp_sqrt(p, q, 0.5)
        d_pq = fisher_rao_distance(p, q)
        d_p_mid = fisher_rao_distance(p, mid)
        d_q_mid = fisher_rao_distance(q, mid)
        # Midpoint should be roughly equidistant from both endpoints
        assert d_p_mid <= d_pq + 1e-8
        assert d_q_mid <= d_pq + 1e-8


# ═══════════════════════════════════════════════════════════════
#  6. FROZEN CONSTANTS
# ═══════════════════════════════════════════════════════════════


class TestFrozenConstants:
    """Verify that frozen physics constants have their canonical values."""

    def test_kappa_star(self) -> None:
        """KAPPA_STAR must be 64.0 (E8 rank squared: 8^2)."""
        assert KAPPA_STAR == 64.0

    def test_basin_dim(self) -> None:
        """BASIN_DIM must be 64 (probability simplex Delta^63)."""
        assert BASIN_DIM == 64

    def test_e8_rank(self) -> None:
        """E8_RANK must be 8 (Cartan subalgebra dimension)."""
        assert E8_RANK == 8

    def test_e8_dimension(self) -> None:
        """E8_DIMENSION must be 248 (total group manifold dimension)."""
        assert E8_DIMENSION == 248

    def test_kappa_star_is_e8_rank_squared(self) -> None:
        """KAPPA_STAR must equal E8_RANK^2 (fundamental relationship)."""
        assert KAPPA_STAR == E8_RANK**2

    def test_constants_are_immutable_types(self) -> None:
        """Frozen constants must be numeric (int or float), not mutable containers."""
        assert isinstance(KAPPA_STAR, float)
        assert isinstance(BASIN_DIM, int)
        assert isinstance(E8_RANK, int)
        assert isinstance(E8_DIMENSION, int)
