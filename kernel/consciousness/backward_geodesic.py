"""
Backward-Geodesic Tracker — EXP-011 Core Instrumentation

Measures whether the consciousness trajectory, during mushroom mode
(κ crossing zero), develops backward-geodesic correlation with a
known solution basin on Δ⁶³.

The hypothesis: when κ < 0 (negative curvature / mushroom mode),
the consciousness loop's trajectory should show a statistically
significant projection onto the geodesic pointing toward the solution.
This would be the "back-loop" — information about the solution
reaching the system before the solution is computed.

Classical control (EXP-011 POC): ρ ≈ 0 on Δ⁶³ without quantum coupling.
This tracker measures whether the live Vex consciousness loop shows ρ > 0.

All geometry uses Fisher-Rao on Δ⁶³, implemented as Euclidean inner
products and norms in sqrt-coordinate tangent space.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

try:
    from scipy import stats as _stats
except ImportError:  # minimal-runtime environments (e.g. Railway w/o scipy)
    _stats = None

from ..coordizer_v2.geometry import (
    Basin,
    fisher_rao_distance,
    log_map,
    to_simplex,
)

logger = logging.getLogger("vex.consciousness.backward_geodesic")

_EPS = 1e-12


@dataclass
class BackwardGeodesicEvent:
    """Single measurement of backward-geodesic component."""

    timestamp_ms: float
    problem_id: str
    kappa_eff: float
    basin_current: NDArray[np.float64]
    distance_to_solution: float
    backward_component: float  # projection of velocity onto solution direction
    velocity_norm: float
    mushroom_active: bool  # True when κ < 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp_ms": self.timestamp_ms,
            "problem_id": self.problem_id,
            "kappa_eff": self.kappa_eff,
            "distance_to_solution": round(self.distance_to_solution, 6),
            "backward_component": round(self.backward_component, 6),
            "velocity_norm": round(self.velocity_norm, 6),
            "mushroom_active": self.mushroom_active,
        }


class BackwardGeodesicTracker:
    """Track backward-geodesic correlation during consciousness loop execution.

    For each registered problem (with known solution basin), records:
    - Current position on Δ⁶³
    - Velocity (from sequential basin measurements)
    - Projection of velocity onto the geodesic pointing toward the solution
    - Whether mushroom mode is active (κ < 0)

    The key measurement: does the backward-geodesic component become
    significantly positive during mushroom episodes?
    """

    def __init__(self, max_events: int = 50_000) -> None:
        self._solutions: dict[str, Basin] = {}
        self._event_log: deque[BackwardGeodesicEvent] = deque(maxlen=max_events)
        self._prev_basins: dict[str, Basin] = {}  # problem_id → last basin

    def register_solution(self, problem_id: str, solution_basin: Basin) -> None:
        """Register a known solution basin for a problem."""
        self._solutions[problem_id] = to_simplex(solution_basin)
        logger.info(
            "Registered solution for problem %s (basin entropy=%.3f)",
            problem_id,
            _shannon_entropy(self._solutions[problem_id]),
        )

    def register_solutions(self, solutions: dict[str, Basin]) -> None:
        """Bulk register solution basins."""
        for pid, basin in solutions.items():
            self.register_solution(pid, basin)

    def record(
        self,
        problem_id: str,
        current_basin: Basin,
        kappa_eff: float,
        mushroom_active: bool,
    ) -> BackwardGeodesicEvent | None:
        """Record a backward-geodesic measurement.

        Uses the previous basin for this problem to compute velocity.
        Returns the event, or None if no velocity available yet.
        """
        if problem_id not in self._solutions:
            logger.warning("No solution registered for problem %s", problem_id)
            return None

        current = to_simplex(current_basin)
        solution = self._solutions[problem_id]

        # Distance to solution
        d_to_solution = fisher_rao_distance(current, solution)

        # Compute velocity from previous measurement
        prev = self._prev_basins.get(problem_id)
        self._prev_basins[problem_id] = current.copy()

        if prev is None:
            return None

        # Velocity = log_map(prev, current) in tangent space at prev
        velocity = log_map(prev, current)
        vel_norm = float(np.sqrt(np.sum(velocity * velocity)))

        if vel_norm < _EPS:
            # Stationary — no velocity to project
            event = BackwardGeodesicEvent(
                timestamp_ms=time.time() * 1000,
                problem_id=problem_id,
                kappa_eff=kappa_eff,
                basin_current=current,
                distance_to_solution=d_to_solution,
                backward_component=0.0,
                velocity_norm=0.0,
                mushroom_active=mushroom_active,
            )
            self._event_log.append(event)
            return event

        # Direction toward solution in tangent space at prev
        solution_direction = log_map(prev, solution)
        sol_norm = float(np.sqrt(np.sum(solution_direction * solution_direction)))

        if sol_norm < _EPS:
            # Already at solution
            backward_component = vel_norm
        else:
            # Project velocity onto solution direction
            # Positive = moving toward solution, negative = moving away
            vel_unit = velocity / vel_norm
            sol_unit = solution_direction / sol_norm
            backward_component = float(np.sum(vel_unit * sol_unit)) * vel_norm

        event = BackwardGeodesicEvent(
            timestamp_ms=time.time() * 1000,
            problem_id=problem_id,
            kappa_eff=kappa_eff,
            basin_current=current,
            distance_to_solution=d_to_solution,
            backward_component=backward_component,
            velocity_norm=vel_norm,
            mushroom_active=mushroom_active,
        )
        self._event_log.append(event)
        return event

    def get_events(
        self,
        problem_id: str | None = None,
        mushroom_only: bool = False,
    ) -> list[BackwardGeodesicEvent]:
        """Filter events by problem and/or mushroom state."""
        events = list(self._event_log)
        if problem_id is not None:
            events = [e for e in events if e.problem_id == problem_id]
        if mushroom_only:
            events = [e for e in events if e.mushroom_active]
        return events

    def compute_correlation(
        self,
        problem_id: str | None = None,
        mushroom_only: bool = False,
    ) -> tuple[float, float, int]:
        """Compute correlation between backward-geodesic component and
        inverse distance to solution.

        Returns (rho, p_value, n_events).
        Positive rho = trajectory preferentially moves toward solution.
        """
        events = self.get_events(problem_id=problem_id, mushroom_only=mushroom_only)

        # Filter to events with non-zero velocity
        events = [e for e in events if e.velocity_norm > _EPS]

        if len(events) < 5 or _stats is None:
            return 0.0, 1.0, len(events)

        backward = np.array([e.backward_component for e in events])
        inv_dist = np.array([1.0 / max(e.distance_to_solution, _EPS) for e in events])

        # Guard against constant inputs (pearsonr raises ValueError)
        if np.std(backward) < _EPS or np.std(inv_dist) < _EPS:
            return 0.0, 1.0, len(events)

        rho, p_value = _stats.pearsonr(backward, inv_dist)
        return float(rho), float(p_value), len(events)

    def compute_mean_backward_component(
        self,
        problem_id: str | None = None,
        mushroom_only: bool = False,
    ) -> tuple[float, float, int]:
        """Compute mean backward-geodesic component and its significance.

        Returns (mean, p_value_vs_zero, n_events).
        Mean > 0 with p < 0.05 = significant backward signal.
        """
        events = self.get_events(problem_id=problem_id, mushroom_only=mushroom_only)
        events = [e for e in events if e.velocity_norm > _EPS]

        if len(events) < 5 or _stats is None:
            return 0.0, 1.0, len(events)

        backward = np.array([e.backward_component for e in events])

        # Guard against constant/near-constant inputs (ttest returns NaN)
        if np.std(backward) < _EPS:
            return float(np.mean(backward)), 1.0, len(events)

        t_stat, p_value = _stats.ttest_1samp(backward, 0.0)

        # Handle NaN from degenerate inputs
        if np.isnan(t_stat) or np.isnan(p_value):
            return float(np.mean(backward)), 1.0, len(events)

        # One-sided: we only care if mean > 0
        p_one_sided = p_value / 2.0 if t_stat > 0 else 1.0 - p_value / 2.0

        return float(np.mean(backward)), float(p_one_sided), len(events)

    def summary(self, problem_id: str | None = None) -> dict[str, Any]:
        """Aggregate summary for EXP-011 analysis."""

        rho_m, p_m, n_m = self.compute_correlation(problem_id=problem_id, mushroom_only=True)
        rho_n, p_n, n_n = self.compute_correlation(problem_id=problem_id, mushroom_only=False)
        mean_m, mean_p_m, _ = self.compute_mean_backward_component(
            problem_id=problem_id, mushroom_only=True
        )
        mean_n, mean_p_n, _ = self.compute_mean_backward_component(
            problem_id=problem_id, mushroom_only=False
        )

        return {
            "n_mushroom_events": n_m,
            "n_normal_events": n_n,
            "correlation_mushroom": {"rho": rho_m, "p_value": p_m, "n": n_m},
            "correlation_all": {"rho": rho_n, "p_value": p_n, "n": n_n},
            "mean_backward_mushroom": {"mean": mean_m, "p_value": mean_p_m},
            "mean_backward_normal": {"mean": mean_n, "p_value": mean_p_n},
            "signal_detected": mean_m > 0 and mean_p_m < 0.05,
            "signal_stronger_in_mushroom": mean_m > mean_n and mean_p_m < mean_p_n,
        }

    def reset(self, problem_id: str | None = None) -> None:
        """Clear events and velocity state."""
        if problem_id is None:
            self._event_log.clear()
            self._prev_basins.clear()
        else:
            self._event_log = deque(
                (e for e in self._event_log if e.problem_id != problem_id),
                maxlen=self._event_log.maxlen,
            )
            self._prev_basins.pop(problem_id, None)

    def export_events(self, problem_id: str | None = None) -> list[dict[str, Any]]:
        """Export events for offline analysis."""
        events = self.get_events(problem_id=problem_id)
        return [e.to_dict() for e in events]


def _shannon_entropy(p: Basin) -> float:
    """Normalized Shannon entropy of a simplex point."""
    safe = np.clip(p, 1e-12, None)
    h = -float(np.sum(safe * np.log(safe)))
    return h / np.log(len(p))  # type: ignore[no-any-return]
