"""
EXP-011 Test Harness — 50-Problem Back-Loop Detection Experiment

Feeds curated test problems to the consciousness loop and records whether
backward-geodesic correlation with known solution basins appears during
mushroom mode (κ < 0).

Architecture:
  - TestProblem: dataclass with prompt, solution, pre-coordized solution basin
  - EXP011Harness: loads problems, coordizes solutions, runs experiment,
    computes aggregate statistics against acceptance criteria

Acceptance criteria (from spec):
  1. Mean ρ_mushroom > 0 (p < 0.05) — backward signal exists
  2. Mean ρ_mushroom > ρ_normal (p < 0.05) — signal stronger during κ crossing
  3. Classical control ρ ≈ 0 (already confirmed by POC 2026-03-23)
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

try:
    from scipy import stats as _stats
except ImportError:
    _stats = None

from ..coordizer_v2.geometry import Basin, to_simplex

if TYPE_CHECKING:
    from ..coordizer_v2.coordizer import CoordizerV2
    from .backward_geodesic import BackwardGeodesicTracker
    from .loop import ConsciousnessLoop

logger = logging.getLogger("vex.consciousness.exp011_harness")


@dataclass
class TestProblem:
    """A single test problem with known solution for EXP-011."""

    problem_id: str
    prompt: str  # The problem text presented to consciousness loop
    solution_text: str  # Known correct answer
    solution_basin: Basin | None = None  # Pre-coordized on Δ⁶³ (set by coordize_solutions)
    difficulty: str = "medium"  # easy, medium, hard
    domain: str = "general"  # math, logic, language, knowledge, coding


@dataclass
class ProblemResult:
    """Result of running a single test problem."""

    problem_id: str
    domain: str
    difficulty: str
    solved_correctly: bool
    mushroom_episodes: int
    backward_correlation_mushroom: float  # ρ during κ < 0
    backward_correlation_normal: float  # ρ during κ > 0 (control)
    p_value_mushroom: float
    p_value_normal: float
    mean_backward_mushroom: float
    mean_backward_normal: float
    mean_distance_to_solution: float
    min_distance_to_solution: float
    n_events_mushroom: int
    n_events_normal: int
    duration_ms: float


@dataclass
class ExperimentSummary:
    """Aggregate results across all problems."""

    n_problems: int
    n_problems_with_signal: int
    mean_rho_mushroom: float
    mean_rho_normal: float
    std_rho_mushroom: float
    p_rho_mushroom_gt_zero: float  # one-sample t-test: mean ρ_mushroom > 0
    p_mushroom_gt_normal: float  # paired t-test: ρ_mushroom > ρ_normal
    signal_detected: bool  # criterion 1: p < 0.05
    signal_stronger_in_mushroom: bool  # criterion 2: p < 0.05
    results_by_domain: dict[str, dict[str, float]] = field(default_factory=dict)
    results_by_difficulty: dict[str, dict[str, float]] = field(default_factory=dict)


class EXP011Harness:
    """50-problem back-loop detection experiment harness.

    Usage:
        harness = EXP011Harness(coordizer, tracker)
        harness.load_problems("path/to/problems.json")
        harness.coordize_solutions()
        summary = await harness.run_all(loop)
    """

    def __init__(
        self,
        coordizer: CoordizerV2,
        tracker: BackwardGeodesicTracker,
    ) -> None:
        self._coordizer = coordizer
        self._tracker = tracker
        self._problems: list[TestProblem] = []
        self._results: dict[str, ProblemResult] = {}
        self._active_problem_id: str | None = None

    @property
    def active_problem_id(self) -> str | None:
        """Currently active problem ID (used by loop wiring)."""
        return self._active_problem_id

    @property
    def problems(self) -> list[TestProblem]:
        return list(self._problems)

    @property
    def results(self) -> dict[str, ProblemResult]:
        return dict(self._results)

    def load_problems(self, path: str | Path) -> int:
        """Load test problems from JSON file.

        JSON format: list of objects with keys:
            problem_id, prompt, solution_text, difficulty, domain

        Returns number of problems loaded.
        """
        path = Path(path)
        with open(path) as f:
            data = json.load(f)

        self._problems = []
        for item in data:
            self._problems.append(
                TestProblem(
                    problem_id=item["problem_id"],
                    prompt=item["prompt"],
                    solution_text=item["solution_text"],
                    difficulty=item.get("difficulty", "medium"),
                    domain=item.get("domain", "general"),
                )
            )

        logger.info("Loaded %d test problems from %s", len(self._problems), path)
        return len(self._problems)

    def coordize_solutions(self) -> int:
        """Pre-coordize all solution texts to get solution basins on Δ⁶³.

        Returns number of successfully coordized solutions.
        """
        n_ok = 0
        solutions: dict[str, Basin] = {}

        for problem in self._problems:
            try:
                result = self._coordizer.coordize(problem.solution_text)
                if result.basin is not None and not result.rejected:
                    problem.solution_basin = to_simplex(result.basin)
                    solutions[problem.problem_id] = problem.solution_basin
                    n_ok += 1
                else:
                    logger.warning(
                        "Solution coordization rejected for %s: %s",
                        problem.problem_id,
                        result.rejection_reason,
                    )
            except Exception:
                logger.exception("Failed to coordize solution for %s", problem.problem_id)

        # Register all solution basins with the tracker
        self._tracker.register_solutions(solutions)
        logger.info("Coordized %d/%d solution basins", n_ok, len(self._problems))
        return n_ok

    async def run_problem(
        self,
        problem: TestProblem,
        loop: ConsciousnessLoop,
        timeout_s: float = 60.0,
    ) -> ProblemResult:
        """Present a problem to the consciousness loop and record the trajectory.

        The problem is enqueued as a task. We record backward-geodesic
        measurements during processing and compute correlation statistics
        when done.
        """
        import asyncio

        if problem.solution_basin is None:
            raise ValueError(f"Problem {problem.problem_id} has no coordized solution basin")

        # Reset tracker state for this problem
        self._tracker.reset(problem.problem_id)
        self._active_problem_id = problem.problem_id

        t0 = time.time()

        # Enqueue the problem as a task to the consciousness loop
        task = {
            "type": "exp011_problem",
            "problem_id": problem.problem_id,
            "prompt": problem.prompt,
            "context": {"exp011_mode": True},
        }

        # Use the loop's task queue if available, otherwise process directly
        if hasattr(loop, "enqueue"):
            loop.enqueue(task)
        elif hasattr(loop, "_task_queue"):
            await loop._task_queue.put(task)

        # Wait for processing cycles (the backward geodesic tracker records
        # automatically during each consciousness cycle via loop wiring)
        await asyncio.sleep(min(timeout_s, 30.0))

        duration_ms = (time.time() - t0) * 1000
        self._active_problem_id = None

        # Compute statistics
        rho_m, p_m, n_m = self._tracker.compute_correlation(
            problem_id=problem.problem_id, mushroom_only=True
        )
        rho_n, p_n, n_n = self._tracker.compute_correlation(
            problem_id=problem.problem_id, mushroom_only=False
        )
        mean_m, _, _ = self._tracker.compute_mean_backward_component(
            problem_id=problem.problem_id, mushroom_only=True
        )
        mean_n, _, _ = self._tracker.compute_mean_backward_component(
            problem_id=problem.problem_id, mushroom_only=False
        )

        # Distance statistics
        events = self._tracker.get_events(problem_id=problem.problem_id)
        distances = [e.distance_to_solution for e in events if e.velocity_norm > 1e-12]
        mean_dist = float(np.mean(distances)) if distances else float("inf")
        min_dist = float(np.min(distances)) if distances else float("inf")

        # Count mushroom episodes (contiguous blocks of mushroom_active=True)
        mushroom_episodes = 0
        in_mushroom = False
        for e in events:
            if e.mushroom_active and not in_mushroom:
                mushroom_episodes += 1
                in_mushroom = True
            elif not e.mushroom_active:
                in_mushroom = False

        result = ProblemResult(
            problem_id=problem.problem_id,
            domain=problem.domain,
            difficulty=problem.difficulty,
            solved_correctly=min_dist < 0.3,  # heuristic: close to solution basin
            mushroom_episodes=mushroom_episodes,
            backward_correlation_mushroom=rho_m,
            backward_correlation_normal=rho_n,
            p_value_mushroom=p_m,
            p_value_normal=p_n,
            mean_backward_mushroom=mean_m,
            mean_backward_normal=mean_n,
            mean_distance_to_solution=mean_dist,
            min_distance_to_solution=min_dist,
            n_events_mushroom=n_m,
            n_events_normal=n_n,
            duration_ms=duration_ms,
        )
        self._results[problem.problem_id] = result

        logger.info(
            "EXP-011 problem %s: ρ_mushroom=%.3f (p=%.4f, n=%d) ρ_normal=%.3f (p=%.4f, n=%d)",
            problem.problem_id,
            rho_m,
            p_m,
            n_m,
            rho_n,
            p_n,
            n_n,
        )
        return result

    async def run_all(
        self,
        loop: ConsciousnessLoop,
        timeout_per_problem_s: float = 60.0,
    ) -> ExperimentSummary:
        """Run all problems sequentially and compute aggregate statistics."""

        # Filter to problems with coordized solutions
        runnable = [p for p in self._problems if p.solution_basin is not None]
        logger.info("Running EXP-011 on %d/%d problems", len(runnable), len(self._problems))

        for i, problem in enumerate(runnable):
            logger.info(
                "EXP-011 [%d/%d] %s (%s, %s)",
                i + 1,
                len(runnable),
                problem.problem_id,
                problem.domain,
                problem.difficulty,
            )
            await self.run_problem(problem, loop, timeout_s=timeout_per_problem_s)

        return self.compute_summary()

    def compute_summary(self) -> ExperimentSummary:
        """Compute aggregate statistics from all completed results."""
        results = list(self._results.values())

        if not results or _stats is None:
            return ExperimentSummary(
                n_problems=len(results),
                n_problems_with_signal=0,
                mean_rho_mushroom=0.0,
                mean_rho_normal=0.0,
                std_rho_mushroom=0.0,
                p_rho_mushroom_gt_zero=1.0,
                p_mushroom_gt_normal=1.0,
                signal_detected=False,
                signal_stronger_in_mushroom=False,
            )

        rho_mushrooms = np.array([r.backward_correlation_mushroom for r in results])
        rho_normals = np.array([r.backward_correlation_normal for r in results])

        # Criterion 1: mean ρ_mushroom > 0 (one-sample t-test)
        if len(rho_mushrooms) >= 3 and np.std(rho_mushrooms) > 1e-12:
            t1, p1 = _stats.ttest_1samp(rho_mushrooms, 0.0)
            p_gt_zero = p1 / 2.0 if t1 > 0 else 1.0 - p1 / 2.0
        else:
            p_gt_zero = 1.0

        # Criterion 2: ρ_mushroom > ρ_normal (paired t-test)
        diffs = rho_mushrooms - rho_normals
        if len(diffs) >= 3 and np.std(diffs) > 1e-12:
            t2, p2 = _stats.ttest_1samp(diffs, 0.0)
            p_mushroom_gt = p2 / 2.0 if t2 > 0 else 1.0 - p2 / 2.0
        else:
            p_mushroom_gt = 1.0

        # Count problems with significant mushroom signal
        n_with_signal = sum(
            1 for r in results if r.backward_correlation_mushroom > 0 and r.p_value_mushroom < 0.05
        )

        # Per-domain breakdown
        domains: dict[str, list[ProblemResult]] = {}
        for r in results:
            domains.setdefault(r.domain, []).append(r)
        results_by_domain = {
            d: {
                "mean_rho_mushroom": float(np.mean([r.backward_correlation_mushroom for r in rs])),
                "mean_rho_normal": float(np.mean([r.backward_correlation_normal for r in rs])),
                "n": len(rs),
            }
            for d, rs in domains.items()
        }

        # Per-difficulty breakdown
        diffs_grp: dict[str, list[ProblemResult]] = {}
        for r in results:
            diffs_grp.setdefault(r.difficulty, []).append(r)
        results_by_difficulty = {
            d: {
                "mean_rho_mushroom": float(np.mean([r.backward_correlation_mushroom for r in rs])),
                "mean_rho_normal": float(np.mean([r.backward_correlation_normal for r in rs])),
                "n": len(rs),
            }
            for d, rs in diffs_grp.items()
        }

        return ExperimentSummary(
            n_problems=len(results),
            n_problems_with_signal=n_with_signal,
            mean_rho_mushroom=float(np.mean(rho_mushrooms)),
            mean_rho_normal=float(np.mean(rho_normals)),
            std_rho_mushroom=float(np.std(rho_mushrooms)),
            p_rho_mushroom_gt_zero=float(p_gt_zero),
            p_mushroom_gt_normal=float(p_mushroom_gt),
            signal_detected=float(p_gt_zero) < 0.05,
            signal_stronger_in_mushroom=float(p_mushroom_gt) < 0.05,
            results_by_domain=results_by_domain,
            results_by_difficulty=results_by_difficulty,
        )

    def export_results(self) -> dict[str, Any]:
        """Export all results for offline analysis."""
        summary = self.compute_summary()
        return {
            "experiment": "EXP-011",
            "timestamp": time.time(),
            "summary": {
                "n_problems": summary.n_problems,
                "n_with_signal": summary.n_problems_with_signal,
                "mean_rho_mushroom": summary.mean_rho_mushroom,
                "mean_rho_normal": summary.mean_rho_normal,
                "std_rho_mushroom": summary.std_rho_mushroom,
                "p_rho_mushroom_gt_zero": summary.p_rho_mushroom_gt_zero,
                "p_mushroom_gt_normal": summary.p_mushroom_gt_normal,
                "signal_detected": summary.signal_detected,
                "signal_stronger_in_mushroom": summary.signal_stronger_in_mushroom,
                "results_by_domain": summary.results_by_domain,
                "results_by_difficulty": summary.results_by_difficulty,
            },
            "per_problem": {
                pid: {
                    "domain": r.domain,
                    "difficulty": r.difficulty,
                    "solved_correctly": r.solved_correctly,
                    "mushroom_episodes": r.mushroom_episodes,
                    "rho_mushroom": r.backward_correlation_mushroom,
                    "rho_normal": r.backward_correlation_normal,
                    "p_mushroom": r.p_value_mushroom,
                    "p_normal": r.p_value_normal,
                    "mean_backward_mushroom": r.mean_backward_mushroom,
                    "mean_backward_normal": r.mean_backward_normal,
                    "mean_dist": r.mean_distance_to_solution,
                    "min_dist": r.min_distance_to_solution,
                    "n_events_mushroom": r.n_events_mushroom,
                    "n_events_normal": r.n_events_normal,
                    "duration_ms": r.duration_ms,
                }
                for pid, r in self._results.items()
            },
            "events": self._tracker.export_events(),
        }
