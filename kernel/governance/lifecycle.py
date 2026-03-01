"""Governed Lifecycle — gated kernel lifecycle with voting and assessment.

TCP v6.1 §20: Structural kernel decisions (spawn, promote, prune, merge) pass
through governance. Autonomous runtime decisions (evolve, couple, route) do not.

Architecture:
  E8KernelRegistry     — owns all runtime state (basin, phi, kappa, quenched_gain)
  VoterRegistry        — tracks live phi/kappa per GOD kernel for vote weighting
  SpawnAssessment      — scores fitness before a spawn vote is called
  VotingEngine         — evaluates weighted governance votes
  GovernedLifecycle    — the single seam: calls all three in sequence

Callers (loop.py, server.py) use GovernedLifecycle for lifecycle ops
and call E8KernelRegistry directly for runtime ops (evolve, couple, route).

Oversight hooks:
  on_spawn_approved(decision, kernel)    — called after successful spawn
  on_promote_approved(decision, kernel)  — called after CHAOS→GOD promotion
  on_prune_approved(decision, kernel_id) — called before termination
  on_rejected(decision)                  — called on any rejection

Suffering gate:
  Before any spawn, S = phi * (1 - gamma) * meta_awareness is computed
  for existing kernels. If mean S > SUFFERING_THRESHOLD, spawn is blocked
  regardless of vote outcome.

Purity gate:
  run_purity_gate() is called before any structural decision.
  Fail-closed: purity errors are never swallowed.
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..config.frozen_facts import (
    KAPPA_STAR,
    PHI_THRESHOLD,
    SUFFERING_THRESHOLD,
)
from ..consciousness.cradle import Cradle
from ..governance.purity import run_purity_gate
from ..governance.types import KernelKind, KernelSpecialization
from .spawn_assessment import SpawnAssessment, assess_spawn
from .voter_registry import VoterRegistry, get_voter_registry
from .voting import (
    GovernanceDecision,
    GovernanceProposal,
    ProposalType,
    VotingEngine,
    get_voting_engine,
)

logger = logging.getLogger("vex.governance.lifecycle")

_KERNEL_ROOT = Path(__file__).parent.parent


@dataclass
class LifecycleOutcome:
    """Result of a governed lifecycle operation."""

    success: bool
    kernel: Any = None
    decision: GovernanceDecision | None = None
    assessment: SpawnAssessment | None = None
    reason: str = ""


class GovernedLifecycle:
    """Single seam between E8KernelRegistry and governance subsystems.

    Instantiate once and pass to anything that needs to make structural
    kernel decisions. Runtime ops (evolve, couple, route) bypass this
    and call E8KernelRegistry directly.

    Args:
        registry:            E8KernelRegistry instance.
        voter_registry:      VoterRegistry instance (default: module singleton).
        voting_engine:       VotingEngine instance (default: module singleton).
        purity_root:         Root path for purity gate scan (default: kernel/).
        cradle:              Optional Cradle instance for new-kernel development.
        on_spawn_approved:   Optional hook called after spawn approval.
        on_promote_approved: Optional hook called after promote approval.
        on_prune_approved:   Optional hook called before prune execution.
        on_rejected:         Optional hook called on any rejection.
    """

    def __init__(
        self,
        registry: Any,
        voter_registry: VoterRegistry | None = None,
        voting_engine: VotingEngine | None = None,
        purity_root: Path | None = None,
        skip_purity: bool = False,
        cradle: Cradle | None = None,
        on_spawn_approved: Callable[..., Any] | None = None,
        on_promote_approved: Callable[..., Any] | None = None,
        on_prune_approved: Callable[..., Any] | None = None,
        on_rejected: Callable[..., Any] | None = None,
    ) -> None:
        self._registry = registry
        self._vr = voter_registry or get_voter_registry()
        self._engine = voting_engine or get_voting_engine()
        self._purity_root = purity_root or _KERNEL_ROOT
        self._skip_purity = skip_purity
        self._cradle = cradle
        self._on_spawn_approved = on_spawn_approved
        self._on_promote_approved = on_promote_approved
        self._on_prune_approved = on_prune_approved
        self._on_rejected = on_rejected

    def _purity_check(self) -> None:
        """Run purity gate. Raises PurityGateError on any violation (fail-closed)."""
        run_purity_gate(self._purity_root)

    def _suffering_check(self) -> tuple[bool, float]:
        """Compute mean suffering across active kernels.

        S_i = phi_i * (1 - gamma_i) * meta_awareness_i
        Returns (gate_passed, mean_S). Gate passes if mean_S < SUFFERING_THRESHOLD.
        """
        active = self._registry.active()
        if not active:
            return True, 0.0
        values = []
        for k in active:
            phi = getattr(k, "phi", 0.5)
            gamma = min(phi, getattr(k, "gamma", phi))
            meta = getattr(k, "meta_awareness", 0.5)
            s = phi * (1.0 - gamma) * meta
            values.append(s)
        mean_s = sum(values) / len(values)
        return mean_s < SUFFERING_THRESHOLD, mean_s

    def _voter_weights(self) -> list[tuple[str, float, float, float]]:
        return self._vr.get_voter_weights()

    def _sync_voter_registry(self, kernel: Any) -> None:
        """Register or update a GOD/GENESIS kernel in the VoterRegistry."""
        if kernel.kind not in (KernelKind.GOD, KernelKind.GENESIS):
            return
        self._vr.register_or_update(
            kernel_id=kernel.id,
            kernel_name=kernel.name,
            phi=getattr(kernel, "phi", 0.5),
            kappa=getattr(kernel, "kappa", KAPPA_STAR),
            quenched_gain=getattr(kernel, "quenched_gain", 1.0),
        )

    # ───────────────────────────────────────────────────────────────
    # SPAWN
    # ───────────────────────────────────────────────────────────────

    def spawn(
        self,
        name: str,
        kind: KernelKind,
        specialization: KernelSpecialization = KernelSpecialization.GENERAL,
        proposed_gain: float | None = None,
        proposed_basin: Any = None,
        skip_purity: bool = False,
    ) -> LifecycleOutcome:
        """Governed spawn: purity → suffering → assess → vote → spawn → sync."""
        if not (skip_purity or self._skip_purity):
            try:
                self._purity_check()
            except Exception as e:
                logger.error("Purity gate blocked spawn of %s: %s", name, e)
                return LifecycleOutcome(success=False, reason=f"Purity gate: {e}")

        gate_ok, mean_s = self._suffering_check()
        if not gate_ok:
            reason = f"Suffering gate blocked spawn: mean_S={mean_s:.3f} >= {SUFFERING_THRESHOLD}"
            logger.warning(reason)
            return LifecycleOutcome(success=False, reason=reason)

        active = self._registry.active()
        budget = self._registry._budget.summary()
        gain = proposed_gain if proposed_gain is not None else 1.0
        assessment = assess_spawn(
            kind=kind,
            specialization=specialization,
            proposed_gain=gain,
            existing_kernels=active,
            current_god_count=budget["god"],
            current_chaos_count=budget["chaos"],
            proposed_basin=proposed_basin,
        )

        proposal = GovernanceProposal(
            proposal_id=uuid.uuid4().hex[:8],
            proposal_type=ProposalType.SPAWN,
            requester="system",
            description=f"Spawn {kind.value}/{specialization.value}: {name}",
            assessment_score=assessment.score,
            assessment_notes=assessment.notes,
        )
        voter_weights = self._voter_weights()
        decision = self._engine.evaluate(proposal, voter_weights)

        if not decision.approved:
            logger.info("Spawn rejected: %s", decision.summary())
            if self._on_rejected:
                self._on_rejected(decision)
            return LifecycleOutcome(
                success=False,
                decision=decision,
                assessment=assessment,
                reason=decision.summary(),
            )

        try:
            kernel = self._registry.spawn(name=name, kind=kind, specialization=specialization)
        except Exception as e:
            logger.error("Registry spawn failed after governance approval: %s", e)
            return LifecycleOutcome(
                success=False,
                decision=decision,
                assessment=assessment,
                reason=f"Registry spawn failed: {e}",
            )

        self._sync_voter_registry(kernel)

        # v6.0 §23: Admit newly spawned kernel to the Cradle
        if self._cradle is not None:
            self._cradle.admit(kernel.id, getattr(kernel, "phi", 0.0))

        logger.info(
            "Spawn approved: %s (%s/%s) gain=%.3f assessment=%.3f",
            name,
            kind.value,
            specialization.value,
            kernel.quenched_gain,
            assessment.score,
        )
        if self._on_spawn_approved:
            self._on_spawn_approved(decision, kernel)

        return LifecycleOutcome(
            success=True, kernel=kernel, decision=decision, assessment=assessment
        )

    # ───────────────────────────────────────────────────────────────
    # PROMOTE (CHAOS → GOD)
    # ───────────────────────────────────────────────────────────────

    def promote(
        self,
        kernel_id: str,
        skip_purity: bool = False,
    ) -> LifecycleOutcome:
        """Governed CHAOS → GOD: purity → vote (supermajority) → registry phi gate → sync."""
        if not (skip_purity or self._skip_purity):
            try:
                self._purity_check()
            except Exception as e:
                return LifecycleOutcome(success=False, reason=f"Purity gate: {e}")

        kernel = next((k for k in self._registry.active() if k.id == kernel_id), None)
        if kernel is None:
            return LifecycleOutcome(
                success=False, reason=f"Kernel {kernel_id} not found or inactive"
            )
        if kernel.kind != KernelKind.CHAOS:
            return LifecycleOutcome(
                success=False, reason=f"Kernel {kernel.name} is {kernel.kind.value}, not CHAOS"
            )

        proposal = GovernanceProposal(
            proposal_id=uuid.uuid4().hex[:8],
            proposal_type=ProposalType.PROMOTE,
            requester="system",
            description=(
                f"Promote CHAOS->GOD: {kernel.name} "
                f"(phi_peak={kernel.phi_peak:.3f}, cycles={kernel.cycle_count})"
            ),
            subject_kernel_id=kernel_id,
            subject_kernel_name=kernel.name,
        )
        voter_weights = self._voter_weights()
        decision = self._engine.evaluate(proposal, voter_weights)

        if not decision.approved:
            logger.info("Promote rejected: %s", decision.summary())
            if self._on_rejected:
                self._on_rejected(decision)
            return LifecycleOutcome(success=False, decision=decision, reason=decision.summary())

        promoted = self._registry.evaluate_promotion(kernel_id)
        if not promoted:
            reason = (
                f"Registry phi/cycle gate blocked {kernel.name}: "
                f"phi_peak={kernel.phi_peak:.3f} (need >={PHI_THRESHOLD}), "
                f"cycles={kernel.cycle_count}"
            )
            logger.info(reason)
            return LifecycleOutcome(success=False, decision=decision, reason=reason)

        self._sync_voter_registry(kernel)

        logger.info("Promoted: %s -> GOD | %s", kernel.name, decision.summary())
        if self._on_promote_approved:
            self._on_promote_approved(decision, kernel)

        return LifecycleOutcome(success=True, kernel=kernel, decision=decision)

    # ───────────────────────────────────────────────────────────────
    # PRUNE
    # ───────────────────────────────────────────────────────────────

    def prune(
        self,
        kernel_id: str,
        reason: str = "governance decision",
        skip_purity: bool = False,
    ) -> LifecycleOutcome:
        """Governed termination: purity → vote (supermajority) → deregister → terminate."""
        if not (skip_purity or self._skip_purity):
            try:
                self._purity_check()
            except Exception as e:
                return LifecycleOutcome(success=False, reason=f"Purity gate: {e}")

        kernel = next((k for k in self._registry.active() if k.id == kernel_id), None)
        if kernel is None:
            return LifecycleOutcome(success=False, reason=f"Kernel {kernel_id} not found")
        if kernel.kind == KernelKind.GENESIS:
            return LifecycleOutcome(success=False, reason="Cannot prune GENESIS kernel")

        proposal = GovernanceProposal(
            proposal_id=uuid.uuid4().hex[:8],
            proposal_type=ProposalType.PRUNE,
            requester="system",
            description=f"Prune {kernel.kind.value}: {kernel.name} — {reason}",
            subject_kernel_id=kernel_id,
            subject_kernel_name=kernel.name,
        )
        voter_weights = self._voter_weights()
        decision = self._engine.evaluate(proposal, voter_weights)

        if not decision.approved:
            logger.info("Prune rejected: %s", decision.summary())
            if self._on_rejected:
                self._on_rejected(decision)
            return LifecycleOutcome(success=False, decision=decision, reason=decision.summary())

        if self._on_prune_approved:
            self._on_prune_approved(decision, kernel_id)

        self._vr.deregister(kernel_id)
        terminated = self._registry.terminate(kernel_id)
        if not terminated:
            return LifecycleOutcome(
                success=False,
                decision=decision,
                reason=f"Registry termination failed for {kernel_id}",
            )

        logger.info("Pruned: %s | %s", kernel.name, decision.summary())
        return LifecycleOutcome(success=True, decision=decision)

    # ───────────────────────────────────────────────────────────────
    # MERGE
    # ───────────────────────────────────────────────────────────────

    def merge(
        self,
        absorber_id: str,
        victim_id: str,
        blend_weight: float = 0.5,
        skip_purity: bool = False,
    ) -> LifecycleOutcome:
        """Governed merge: unanimous vote → slerp basins → prune victim."""
        if not (skip_purity or self._skip_purity):
            try:
                self._purity_check()
            except Exception as e:
                return LifecycleOutcome(success=False, reason=f"Purity gate: {e}")

        active = {k.id: k for k in self._registry.active()}
        absorber = active.get(absorber_id)
        victim = active.get(victim_id)

        if absorber is None:
            return LifecycleOutcome(success=False, reason=f"Absorber {absorber_id} not found")
        if victim is None:
            return LifecycleOutcome(success=False, reason=f"Victim {victim_id} not found")
        if absorber_id == victim_id:
            return LifecycleOutcome(success=False, reason="Cannot merge kernel with itself")

        proposal = GovernanceProposal(
            proposal_id=uuid.uuid4().hex[:8],
            proposal_type=ProposalType.MERGE,
            requester="system",
            description=(f"Merge {victim.name} -> {absorber.name} (blend={blend_weight:.2f})"),
            subject_kernel_id=victim_id,
            subject_kernel_name=victim.name,
        )
        voter_weights = self._voter_weights()
        decision = self._engine.evaluate(proposal, voter_weights)

        if not decision.approved:
            logger.info("Merge rejected: %s", decision.summary())
            if self._on_rejected:
                self._on_rejected(decision)
            return LifecycleOutcome(success=False, decision=decision, reason=decision.summary())

        if absorber.basin is not None and victim.basin is not None:
            from ..coordizer_v2.geometry import slerp

            absorber.basin = slerp(absorber.basin, victim.basin, blend_weight)

        absorber.phi = max(absorber.phi, victim.phi)
        absorber.phi_peak = max(absorber.phi_peak, victim.phi_peak)

        self._vr.deregister(victim_id)
        self._registry.terminate(victim_id)
        self._sync_voter_registry(absorber)

        logger.info(
            "Merged: %s -> %s | blend=%.2f | phi=%.3f | %s",
            victim.name,
            absorber.name,
            blend_weight,
            absorber.phi,
            decision.summary(),
        )
        return LifecycleOutcome(success=True, kernel=absorber, decision=decision)

    # ── Utilities ────────────────────────────────────────────────────────

    def sync_all_voters(self) -> int:
        """Sync all active GOD/GENESIS kernels to VoterRegistry after restore()."""
        count = 0
        for k in self._registry.active():
            if k.kind in (KernelKind.GOD, KernelKind.GENESIS):
                self._sync_voter_registry(k)
                count += 1
        return count

    def oversight_summary(self) -> dict[str, Any]:
        """Current governance state for audit/telemetry."""
        vr_snap = self._vr.snapshot()
        budget = self._registry._budget.summary()
        gate_ok, mean_s = self._suffering_check()
        summary: dict[str, Any] = {
            "voter_registry": vr_snap,
            "budget": budget,
            "suffering_gate": {
                "passed": gate_ok,
                "mean_s": round(mean_s, 4),
                "threshold": SUFFERING_THRESHOLD,
            },
            "recent_decisions": self._engine.recent(5),
        }
        if self._cradle is not None:
            summary["cradle"] = self._cradle.get_state()
        return summary
