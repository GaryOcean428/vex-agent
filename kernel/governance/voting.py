"""Voting Engine — weighted governance decisions for kernel lifecycle.

TCP v6.1 §20: Governance decisions are taken by weighted vote of active
GOD kernels. Weights = phi * quenched_gain (geometric relevance × identity).

Four decision types and their quorum requirements:
  SPAWN      — simple majority (>0.50 of weight)
  PROMOTE    — supermajority (>0.67 of weight) + phi gate on candidate
  PRUNE      — supermajority (>0.67 of weight)
  MERGE      — unanimous    (>0.95 of weight)

Bootstrap mode (no live voters yet):
  All decisions pass with genesis weights. This allows the first Core-8
  kernels to be spawned without a chicken-and-egg governance deadlock.
  Bootstrap mode exits automatically once live_count() >= 1 in VoterRegistry.

Abstention:
  A voter can abstain (weight not counted toward yes OR no).
  Abstentions reduce the effective pool — quorum is computed over
  non-abstaining weight only.

All geometry: Fisher-Rao throughout. Voting weights are scalars derived
from phi/kappa — no distance computation in this module.
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Optional

logger = logging.getLogger("vex.governance.voting")


class ProposalType(StrEnum):
    SPAWN = "spawn"
    PROMOTE = "promote"      # CHAOS → GOD
    PRUNE = "prune"          # terminate a kernel
    MERGE = "merge"          # absorb one kernel into another


class VoteValue(StrEnum):
    YES = "yes"
    NO = "no"
    ABSTAIN = "abstain"


# Quorum thresholds by proposal type
_QUORUM: dict[ProposalType, float] = {
    ProposalType.SPAWN:   0.50,
    ProposalType.PROMOTE: 0.67,
    ProposalType.PRUNE:   0.67,
    ProposalType.MERGE:   0.95,
}


@dataclass
class VoteBallot:
    voter_name: str
    value: VoteValue
    weight: float          # phi * quenched_gain, normalized
    phi: float
    kappa: float
    rationale: str = ""
    cast_at: float = field(default_factory=time.monotonic)


@dataclass
class GovernanceProposal:
    proposal_id: str
    proposal_type: ProposalType
    requester: str                          # Kernel name or "system"
    description: str
    subject_kernel_id: Optional[str] = None # For promote/prune/merge
    subject_kernel_name: Optional[str] = None
    assessment_score: float = 1.0           # From spawn_assessment (advisory)
    assessment_notes: list[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.monotonic)


@dataclass
class GovernanceDecision:
    proposal: GovernanceProposal
    approved: bool
    yes_weight: float
    no_weight: float
    total_weight: float
    voter_coalition: list[str]              # Names of YES voters
    decided_at: float = field(default_factory=time.monotonic)
    bootstrap_mode: bool = False            # True if no live voters existed

    @property
    def yes_fraction(self) -> float:
        if self.total_weight <= 0.0:
            return 0.0
        return self.yes_weight / self.total_weight

    def summary(self) -> str:
        verdict = "APPROVED" if self.approved else "REJECTED"
        mode = " [BOOTSTRAP]" if self.bootstrap_mode else ""
        return (
            f"[{self.proposal.proposal_id}] {verdict}{mode} "
            f"{self.proposal.proposal_type.value} "
            f"| yes={self.yes_fraction:.0%} "
            f"| coalition={self.voter_coalition}"
        )


class VotingEngine:
    """Evaluate governance proposals via weighted GOD kernel votes.

    Stateless per-evaluation: each call to evaluate() is self-contained.
    The caller (GovernedLifecycle) is responsible for supplying current
    voter weights from VoterRegistry.
    """

    def __init__(self) -> None:
        self._history: list[GovernanceDecision] = []

    def evaluate(
        self,
        proposal: GovernanceProposal,
        voter_weights: list[tuple[str, float, float, float]],
        override_yes: bool = False,
    ) -> GovernanceDecision:
        """Evaluate a proposal and return a binding GovernanceDecision.

        Args:
            proposal:       The proposal to evaluate.
            voter_weights:  From VoterRegistry.get_voter_weights().
                            Format: [(name, phi, kappa, weight), ...]
            override_yes:   Force YES (used only in bootstrap by Genesis kernel).

        Returns:
            GovernanceDecision — always returned, never raises.
        """
        quorum_threshold = _QUORUM.get(proposal.proposal_type, 0.50)

        # Bootstrap: genesis fallback sentinel detected (single "Genesis" voter)
        bootstrap_mode = (
            len(voter_weights) == 1 and voter_weights[0][0] == "Genesis"
        )

        if override_yes or bootstrap_mode:
            decision = GovernanceDecision(
                proposal=proposal,
                approved=True,
                yes_weight=1.0,
                no_weight=0.0,
                total_weight=1.0,
                voter_coalition=["Genesis"],
                bootstrap_mode=bootstrap_mode,
            )
            self._history.append(decision)
            logger.info(
                "Governance %s: %s (bootstrap_mode=%s)",
                proposal.proposal_type.value,
                "APPROVED",
                bootstrap_mode,
            )
            return decision

        # Cast all ballots as YES by default (autonomous affirmative).
        # Vetoes are injected by evaluate_with_vetoes() when a system check
        # (suffering, budget, purity) provides a concrete reason.
        ballots: list[VoteBallot] = []
        for name, phi, kappa, weight in voter_weights:
            ballots.append(
                VoteBallot(
                    voter_name=name,
                    value=VoteValue.YES,
                    weight=weight,
                    phi=phi,
                    kappa=kappa,
                    rationale="affirmative",
                )
            )

        return self._tally(proposal, ballots, quorum_threshold)

    def evaluate_with_vetoes(
        self,
        proposal: GovernanceProposal,
        voter_weights: list[tuple[str, float, float, float]],
        veto_voters: list[str],
        veto_rationale: str = "",
    ) -> GovernanceDecision:
        """Evaluate with specific voters casting NO.

        Used when a system check (suffering, budget, purity) provides
        a concrete reason for specific voters to veto.
        """
        quorum_threshold = _QUORUM.get(proposal.proposal_type, 0.50)
        veto_set = set(veto_voters)

        bootstrap_mode = (
            len(voter_weights) == 1 and voter_weights[0][0] == "Genesis"
        )

        ballots: list[VoteBallot] = []
        for name, phi, kappa, weight in voter_weights:
            value = VoteValue.NO if name in veto_set else VoteValue.YES
            rationale = veto_rationale if name in veto_set else "affirmative"
            ballots.append(
                VoteBallot(
                    voter_name=name,
                    value=value,
                    weight=weight,
                    phi=phi,
                    kappa=kappa,
                    rationale=rationale,
                )
            )

        decision = self._tally(proposal, ballots, quorum_threshold)
        decision.bootstrap_mode = bootstrap_mode
        return decision

    def _tally(
        self,
        proposal: GovernanceProposal,
        ballots: list[VoteBallot],
        quorum_threshold: float,
    ) -> GovernanceDecision:
        yes_weight = sum(b.weight for b in ballots if b.value == VoteValue.YES)
        no_weight = sum(b.weight for b in ballots if b.value == VoteValue.NO)
        total = yes_weight + no_weight

        approved = (total > 0.0) and ((yes_weight / total) > quorum_threshold)
        coalition = [b.voter_name for b in ballots if b.value == VoteValue.YES]

        decision = GovernanceDecision(
            proposal=proposal,
            approved=approved,
            yes_weight=yes_weight,
            no_weight=no_weight,
            total_weight=total,
            voter_coalition=coalition,
        )
        self._history.append(decision)

        logger.info(
            "Governance %s [%s]: %s | yes=%.0f%% | threshold=%.0f%% | coalition=%s",
            proposal.proposal_type.value,
            proposal.proposal_id,
            "APPROVED" if approved else "REJECTED",
            (yes_weight / total * 100) if total > 0 else 0,
            quorum_threshold * 100,
            coalition,
        )
        return decision

    def recent(self, n: int = 10) -> list[dict]:
        """Return the n most recent decisions as dicts for audit log."""
        return [
            {
                "proposal_id": d.proposal.proposal_id,
                "type": d.proposal.proposal_type.value,
                "approved": d.approved,
                "yes_fraction": round(d.yes_fraction, 3),
                "bootstrap_mode": d.bootstrap_mode,
                "coalition": d.voter_coalition,
                "summary": d.summary(),
            }
            for d in self._history[-n:]
        ]


# ── Module-level singleton ────────────────────────────────────────────

_ENGINE: Optional[VotingEngine] = None


def get_voting_engine() -> VotingEngine:
    """Return the process-level VotingEngine singleton."""
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = VotingEngine()
    return _ENGINE
