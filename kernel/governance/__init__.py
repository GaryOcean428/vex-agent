"""Governance — types, purity, budget, voting, assessment, lifecycle."""

from .budget import BudgetAccountingError, BudgetEnforcer, BudgetExceededError
from .lifecycle import GovernedLifecycle, LifecycleOutcome
from .purity import PurityGateError, run_purity_gate
from .spawn_assessment import SpawnAssessment, assess_spawn
from .types import (
    CORE_8_SPECIALIZATIONS,
    VARIABLE_REGISTRY,
    CoachingStage,
    KernelKind,
    KernelRole,
    KernelSpecialization,
    LifecyclePhase,
    LifecycleState,
    VariableCategory,
    get_variable_category,
    register_variable,
)
from .voter_registry import VoterRegistry, get_voter_registry
from .voting import (
    GovernanceDecision,
    GovernanceProposal,
    ProposalType,
    VotingEngine,
    get_voting_engine,
)

__all__ = [
    # budget
    "BudgetAccountingError",
    "BudgetEnforcer",
    "BudgetExceededError",
    # lifecycle
    "GovernedLifecycle",
    "LifecycleOutcome",
    # purity
    "PurityGateError",
    "run_purity_gate",
    # spawn assessment
    "SpawnAssessment",
    "assess_spawn",
    # types
    "CoachingStage",
    "CORE_8_SPECIALIZATIONS",
    "VARIABLE_REGISTRY",
    "get_variable_category",
    "register_variable",
    "KernelKind",
    "KernelRole",
    "KernelSpecialization",
    "LifecyclePhase",
    "LifecycleState",
    "VariableCategory",
    # voter registry
    "VoterRegistry",
    "get_voter_registry",
    # voting
    "GovernanceDecision",
    "GovernanceProposal",
    "ProposalType",
    "VotingEngine",
    "get_voting_engine",
]
