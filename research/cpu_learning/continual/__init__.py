"""Stage S8 Continual Learning, Mandatory Regression Replay & Cross-Domain Transfer."""
from __future__ import annotations

from research.cpu_learning.continual.cross_domain import (
    CROSS_DOMAIN_SCHEMA_VERSION,
    AbstractionScope,
    CrossDomainAbstractionRegistry,
    CrossDomainTransferEvaluator,
    RegisteredAbstractionEntry,
    TargetDomainTransferReport,
)
from research.cpu_learning.continual.replay_manager import (
    REPLAY_MANAGER_SCHEMA_VERSION,
    CandidateUpdateOutcome,
    ContinualKnowledgeBase,
    ContinualLearnerFrozenError,
    DomainRegressionCase,
    LibraryGrowthSnapshot,
    NaiveOverwriteNoReplayBaseline,
    QuarantineRecord,
    RegressionReplayManager,
    ReplayVerificationReport,
    SkillLifecycleState,
    SmallCPUNeuralContinualRef,
    VersionedSkillEntry,
    encode_hashed_feature_vector,
)

__all__ = [
    "AbstractionScope",
    "CROSS_DOMAIN_SCHEMA_VERSION",
    "CandidateUpdateOutcome",
    "ContinualKnowledgeBase",
    "ContinualLearnerFrozenError",
    "CrossDomainAbstractionRegistry",
    "CrossDomainTransferEvaluator",
    "DomainRegressionCase",
    "LibraryGrowthSnapshot",
    "NaiveOverwriteNoReplayBaseline",
    "QuarantineRecord",
    "REPLAY_MANAGER_SCHEMA_VERSION",
    "RegisteredAbstractionEntry",
    "RegressionReplayManager",
    "ReplayVerificationReport",
    "SkillLifecycleState",
    "SmallCPUNeuralContinualRef",
    "TargetDomainTransferReport",
    "VersionedSkillEntry",
    "encode_hashed_feature_vector",
]
