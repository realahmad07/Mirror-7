"""
Mirror Crystal — Zero-Weight, Zero-Gradient, Surprise-Driven Byte Crystallization Engine.
Author: Ahmad Hassan | Repository: https://github.com/realahmad07/Mirror-7 | License: Apache-2.0
"""
from __future__ import annotations

from .autonomous_streamer import AutonomousDatasetStreamer
from .answer_verification import BlueprintAnswerVerifier, BlueprintVerificationReport
from .blueprint_runtime import BlueprintCompositionEngine, BlueprintCompositionError, BlueprintCompositionResult
from .claim_verification import ClaimEvidence, ClaimEvidenceError, ClaimEvidenceReport, ClaimEvidenceVerifier
from .blueprint_contracts import (
    Blueprint, BlueprintDependencyGraph, BlueprintKind, BlueprintProvenance,
    BlueprintValidationError, EvidenceLevel, VerificationEvidence,
)
from .cross_session_memory import CrossSessionMemoryGraph
from .vertical_blueprints import VerticalBlueprintCurator
from .roadmap_blueprint_engine import RoadmapBlueprintEngine
from .bit_signatures import ContextBitTable
from .dataset_curator import CurationConfig, curate_all_datasets
from .disk_vault import DiskCrystalVault, MmapPostingsIndex
from .domain_sharded_vault import DomainShardedVault
from .engine import CrystalResponse, MirrorCrystalEngine, split_compound_intents
from .grammar_and_sandbox import ByteMarkovGrammar, ZeroTrustSandboxJudge
from .lsh_rule_store import CrystalRule, LSHRuleIndex
from .phase2_cognitive_core import (
    CoreferenceAndSenseTracker,
    DreamUtilityPruner,
    EpistemicConfidenceCalibrator,
    FluentByteRealizer,
    NightlyDreamSynthesizer,
    PerceptualVQLens,
    SandboxMCTSPlanner,
    UniversalConceptHub,
)
from .phase5_bicameral_and_cas import (
    ActiveCuriosityPlanner,
    CASComplexityExplosionError,
    CASResult,
    DocumentAutoCrystallizer,
    FactFidelityGuard,
    PackedTernaryProjection,
    SymbolicCASOrgan,
    TernaryFluidComposer,
)
from .phase5_sensory_shards_wasm import (
    PNGSecurityError,
    PackedSimdBitKernel,
    SensoryCodec,
    ShardedCrystalRouter,
    WasmPocketCompiler,
)
from .phase7_qwen_to_opus import (
    AutonomousTDDEngine,
    CerebrumCrystalBridge,
    HierarchicalProjectState,
    ProcessRewardVerifier,
    ScalableBloomFilter,
    SixGateDataHygienePipeline,
    StepEvaluation,
    TDDExecutionReport,
)
from .pocket_teleport import (
    CloudCrystalNode,
    OnDevicePrivacyScrubber,
    PocketCrystalClient,
    SignedRuleCapsule,
)
from .slot_abstractor import AbstractedPair, SlotAbstractor
from .universal_representation import (
    EpistemicStatus,
    RepresentationConflictError,
    RepresentationGraph,
    RepresentationKind,
    RepresentationNode,
    RepresentationProvenance,
    RepresentationRelation,
    RepresentationUncertainty,
    RepresentationValidationError,
    UncertaintyKind,
    canonical_json,
    stable_representation_id,
)
from .capability_model import (
    CapabilityConflictError, CapabilityRegistry, CapabilitySpec,
    CapabilityValidationError, PortContract,
)
from .capability_matching import (
    CapabilityMatch, CapabilityMatchReport, CapabilityMatcher,
)
from .capability_composition import (
    CapabilityCompositionPlanner, CompositionPlan, CompositionPlanningError,
    InputBinding, PlanStatus, PlanStep,
)
from .execution import (
    CapabilityExecutionEngine, ExecutionError, ExecutionPolicy, ExecutionStatus,
    NativeOperationRegistry, PlanExecutionReport, StepExecutionResult,
)
from .verification import (
    CapabilityVerification, CheckResult, IndependentCheck,
    IndependentVerificationEngine, IndependentVerifierRegistry,
    VerificationContext, VerificationError, VerificationReport, VerificationStatus,
)
from .domain_verification import (
    DomainCapabilityVerification, DomainCheckResult, DomainDecision,
    DomainEvidenceArtifact, DomainVerificationCheck, DomainVerificationEngine,
    DomainVerificationError, DomainVerificationReport, DomainVerificationRequirement,
    DomainVerificationStatus, DomainVerifierRegistry,
)
from .capability_learning import (
    CapabilityLearningEngine, CapabilityLearningError,
    CapabilityLearningObservation, CapabilityLearningProposal,
)
from .knowledge_foundation import (
    KnowledgeConflict, KnowledgeConflictError, KnowledgeKind, KnowledgeRecord,
    KnowledgeRegistry, KnowledgeSearchResult, KnowledgeValidationError,
)
from .task_understanding import (
    ConstraintKind,
    TaskConstraint,
    TaskIntent,
    TaskSpec,
    TaskStatus,
    TaskSubtask,
    TaskUnderstandingEngine,
    TaskUnderstandingError,
)

from .operational_language import (
    UOL, UOLError, UOLExecutionError, RuleStatus, VariableSpec, Lit, Var, Call,
    If, ArrayExpr, RecordExpr, Field, Index, Let, MapExpr, FoldExpr,
    OperationalRule, RuleCatalog, Program, Expression, build_program,
    expr_dict, expr_from_dict, program_dict, program_from_dict, knowledge_rule,
)
from .capability_synthesizer import (
    CapabilitySynthesizer, Counterexample, Example, SynthesisError, SynthesisGoal,
    SynthesisReport, SynthesisStatus,
)
from .knowledge_uol import (
    KnowledgeRuleImporter, KnowledgeRuleImportIssue, KnowledgeRuleImportReport,
)
from .native_verifiers import NativeDomainVerifierSuite, NativeVerificationError
from .verified_program_learning import (
    ProgramLearningError, UOLProgramHandler, UOLProgramLearningProposal,
    VerifiedProgramLearningEngine, VerifiedProgramObservation,
)
from .cross_domain_transfer import (
    CrossDomainTransferEngine, CrossDomainTransferError, CrossDomainTransferProposal,
)
from .failure_repair import (
    FailureDiagnosis, FailureKind, FailureRepairEngine, FailureRepairError,
    RepairProposal, RepairStatus,
)
from .reasoning_engine import ReasoningIntelligenceEngine, ReasoningResult, ReasoningStep
from .autonomous_learning import (
    AutonomousLearningEngine,
    AutonomousLearningReport,
    BenchmarkTaskResult,
    LearnedCandidate,
    ProductionEvaluationReport,
)


__all__ = [
    "KnowledgeConflict",
    "KnowledgeConflictError",
    "KnowledgeKind",
    "KnowledgeRecord",
    "KnowledgeRegistry",
    "KnowledgeSearchResult",
    "KnowledgeValidationError",
    "CapabilityConflictError",
    "CapabilityRegistry",
    "CapabilitySpec",
    "CapabilityValidationError",
    "PortContract",
    "CapabilityMatch",
    "CapabilityMatchReport",
    "CapabilityMatcher",
    "CapabilityCompositionPlanner",
    "CompositionPlan",
    "CompositionPlanningError",
    "InputBinding",
    "PlanStatus",
    "PlanStep",
    "CapabilityExecutionEngine",
    "ExecutionError",
    "ExecutionPolicy",
    "ExecutionStatus",
    "NativeOperationRegistry",
    "PlanExecutionReport",
    "StepExecutionResult",
    "CapabilityVerification",
    "CheckResult",
    "IndependentCheck",
    "IndependentVerificationEngine",
    "IndependentVerifierRegistry",
    "VerificationContext",
    "VerificationError",
    "VerificationReport",
    "VerificationStatus",
    "DomainCapabilityVerification",
    "DomainCheckResult",
    "DomainDecision",
    "DomainEvidenceArtifact",
    "DomainVerificationCheck",
    "DomainVerificationEngine",
    "DomainVerificationError",
    "DomainVerificationReport",
    "DomainVerificationRequirement",
    "DomainVerificationStatus",
    "DomainVerifierRegistry",
    "CapabilityLearningEngine",
    "CapabilityLearningError",
    "CapabilityLearningObservation",
    "CapabilityLearningProposal",
    "AbstractedPair",
    "EpistemicStatus",
    "RepresentationConflictError",
    "RepresentationGraph",
    "RepresentationKind",
    "RepresentationNode",
    "RepresentationProvenance",
    "RepresentationRelation",
    "RepresentationUncertainty",
    "RepresentationValidationError",
    "UncertaintyKind",
    "canonical_json",
    "stable_representation_id",
    "ConstraintKind",
    "TaskConstraint",
    "TaskIntent",
    "TaskSpec",
    "TaskStatus",
    "TaskSubtask",
    "TaskUnderstandingEngine",
    "TaskUnderstandingError",
    "Blueprint",
    "ClaimEvidence",
    "ClaimEvidenceError",
    "ClaimEvidenceReport",
    "ClaimEvidenceVerifier",
    "BlueprintAnswerVerifier",
    "BlueprintVerificationReport",
    "BlueprintCompositionEngine",
    "BlueprintCompositionError",
    "BlueprintCompositionResult",
    "BlueprintDependencyGraph",
    "BlueprintKind",
    "BlueprintProvenance",
    "BlueprintValidationError",
    "EvidenceLevel",
    "VerificationEvidence",
    "ActiveCuriosityPlanner",
    "AutonomousTDDEngine",
    "ByteMarkovGrammar",
    "CASComplexityExplosionError",
    "CASResult",
    "CerebrumCrystalBridge",
    "CloudCrystalNode",
    "ContextBitTable",
    "CoreferenceAndSenseTracker",
    "CrystalResponse",
    "CrystalRule",
    "CurationConfig",
    "DiskCrystalVault",
    "DomainShardedVault",
    "MmapPostingsIndex",
    "DocumentAutoCrystallizer",
    "DreamUtilityPruner",
    "EpistemicConfidenceCalibrator",
    "FactFidelityGuard",
    "FluentByteRealizer",
    "HierarchicalProjectState",
    "LSHRuleIndex",
    "MirrorCrystalEngine",
    "NightlyDreamSynthesizer",
    "OnDevicePrivacyScrubber",
    "PNGSecurityError",
    "PackedSimdBitKernel",
    "PackedTernaryProjection",
    "PerceptualVQLens",
    "PocketCrystalClient",
    "ProcessRewardVerifier",
    "SandboxMCTSPlanner",
    "ScalableBloomFilter",
    "SensoryCodec",
    "ShardedCrystalRouter",
    "SignedRuleCapsule",
    "SixGateDataHygienePipeline",
    "SlotAbstractor",
    "StepEvaluation",
    "SymbolicCASOrgan",
    "TDDExecutionReport",
    "TernaryFluidComposer",
    "UniversalConceptHub",
    "WasmPocketCompiler",
    "ZeroTrustSandboxJudge",
    "curate_all_datasets",
    "split_compound_intents",
]

__all__ += [
    "UOL", "UOLError", "UOLExecutionError", "RuleStatus", "VariableSpec",
    "Lit", "Var", "Call", "If", "ArrayExpr", "RecordExpr", "Field", "Index",
    "Let", "MapExpr", "FoldExpr", "OperationalRule", "RuleCatalog", "Program",
    "Expression", "build_program", "expr_dict", "expr_from_dict",
    "program_dict", "program_from_dict", "knowledge_rule",
    "CapabilitySynthesizer", "Example", "SynthesisError", "SynthesisGoal",
    "SynthesisReport", "SynthesisStatus", "Counterexample",
    "KnowledgeRuleImporter", "KnowledgeRuleImportIssue", "KnowledgeRuleImportReport",
    "NativeDomainVerifierSuite", "NativeVerificationError",
    "ProgramLearningError", "UOLProgramHandler", "UOLProgramLearningProposal",
    "VerifiedProgramLearningEngine", "VerifiedProgramObservation",
    "CrossDomainTransferEngine", "CrossDomainTransferError", "CrossDomainTransferProposal",
]

__all__ += [
    "FailureDiagnosis", "FailureKind", "FailureRepairEngine", "FailureRepairError",
    "RepairProposal", "RepairStatus",
]

__all__ += [
    "ReasoningIntelligenceEngine", "ReasoningResult", "ReasoningStep",
    "AutonomousLearningEngine", "AutonomousLearningReport", "BenchmarkTaskResult",
    "LearnedCandidate", "ProductionEvaluationReport",
]

from .production_readiness import (
    ProductionConfig,
    StructuredLogger,
    MetricsCollector,
    CircuitBreakerState,
    CircuitBreakerOpenError,
    CircuitBreaker,
    TokenBucketRateLimiter,
    InferenceCapacityExceededError,
    InferenceResult,
    InferenceStabilizer,
    VaultSelfHealingManager,
    GracefulShutdownManager,
)

__all__ += [
    "ProductionConfig",
    "StructuredLogger",
    "MetricsCollector",
    "CircuitBreakerState",
    "CircuitBreakerOpenError",
    "CircuitBreaker",
    "TokenBucketRateLimiter",
    "InferenceCapacityExceededError",
    "InferenceResult",
    "InferenceStabilizer",
    "VaultSelfHealingManager",
    "GracefulShutdownManager",
]

from .safety_alignment import (
    SafetyCategory,
    RefusalContract,
    SafetyEvaluationResult,
    SafeRefusalEngine,
    PromptInjectionGuard,
    PrivacyProtectionEngine,
    ExecutionSafetyGuard,
    SafetyAlignmentManager,
)

__all__ += [
    "SafetyCategory",
    "RefusalContract",
    "SafetyEvaluationResult",
    "SafeRefusalEngine",
    "PromptInjectionGuard",
    "PrivacyProtectionEngine",
    "ExecutionSafetyGuard",
    "SafetyAlignmentManager",
]

from .evidence_intelligence import (
    EpistemicCategory,
    ContradictionType,
    EvidenceError,
    SourceProvenance,
    EvidencePassage,
    AtomicClaim,
    ContradictionConflict,
    ContradictionDetector,
    ClaimGroundingVerifier,
    ConfidenceCalibrator,
    EvidenceProvenanceGraph,
    GroundedSynthesisReport,
    EvidenceBasedIntelligenceEngine,
)

__all__ += [
    "EpistemicCategory",
    "ContradictionType",
    "EvidenceError",
    "SourceProvenance",
    "EvidencePassage",
    "AtomicClaim",
    "ContradictionConflict",
    "ContradictionDetector",
    "ClaimGroundingVerifier",
    "ConfidenceCalibrator",
    "EvidenceProvenanceGraph",
    "GroundedSynthesisReport",
    "EvidenceBasedIntelligenceEngine",
]

from .long_term_memory import (
    MemoryTier,
    MemoryProtectionError,
    MemoryError,
    MemoryNode,
    ForgettingCurveScheduler,
    MemoryConsolidationEngine,
    LongTermMemoryManager,
)

__all__ += [
    "MemoryTier",
    "MemoryProtectionError",
    "MemoryError",
    "MemoryNode",
    "ForgettingCurveScheduler",
    "MemoryConsolidationEngine",
    "LongTermMemoryManager",
]

from .advanced_reasoning import (
    ReasoningStepType,
    ReasoningError,
    ReasoningStep,
    CausalEdge,
    CausalReasoningGraph,
    ExactMathVerifier,
    HypothesisGenerator,
    IndependentAnswerChecker,
    AdvancedReasoningEngine,
)

__all__ += [
    "ReasoningStepType",
    "ReasoningError",
    "ReasoningStep",
    "CausalEdge",
    "CausalReasoningGraph",
    "ExactMathVerifier",
    "HypothesisGenerator",
    "IndependentAnswerChecker",
    "AdvancedReasoningEngine",
]

from .planning_and_tools import (
    StepStatus,
    ToolRiskProfile,
    PlanningError,
    ToolDefinition,
    ToolRegistry,
    TaskNode,
    ExecutionPlan,
    TaskDecomposer,
    FailureRecoveryManager,
    ResultVerificationContract,
    PlanningAndToolEngine,
)

__all__ += [
    "StepStatus",
    "ToolRiskProfile",
    "PlanningError",
    "ToolDefinition",
    "ToolRegistry",
    "TaskNode",
    "ExecutionPlan",
    "TaskDecomposer",
    "FailureRecoveryManager",
    "ResultVerificationContract",
    "PlanningAndToolEngine",
]



from .phase8_cross_domain_intelligence import (
    CrossDomainBenchmarkCase,
    CrossDomainBenchmarkError,
    CrossDomainBenchmarkManifest,
    CrossDomainBenchmarkReport,
    CrossDomainBenchmarkRunner,
    CrossDomainCaseResult,
)
from .phase9_continuous_expansion import (
    ContinuousExpansionEngine,
    ContinuousExpansionError,
    ContinuousExpansionProposal,
    ExpansionCycleEvidence,
)

__all__ += [
    "CrossDomainBenchmarkCase", "CrossDomainBenchmarkError",
    "CrossDomainBenchmarkManifest", "CrossDomainBenchmarkReport",
    "CrossDomainBenchmarkRunner", "CrossDomainCaseResult",
    "ContinuousExpansionEngine", "ContinuousExpansionError",
    "ContinuousExpansionProposal", "ExpansionCycleEvidence",
]

from .autonomous_research import (
    AutonomousResearchEngine,
    ExperimentDirection,
    ExperimentEvaluation,
    ExperimentObservation,
    ExperimentOutcomeStatus,
    ResearchError,
    ResearchExperimentDesign,
    ResearchHypothesis,
    ResearchQuestionPlan,
    ResearchQuestionSynthesizer,
    ResearchReport,
    ResearchSourceCandidate,
    ResearchStatus,
    evaluate_experiment,
)

__all__ += [
    "AutonomousResearchEngine",
    "ExperimentDirection",
    "ExperimentEvaluation",
    "ExperimentObservation",
    "ExperimentOutcomeStatus",
    "ResearchError",
    "ResearchExperimentDesign",
    "ResearchHypothesis",
    "ResearchQuestionPlan",
    "ResearchQuestionSynthesizer",
    "ResearchReport",
    "ResearchSourceCandidate",
    "ResearchStatus",
    "evaluate_experiment",
]


from .stage19_controlled_improvement import (
    ApprovalDecision,
    CaseExecutionResult,
    ChangeKind,
    ContainerRegressionExecutor,
    ControlledSelfImprovementEngine,
    EvaluationStatus,
    ExecutionEvidence,
    FrozenRegressionSuite,
    HumanApprovalRecord,
    ImprovementProposal,
    PromotionAuthorization,
    RegressionCase,
    RegressionCaseKind,
    SelfImprovementError,
    SelfImprovementEvaluation,
)

__all__ += [
    "ApprovalDecision",
    "CaseExecutionResult",
    "ChangeKind",
    "ContainerRegressionExecutor",
    "ControlledSelfImprovementEngine",
    "EvaluationStatus",
    "ExecutionEvidence",
    "FrozenRegressionSuite",
    "HumanApprovalRecord",
    "ImprovementProposal",
    "PromotionAuthorization",
    "RegressionCase",
    "RegressionCaseKind",
    "SelfImprovementError",
    "SelfImprovementEvaluation",
]

from .stage20_multimodal_grounding import (
    AssetProvenance,
    DocumentEvidenceSegment,
    ExtractedSegment,
    GroundedObservation,
    GroundingError,
    GroundingReport,
    GroundingStatus,
    Modality,
    ModalConclusion,
    RealWorldGroundingEngine,
    SourceTrustRegistry,
)

__all__ += [
    "AssetProvenance",
    "DocumentEvidenceSegment",
    "ExtractedSegment",
    "GroundedObservation",
    "GroundingError",
    "GroundingReport",
    "GroundingStatus",
    "Modality",
    "ModalConclusion",
    "RealWorldGroundingEngine",
    "SourceTrustRegistry",
]


from .stage21_grounded_action import (
    ActionCheckDecision,
    ActionCheckResult,
    ActionExecutionReport,
    ActionExecutionStatus,
    ActionFeedback,
    ActionRisk,
    ActionSandboxResult,
    ActionVerificationReport,
    ContainerActionExecutor,
    GroundedActionDefinition,
    GroundedActionEngine,
    GroundedActionError,
    GroundedActionProposal,
    GroundedActionRegistry,
    HumanActionApproval,
)

__all__ += [
    "ActionCheckDecision", "ActionCheckResult", "ActionExecutionReport",
    "ActionExecutionStatus", "ActionFeedback", "ActionRisk", "ActionSandboxResult",
    "ActionVerificationReport", "ContainerActionExecutor", "GroundedActionDefinition",
    "GroundedActionEngine", "GroundedActionError", "GroundedActionProposal",
    "GroundedActionRegistry", "HumanActionApproval",
]

from .stage22_trusted_execution import (
    ActionAdapterRequest,
    AdapterPhase,
    AdapterReceipt,
    DurableAuditLedger,
    FeedbackReviewReport,
    GatewayExecutionReport,
    GatewayExecutionStatus,
    LedgerVerification,
    ReviewerAuthenticator,
    ReviewerIdentity,
    Stage22Error,
    TrustedActionAdapter,
    TrustedActionAdapterRegistry,
    TrustedActionGateway,
)

__all__ += [
    "ActionAdapterRequest", "AdapterPhase", "AdapterReceipt",
    "DurableAuditLedger", "FeedbackReviewReport", "GatewayExecutionReport",
    "GatewayExecutionStatus", "LedgerVerification", "ReviewerAuthenticator",
    "ReviewerIdentity", "Stage22Error", "TrustedActionAdapter",
    "TrustedActionAdapterRegistry", "TrustedActionGateway",
]
