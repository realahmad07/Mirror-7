"""
Mirror Crystal — Zero-Weight, Zero-Gradient, Surprise-Driven Byte Crystallization Engine.
Author: Ahmad Hassan | Repository: https://github.com/realahmad07/Mirror-7 | License: Apache-2.0
"""
from __future__ import annotations

from .bit_signatures import ContextBitTable
from .dataset_curator import CurationConfig, curate_all_datasets
from .disk_vault import DiskCrystalVault
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

__all__ = [
    "AbstractedPair",
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
