"""
Mirror Crystal — Zero-Weight, Zero-Gradient, Surprise-Driven Byte Crystallization Engine.
Author: Ahmad Hassan | Repository: https://github.com/realahmad07/Mirror-7 | License: Apache-2.0
"""
from __future__ import annotations

from .bit_signatures import ContextBitTable
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
    "ByteMarkovGrammar",
    "CASComplexityExplosionError",
    "CASResult",
    "CloudCrystalNode",
    "ContextBitTable",
    "CoreferenceAndSenseTracker",
    "CrystalResponse",
    "CrystalRule",
    "DiskCrystalVault",
    "DocumentAutoCrystallizer",
    "DreamUtilityPruner",
    "EpistemicConfidenceCalibrator",
    "FactFidelityGuard",
    "FluentByteRealizer",
    "LSHRuleIndex",
    "MirrorCrystalEngine",
    "NightlyDreamSynthesizer",
    "OnDevicePrivacyScrubber",
    "PNGSecurityError",
    "PackedSimdBitKernel",
    "PackedTernaryProjection",
    "PerceptualVQLens",
    "PocketCrystalClient",
    "SandboxMCTSPlanner",
    "SensoryCodec",
    "ShardedCrystalRouter",
    "SignedRuleCapsule",
    "SlotAbstractor",
    "SymbolicCASOrgan",
    "TernaryFluidComposer",
    "UniversalConceptHub",
    "WasmPocketCompiler",
    "ZeroTrustSandboxJudge",
    "split_compound_intents",
]
