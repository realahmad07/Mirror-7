"""Stage S9 Autonomous Learning Controller & Self-Curriculum package.

STRICT ISOLATION & PYTHON 3.11 COMPATIBILITY (PDF pp. 4–5, 7, 10, 12–14, 17):
- Pure Python 3.11 standard library ONLY (zero Python 3.12+ APIs).
- ZERO imports of ``hidden_actions``, ``oracle``, or ``generators``.
- ZERO hardcoded domain action, predicate, attribute, or entity-type literals.
"""
from __future__ import annotations

from research.cpu_learning.controller.curriculum import (
    CURRICULUM_SCHEMA_VERSION,
    AnomalyKind,
    AnomalySignal,
    ControllerFrozenError,
    CurriculumAction,
    CurriculumActionKind,
    CurriculumBudget,
    CurriculumFrozenError,
    FunctionalFrontierState,
    FunctionalSkillState,
    GroundingFrontierState,
    PassiveStreamController,
    RandomCurriculumController,
    RelationalFrontierState,
    SelfCurriculumScheduler,
    enumerate_consistent_slot_programs,
    sync_grounding_learner_to_kb,
)
from research.cpu_learning.controller.meta_controller import (
    CLOSED_LOOP_L0_L7_STEPS,
    META_CONTROLLER_SCHEMA_VERSION,
    AutonomousMetaController,
    ControllerTickTrace,
    FreezeGatedVerifierGate,
    IndependentVerificationReceipt,
    InteractiveLatencyMonitor,
    InteractiveQueryRecord,
    InteractiveQueryRequest,
)

__all__ = [
    "AnomalyKind",
    "AnomalySignal",
    "AutonomousMetaController",
    "CLOSED_LOOP_L0_L7_STEPS",
    "CURRICULUM_SCHEMA_VERSION",
    "ControllerFrozenError",
    "ControllerTickTrace",
    "CurriculumAction",
    "CurriculumActionKind",
    "CurriculumBudget",
    "CurriculumFrozenError",
    "FreezeGatedVerifierGate",
    "FunctionalFrontierState",
    "FunctionalSkillState",
    "GroundingFrontierState",
    "IndependentVerificationReceipt",
    "InteractiveLatencyMonitor",
    "InteractiveQueryRecord",
    "InteractiveQueryRequest",
    "META_CONTROLLER_SCHEMA_VERSION",
    "PassiveStreamController",
    "RandomCurriculumController",
    "RelationalFrontierState",
    "SelfCurriculumScheduler",
    "enumerate_consistent_slot_programs",
    "sync_grounding_learner_to_kb",
]
