"""
Mirror 7 Honest Independent-Architecture CPU Benchmark Suite (`mirror_eval`).

Evaluates Mirror 7 purely inside its existing independent pure-Python CPU
architecture (zero weights, zero GPU, zero external LLMs) across four pillars:
1. Compositional & Unseen Symbolic Math (`CerebrumCrystalBridge`)
2. Non-Vacuous TDD Code Verification & Repair (`AutonomousTDDEngine`)
3. Vault Precision & Epistemic Abstention Calibration (`DiskCrystalVault`)
4. Strict Zero-LLM / Pure-Python Architecture Independence Audit
"""
from __future__ import annotations

from .runner import (
    audit_architecture_independence,
    evaluate_math_benchmark,
    evaluate_tdd_benchmark,
    evaluate_vault_benchmark,
    format_eval_report,
    run_full_eval,
)

__all__ = [
    "audit_architecture_independence",
    "evaluate_math_benchmark",
    "evaluate_tdd_benchmark",
    "evaluate_vault_benchmark",
    "format_eval_report",
    "run_full_eval",
]
