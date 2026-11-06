"""
Mirror 7 — Stage 9 (Metacognitive Intelligence Specialization) & Parameter Capacity / Overwrite Auditor.

Covers all 20 Metacognitive Categories:
1. Self-evaluation
2. Self-assessment
3. Solution verification
4. Answer verification
5. Error identification
6. Error correction
7. Confidence calibration
8. Uncertainty estimation
9. Assumption detection
10. Contradiction detection
11. Reasoning consistency
12. Alternative reasoning
13. Self-reflection
14. Answer quality evaluation
15. Evidence evaluation
16. Source reliability evaluation
17. Solution comparison
18. Failure analysis
19. Knowledge limitation recognition
20. Reconsideration and revision

Covers all 6+ Named Metacognitive & Process-Reward Datasets:
- ProcessBench (`Qwen/ProcessBench` — GSM8K, MATH, OlympiadBench, Omni-MATH step-level error localization)
- PRM800K (`tasksource/PRM800K` / `Birchlabs/openai-prm800k-stepwise-critic` / `trl-lib/prm800k`)
- HelpSteer2 (`nvidia/HelpSteer2` — multi-attribute correctness, coherence, helpfulness evaluation)
- Critique Datasets (`argilla/ultrafeedback-binarized-preferences-cleaned`, `allenai/reward-bench`)
- Self-Correction Datasets (`meta-math/MetaMathQA` backward/FOBAR verification & self-correction)
- Reward-Model Preference Datasets (`Intel/orca_dpo_pairs` unseen offset=350)

Includes `Mirror7CapacityAndOverwriteAuditor`:
- SVD Spectral Entropy Effective Rank (measures % of 185.5M parameter capacity utilized vs. free)
- 16-Expert MoE & 16,384-Slot Memory Bank Saturation Check
- Weight-Space Anchor Regularization (`L2_anchor`) + All-Stage Experience Replay to prevent data overwrite
- Cross-Stage Retention Audit across Communication, Thinking, Math, Quantum, Science, and Metacognition.
"""
from __future__ import annotations


TWENTY_METACOGNITIVE_CATEGORIES: tuple[str, ...] = (
    "01_self_evaluation",
    "02_self_assessment",
    "03_solution_verification",
    "04_answer_verification",
    "05_error_identification",
    "06_error_correction",
    "07_confidence_calibration",
    "08_uncertainty_estimation",
    "09_assumption_detection",
    "10_contradiction_detection",
    "11_reasoning_consistency",
    "12_alternative_reasoning",
    "13_self_reflection",
    "14_answer_quality_evaluation",
    "15_evidence_evaluation",
    "16_source_reliability_evaluation",
    "17_solution_comparison",
    "18_failure_analysis",
    "19_knowledge_limitation_recognition",
    "20_reconsideration_and_revision",
)
