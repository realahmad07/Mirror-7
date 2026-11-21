# Mirror 7 — Frozen CPU Intelligence Benchmark Report (v1)

- **Hardware / Mode:** CPU-only (`uses_gpu=False`, `uses_third_party_llm=False`)
- **Persisted Verified UOL Skills Loaded:** `120` (`artifacts/mirror_uol_skills.dcrystal`)
- **Overall Score:** **60 / 60 (100.0%)**
- **Hallucination Rate (Abstention Traps):** **0.0%** (`0/10`)
- **Median Task Latency:** **1.735 ms** (Mean: `37.671 ms`)
- **Result Fingerprint (SHA-256):** `7c7017b093bb2c7bcbfcc3e7f2895d0b1f34b5033690e98ab1d2ce39645a0e22`

## Category Breakdown

| Category | Passed | Accuracy | Median Latency (ms) |
|---|---:|---:|---:|
| 1. Forward UOL Formulas | 10 / 10 | **100.0%** | 0.935 |
| 2. Bidirectional Inverse Equations | 10 / 10 | **100.0%** | 0.195 |
| 3. Zero-Shot Multi-Hop Graph Planning (2–3 hops) | 10 / 10 | **100.0%** | 2.521 |
| 4. Multi-Turn Stateful UOL Dialogue | 10 / 10 | **100.0%** | 1.395 |
| 5. UOL Compositional Synthesis & TDD Code | 10 / 10 | **100.0%** | 146.42 |
| 6. Hallucination & Missing-Input Abstention Traps | 10 / 10 | **100.0%** | 2.47 |

## Scope & Boundary

This frozen benchmark evaluates deterministic mathematical/scientific formula execution, symbolic inverse solving, zero-shot multi-hop quantity graph planning, multi-turn stateful reasoning, bounded UOL program synthesis, TDD-verified beginner Python generation, and honest abstention on pure CPU with zero external LLMs. It does not claim general open-ended prose fluency or AGI.
