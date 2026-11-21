# Mirror 7 — Frozen CPU Intelligence Benchmark Report (v1)

- **Hardware / Mode:** CPU-only (`uses_gpu=False`, `uses_third_party_llm=False`)
- **Persisted Verified UOL Skills Loaded:** `120` (`artifacts/mirror_uol_skills.dcrystal`)
- **Overall Score:** **60 / 60 (100.0%)**
- **Hallucination Rate (Abstention Traps):** **0.0%** (`0/10`)
- **Median Task Latency:** **2.136 ms** (Mean: `42.323 ms`)
- **Result Fingerprint (SHA-256):** `f5cdc31ea605277c582f041b03c271ed8af5897bb3e884e6a50292114edb9df3`

## Category Breakdown

| Category | Passed | Accuracy | Median Latency (ms) |
|---|---:|---:|---:|
| 1. Forward UOL Formulas | 10 / 10 | **100.0%** | 1.175 |
| 2. Bidirectional Inverse Equations | 10 / 10 | **100.0%** | 0.236 |
| 3. Zero-Shot Multi-Hop Graph Planning (2–3 hops) | 10 / 10 | **100.0%** | 2.521 |
| 4. Multi-Turn Stateful UOL Dialogue | 10 / 10 | **100.0%** | 1.747 |
| 5. UOL Compositional Synthesis & TDD Code | 10 / 10 | **100.0%** | 139.664 |
| 6. Hallucination & Missing-Input Abstention Traps | 10 / 10 | **100.0%** | 2.457 |

## Scope & Boundary

This frozen benchmark evaluates deterministic mathematical/scientific formula execution, symbolic inverse solving, zero-shot multi-hop quantity graph planning, multi-turn stateful reasoning, bounded UOL program synthesis, TDD-verified beginner Python generation, and honest abstention on pure CPU with zero external LLMs. It does not claim general open-ended prose fluency or AGI.
