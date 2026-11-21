# Mirror 7 — Frozen CPU Intelligence Benchmark Report (v1)

- **Hardware / Mode:** CPU-only (`uses_gpu=False`, `uses_third_party_llm=False`)
- **Persisted Verified UOL Skills Loaded:** `120` (`artifacts/mirror_uol_skills.dcrystal`)
- **Overall Score:** **60 / 60 (100.0%)**
- **Hallucination Rate (Abstention Traps):** **0.0%** (`0/10`)
- **Median Task Latency:** **2.502 ms** (Mean: `54.566 ms`)
- **Result Fingerprint (SHA-256):** `e416aad1f5237c0a3cf0f72ccfe603fda8d910bef15c94d5294fd0127001baf3`

## Category Breakdown

| Category | Passed | Accuracy | Median Latency (ms) |
|---|---:|---:|---:|
| 1. Forward UOL Formulas | 10 / 10 | **100.0%** | 1.36 |
| 2. Bidirectional Inverse Equations | 10 / 10 | **100.0%** | 0.234 |
| 3. Zero-Shot Multi-Hop Graph Planning (2–3 hops) | 10 / 10 | **100.0%** | 3.101 |
| 4. Multi-Turn Stateful UOL Dialogue | 10 / 10 | **100.0%** | 2.167 |
| 5. UOL Compositional Synthesis & TDD Code | 10 / 10 | **100.0%** | 212.193 |
| 6. Hallucination & Missing-Input Abstention Traps | 10 / 10 | **100.0%** | 3.755 |

## Scope & Boundary

This frozen benchmark evaluates deterministic mathematical/scientific formula execution, symbolic inverse solving, zero-shot multi-hop quantity graph planning, multi-turn stateful reasoning, bounded UOL program synthesis, TDD-verified beginner Python generation, and honest abstention on pure CPU with zero external LLMs. It does not claim general open-ended prose fluency or AGI.
