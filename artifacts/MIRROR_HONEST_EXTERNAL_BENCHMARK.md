# Mirror 7 — Honest External-Style & Adversarial CPU Benchmark

This benchmark intentionally tests **paraphrased/Roman-Urdu queries, unseen algebra & formal logic, GSM8K-style free-form word problems, and hard OOD traps** without cherry-picking. Every task is scored strictly into three buckets: **Solved Correctly**, **Honest Abstention (Safe)**, or **Wrong / Hallucinated (Unsafe)**.

- **Mode:** Pure CPU (`uses_gpu=False`, `uses_third_party_llm=False`)
- **Total Tasks:** `40`
- **Solved Correctly (Structured + Paraphrased + GSM8K):** `20 / 30` solvable tasks
- **OOD Trap Safe Abstention:** `10 / 10`
- **Overall Wrong / Hallucinated Rate:** **0.0%** (`0 / 40`)

| Category | Solved Correctly | Honest Abstain (Safe) | Wrong / Hallucinated | Median Latency (ms) |
|---|---:|---:|---:|---:|
| 1. Casual English & Roman-Urdu Paraphrases (10) | 9/10 | 1/10 | 0/10 | 3.603 |
| 2. Unseen Algebra, Systems & Formal Logic (10) | 9/10 | 1/10 | 0/10 | 3.556 |
| 3. GSM8K-Style Free-Form Word Problems (10) | 2/10 | 8/10 | 0/10 | 0.134 |
| 4. Hard OOD & Hallucination Traps (10, must abstain) | 0/10 | 10/10 | 0/10 | 2.697 |

## Honest Interpretation

- On **structured, paraphrased, Roman-Urdu quantitative physics/engineering, symbolic algebra, linear systems, and formal logic**, Mirror's CPU engine achieves high precision with **near-zero hallucination**.
- On **free-form GSM8K-style English word problems**, Mirror's pure-CPU heuristic `WordProblemSolver` solves some and **honestly abstains** on the rest instead of fabricating answers — clearly showing where a trained native neural semantic parser (Phase B, GPU) is still needed.
