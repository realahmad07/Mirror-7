# Mirror 7 Documentation

This is the human-facing documentation layer for the repository. Long subjects are split into pages instead of one giant README.

| Page | Purpose |
|---|---|
| [01 — Project overview](01-project-overview.md) | What was built during the 75+ hour phase and what is actually demonstrated. |
| [02 — System architecture](02-system-architecture.md) | Runtime, backend, response-generation, training, and research boundaries. |
| [03 — Backend](03-backend.md) | Service, HTTP API, lifecycle, persistence, action gateway, and response model. |
| [04 — Training](04-training.md) | Dataset, parameters, hardware, checkpoints, diagnostics, and failed routes. |
| [05 — Evaluation](05-evaluation.md) | Evidence levels, benchmark interpretation, and acceptance discipline. |
| [06 — Testing and CI](06-testing-ci.md) | Active checks, historical workflows, and release discipline. |
| [07 — Repository organization](07-repository-organization.md) | Where current code and historical research live. |
| [08 — Limitations](08-limitations.md) | Explicit research and product boundaries. |
| [09 — Roadmap](09-roadmap.md) | The next program after the first 75+ hour build. |
| [10 — Mirror Crystal weightless CPU engine](10-mirror-crystal-weightless-engine.md) | Standalone zero-weight, zero-gradient CPU crystallization engine (`mirror_crystal/`), `.jsonl`/`.txt` ingestion, CLI, and hardware benchmarks. |
| [11 — Mirror Crystal Phase-2, Phase-3 & Phase-4 evolution](11-mirror-crystal-phase2-phase3-evolution.md) | Multimodal `PerceptualVQLens`, `SandboxMCTSPlanner`, `CoreferenceAndSenseTracker`, `NightlyDreamSynthesizer`, `UniversalConceptHub`, and Phase-3/Phase-4 bottleneck defenses. |
| [12 — Mirror Crystal Phase-5 & Phase-6 bicameral, CAS, sensory & sharded scale](12-mirror-crystal-phase5-phase6-bicameral-scale.md) | `SymbolicCASOrgan`, `1.58`-bit `TernaryFluidComposer`, `ActiveCuriosityPlanner`, `DocumentAutoCrystallizer`, pure-stdlib `.png`/`.wav` `SensoryCodec`, `ShardedCrystalRouter`, `184`-byte `WasmPocketCompiler`, Proof-Carrying Capsules, and Phase-6 guards. |
| [13 — Qwen-Math Parity to Claude-Opus Frontier Evolution](13-mirror-qwen-to-opus-evolution.md) | Dual-Brain `CerebrumCrystalBridge` (0% math hallucination), `ProcessRewardVerifier` (11.8 µs PRM), `AutonomousTDDEngine` (test-driven self-repair in 2.5 ms), `HierarchicalProjectState`, and 6-gate data hygiene. |
| [14 — Canonical Blueprint and Rule Architecture](14-blueprint-rule-architecture.md) | Versioned blueprint schema, evidence levels, dependency composition, modularization status, and correctness roadmap. |
| [15 — Source evidence, benchmark and sandbox](15-source-evidence-benchmark-and-sandbox.md) | Legacy migration, claim-level review, frozen factual evaluation and container isolation. |
| [16 — Universal representation layer](16-universal-representation-layer.md) | Domain-neutral typed nodes, relations, epistemic status, uncertainty, provenance and disk-backed graph persistence. |
| [17 — Universal task understanding](17-task-understanding.md) | Request-to-TaskSpec interpretation, explicit constraints/context, ambiguity tracking and dependency-linked subtasks. |
| [18 — Capability contracts and registry](18-capability-contracts-and-registry.md) | Typed reusable operations, provenance, explicit verification evidence and durable registration. |
| [19 — Capability matching](19-capability-matching.md) | Domain/intent candidate ranking, typed input compatibility and evidence eligibility. |
| [20 — Capability composition planning](20-capability-composition-planning.md) | Typed dependency plans, port bindings, prerequisite gates and bounded search. |
| [21 — Controlled capability execution](21-controlled-capability-execution.md) | Bounded process execution of explicitly registered native handlers with fail-fast output validation. |
| [22 — Independent outcome verification](22-independent-outcome-verification.md) | Separate check registries, step-level validation and verified postcondition reporting. |
| [23 — Evidence-gated capability learning](23-evidence-gated-capability-learning.md) | Repeated verified compositions converted into explicit reusable capability proposals. |
| [24 — Universal knowledge foundation](24-knowledge-foundation.md) | Canonical knowledge records, assumption/evidence separation, source-backed fact admission, contradiction detection, and typed knowledge-to-capability bridging. |
| [25 — Domain-specific verification](25-domain-specific-verification.md) | Evidence artifacts bound to exact execution fingerprints, domain policies, checker identity floors, postconditions, and fail-closed outcomes. |
| [26 — Universal operational language and capability synthesis](26-universal-operational-language-and-synthesis.md) | Typed UOL, bounded native interpreter, structured rule integration, held-out synthesis and verification gates. |
| [27 — Knowledge-to-UOL and counterexample-guided synthesis](27-phase1-knowledge-to-uol-and-phase2-cegis.md) | Conservative canonical knowledge import, trust preservation, conflict quarantine, oracle-driven counterexample refinement and bounded CEGIS. |
| [Core pipeline acceptance and roadmap alignment](../PHASE1_9_ACCEPTANCE.md) | Nine engineering-layer test results plus honest status of the original strategic nine-phase roadmap. |
| [Mirror Crystal Live Studio UI](../mirror7_backend/studio.html) | Interactive dark-mode web application and API endpoints on port `8787` (`run_studio.bat`). |

## Research archive

The native-brain training track is split into individual experiment pages under research/:

- [Experiment 1](research/experiment-01.md)
- [Experiment 2](research/experiment-02.md)
- [Experiment 3](research/experiment-03.md)
- [Experiment 4](research/experiment-04.md)
- [Experiment 4B](research/experiment-04b.md)
- [Experiment 5](research/experiment-05.md)
- [Experiment 6](research/experiment-06.md)
- [Experiment 7](research/experiment-07.md)

## Documentation style

These pages deliberately use normal GitHub Markdown, tables, headings, code blocks, and links. No Mermaid diagrams or folded-arrow navigation are required.
