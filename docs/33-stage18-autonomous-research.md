# Stage 18 — Autonomous Research

## Goal and scope

Stage 18 adds an auditable research workflow to Mirror Crystal. It turns a research request into a stable plan, gathers source records through an explicit adapter boundary, delegates provenance and claim analysis to Stage 14, creates a structured report, and defines reproducible experiments that can be evaluated against predeclared criteria.

Implementation: `mirror_crystal/autonomous_research.py`.

The acceptance claim is deliberately bounded. Stage 18 is a research-orchestration and evidence-synthesis engine; it is not an unrestricted autonomous scientist.

## Workflow

1. **Question synthesis.** `ResearchQuestionSynthesizer` normalizes the domain and request, generates research objectives and subquestions, and binds optional falsifiable hypotheses and null hypotheses into a deterministic research ID.
2. **Source discovery adapter.** `AutonomousResearchEngine.investigate` calls a host-supplied `source_discoverer(plan)`. Each result must be a `ResearchSourceCandidate` containing a Stage 14 `SourceProvenance`, bound `EvidencePassage` records, and optional atomic claim statements. The engine does not silently crawl the Internet or invent source records.
3. **Provenance and evidence validation.** Stage 14 validates source URL/URN policy, supported SPDX license identifiers, timezone-aware retrieval timestamps, source-content SHA-256, passage digests and source/passage binding. Duplicate source and passage IDs are rejected.
4. **Claim synthesis.** Supplied claims are used when present; otherwise statements are extracted from the passages. The Stage 14 engine performs its configured grounding, contradiction and epistemic-status analysis. Unsupported or conflicted research is reported as an abstention or conflict rather than silently promoted to fact.
5. **Experiment design.** `ResearchExperimentDesign` records protocol steps, variables, controls, distinct train/validation/held-out manifest hashes, code and environment hashes, at least three distinct random seeds, a metric, direction and minimum effect.
6. **Experiment evaluation.** `evaluate_experiment` checks one baseline and one candidate observation per declared seed, unique run and artifact identifiers, the held-out manifest hash and environment hash. It calculates paired per-seed deltas, mean and sample standard deviation. The result is supported only when every seed meets the predeclared minimum effect.
7. **Structured reporting.** `ResearchReport` includes the question plan, evidence synthesis, source manifest, adapter errors, optional experiment design/evaluation, deterministic report ID and creation timestamp. Its source manifest is exposed as immutable mappings.

## Resource and integrity limits

The current engine enforces:
- Maximum 50 sources, 5 MB total source content and 10 MB total source-plus-evidence input.
- Maximum 500 passages and 500 explicit or extracted claims.
- Maximum 100 passages and 100 supplied claims per source.
- Bounded query, domain, objective, hypothesis, protocol, seed and observation collections.
- Streaming discovery consumption, so an infinite provider cannot be materialized without limit.
- Fail-closed handling for malformed candidate types, duplicate identifiers, digest mismatches and experiment records that do not match the declared held-out data/environment.

## Minimal integration shape

A provider adapter retrieves and reviews source material outside the engine, then returns validated candidates:

```python
from mirror_crystal.autonomous_research import AutonomousResearchEngine

engine = AutonomousResearchEngine()

def source_discoverer(plan):
    # Connect an approved search/index provider here.
    # Convert each retrieved item into SourceProvenance + EvidencePassage.
    return reviewed_candidates

report = engine.investigate(
    "Does the candidate improve held-out accuracy?",
    "machine learning",
    source_discoverer,
    hypotheses=reviewed_hypotheses,
    experiment_design=reviewed_experiment_design,
    experiment_observations=attested_observation_records,
)
```

This adapter contract keeps provider credentials, network policy, crawling rules and licensing decisions outside the core engine. A provider adapter is executable host code and must be trusted and sandboxed by its host as appropriate.

## Acceptance tests

The Stage 18 suite covers:
- Deterministic question synthesis and hypothesis/null-hypothesis binding.
- Provenance-preserving evidence synthesis and report IDs.
- Provider failure, partial discovery failure and honest abstention.
- Bad source digests, duplicate source/passage identities and malformed records.
- Paired multi-seed experiment evaluation, minimum-effect rejection, split reuse and wrong-environment rejection.
- Streaming source-count caps, byte and item budgets, bounded request inputs and immutable report manifests.

Run:

```bash
python -m pytest -q mirror_crystal/tests/test_stage18_autonomous_research.py
python -m pytest -q mirror_crystal mirror_eval
```

## Explicit limitations and next dependency

- No built-in web crawler, literature database connector or automatic source-provider is included. Source discovery is an injected host adapter.
- Experiment observations are supplied by the caller. Their hashes and metadata are validated, but the engine does not execute experiments or cryptographically attest the originating process.
- Stage 14 grounding uses lexical and n-gram heuristics; it does not prove semantic entailment or real-world truth.
- The module does not promote claims into the production knowledge/capability registries.
- Stage 19 must connect experiment execution to Stage 17's isolated execution boundary, attest run outputs, compare code candidates against frozen regression suites, and require human approval before deployment.

Stage 18 is therefore **PASS for bounded research planning, evidence ingestion/synthesis and experiment evaluation**, with network discovery and execution attestation explicitly delegated to future integrations.
