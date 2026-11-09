# Progressive Rule Admission and Evidence Policy

## Objective

Scale Mirror's knowledge base toward 10,000 high-quality candidate rules per domain without turning "unverified" into a blanket rejection and without weakening factual reliability.

## Key distinction

**Admission to the candidate library is not promotion to trusted factual knowledge.** A record can be useful for retrieval, hypothesis generation, or an explicitly qualified answer before it is fully verified. It must retain its evidence label and must not be represented as a verified fact.

## Typed admission lanes

| Lane | Eligible content | Admission checks | Permitted runtime use |
|---|---|---|---|
| A: procedural/heuristic | Problem-solving strategies, workflows, checklists, heuristics | Schema, atomicity, deduplication, scope, safety, provenance | Retrieve as a suggested method; qualify as heuristic |
| B: executable | Code, algorithms, formulas with executable semantics | Lane A plus actual sandbox tests, edge cases, complexity checks | Execute only in sandbox; cite test status |
| C: source-backed factual | Scientific, historical, general factual claims | Source URL, exact locator, quote/hash, scope/date, license review, claim entailment review | Use with source attribution and freshness caveats |
| D: formal/critical | Safety-critical, legal, medical, high-impact or theorem-level claims | Lane C plus domain-appropriate expert/formal independent review | Use only after the relevant high-assurance gate |

## Do not weaken these guarantees

- A source URL or a model's confidence is not proof that a source entails a claim.
- Syntax/AST checks do not establish algorithmic correctness.
- A generated test is not a passed test until executed.
- A heuristic can be useful without being promoted to factual truth.
- Conflicting, stale, unlicensed, unsafe, or scope-ambiguous records remain quarantined.
- Never silently upgrade legacy trust tiers or overwrite evidence history.

## Better algorithm: admit, route, verify, compose

1. **Extract:** Convert a source into atomic candidate blueprints with exact provenance and license metadata.
2. **Classify:** Assign kind and lane (A-D); distinguish facts, procedures, formulas, policies, and heuristics.
3. **Validate structure:** Schema, required fields, type constraints, stable IDs, domain membership, size limits.
4. **Deduplicate:** Exact hash, normalized text, and semantic similarity; preserve distinct versions when scope/date differs.
5. **Check dependencies:** Resolve IDs, detect cycles, validate dependency types and domain boundaries.
6. **Test by kind:** Execute code; property-test algorithms; recompute formulas; check formal proofs; review source entailment for factual claims.
7. **Admit provisionally:** Lane A may enter the searchable candidate/heuristic store after structural and safety checks, with evidence status unchanged. Lane B may enter after tests pass. Lane C/D remain restricted until their evidence gates pass.
8. **Compose at query time:** Retrieve multiple relevant records, rank by intent/domain/scope/evidence/freshness, resolve dependencies, detect conflicts, and synthesize an answer with claim-level support. Do not answer from one matching rule by default.
9. **Verify the composed answer:** Check each factual claim against its evidence; execute calculations/code independently; identify unsupported parts; qualify or abstain only on the unsupported portion.
10. **Learn from outcomes:** Record user corrections, failed tests, retrieval misses, and conflicts as new candidate work items; never auto-promote feedback to truth.
11. **Evaluate:** Run frozen held-out, adversarial, cross-domain, and freshness tests after each batch. Track accuracy, coverage, unsupported-claim rate, abstention precision, calibration, duplicate rate, and cost/latency.

## Failure handling

Do not stop an entire domain because one candidate fails. Quarantine the failed record, preserve its diagnostic, continue independent candidates, and stop only the affected dependency chain or batch when a systemic issue is detected (schema corruption, provenance failure, unsafe source, or evaluator defect). Re-run regression tests after any policy or code change.

## Batch schedule

Work in bounded batches (for example 100-250 candidate records per domain per batch). After each batch publish:
- accepted candidate count by lane and evidence status;
- rejected/quarantined counts with reason codes;
- duplicates removed;
- tests actually executed and pass rates;
- source/license coverage;
- evaluation deltas on the frozen benchmark;
- unresolved dependencies and next batch scope.

The 10,000/domain figure is a coverage target, not a quality guarantee. Stop adding records to a subdomain when new batches stop improving held-out task coverage or introduce unacceptable noise.
