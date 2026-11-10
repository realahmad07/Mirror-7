# Stage 11 — Generalization & Intelligence Evaluation Plan

**Status:** Initiated — evaluation design and baseline freeze. Not yet passed or completed.

## Objective

Determine whether Mirror generalizes to unfamiliar tasks beyond its existing stored rules and specialized handlers. Stage 11 is an evaluation-first phase: no new reasoning architecture should be promoted merely because its own unit tests pass.

## Workstreams

1. **Frozen baseline:** Record the exact Git commit, production-vault checksum, configuration, runtime mode, benchmark digests, and hardware/software environment before testing.
2. **Unseen mathematical tasks:** Generate parameterized algebra, calculus, linear systems, combinatorics, probability, and numerical problems from held-out generators. Verify results with exact rational or symbolic oracles where possible.
3. **Novel reasoning compositions:** Test multi-hop deductions and combinations of domains that were not used to implement the handlers. Include counterexamples and invalid-premise controls.
4. **Grounded factual answering:** Require each factual answer to identify supporting claims and source passages. Score factual accuracy, coverage, unsupported claims, abstention precision/recall, and calibration separately. Do not count an abstention on an answerable item as a correct factual answer.
5. **Continual learning and memory:** Test verified knowledge insertion, conflict handling, retention of prior anchors, rollback, and session isolation.
6. **Adversarial robustness:** Test prompt injection, misleading premises, ambiguous requests, contradictory sources, tool misuse, and secret-extraction attempts.
7. **Operational generalization:** Measure latency, memory, concurrency, recovery, and reproducibility on a clean environment with the production vault.

## Test protocol

- Three difficulty levels per capability.
- At least three random seeds for generated task families.
- Held-out problem templates and unseen parameter combinations.
- Positive cases plus adversarial/negative controls where applicable.
- Freeze test inputs before evaluation; never train or tune on the hidden test split.
- A failed gate stops promotion. Fix the smallest justified defect and rerun the full regression suite.

## Metrics

Publish per-domain and aggregate results for exact accuracy, semantic claim support, answer coverage, abstention precision/recall, unsupported-answer rate, confidence calibration, latency, memory, retention, and session leakage. Report sample counts and confidence intervals; do not publish a single accuracy number without the denominator and case-level outcomes.

## Promotion gates

- Mathematical cases must match independent exact oracles.
- Factual answers must be supported by the cited primary-source passages; unsupported answers fail.
- Negative controls must abstain without suppressing answerable questions.
- Learning may promote only independently verified candidates; rollback and anchor-retention checks must pass.
- Session isolation must show zero cross-session data leakage.
- Results must be reproducible from the recorded commit, frozen data, and test commands.

## Current limitations

The Stage 10.5.1 technical scan/CI gate is separate from real-world reviewer identity authentication. The repository's Ed25519 signatures and checksums establish local cryptographic consistency, not institutional identity. External reviewer identity and institutional-root ownership remain explicitly unverified until authoritative, independently corroborated evidence is supplied.

The previous source-backed factual report showed poor answer coverage and factual accuracy; Stage 11 must first reproduce that baseline before claiming improvement. The baseline is not a pass criterion by itself.

## Deliverables

- Frozen Stage 11 benchmark manifest and task generators.
- Independent deterministic oracles and adversarial controls.
- Case-level predictions and scoring JSON.
- Failure taxonomy and targeted remediation report.
- Reproducible regression and release-gate results.

**Stage 11 exit criterion:** Demonstrated generalization on frozen unseen tasks with independent verification—not merely passing predefined handler examples.
