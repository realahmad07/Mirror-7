# Canonical Blueprint and Rule Architecture

## Purpose
Mirror's knowledge foundation should be a composable system of blueprints and rules. A blueprint describes a capability or transformation; rules provide executable or retrievable pieces used to realize it. Stored examples, retrieval similarity and legacy trust labels are not proof of truth.

## Canonical blueprint
The versioned contract in \`blueprint_contracts.py\` records kind, domain, intent, prompt/response templates, slots, preconditions, postconditions, dependencies, provenance, licensing and verification evidence. Legacy adapters preserve prior trust labels as metadata but always emit \`unverified\` evidence.

## Runtime and storage now implemented
- \`DiskCrystalVault\` remains the compatibility facade, with storage, indexing, retrieval and response behavior extracted into \`vault_storage.py\`, \`vault_indexing.py\`, \`vault_retrieval.py\` and \`vault_response.py\`.
- The existing SQLite file format is retained. Schema creation adds a separate \`blueprints\` table and a domain/intent index; existing rule rows are not destructively rewritten.
- \`store_blueprint\`, \`get_blueprint\`, \`list_blueprints\` and bounded \`backfill_legacy_blueprints\` provide canonical persistence. Backfill is paginated and every adapted legacy rule remains unverified.
- \`respond_blueprint\` is an explicit runtime entry point. It resolves dependencies in deterministic order, checks exact preconditions against externally verified condition tokens, and requires an explicit verifier for each postcondition. Formula blueprints with derived expressions require a caller-supplied safe executor; the runtime never calls Python \`eval\`.
- Missing dependencies, cycles, missing template values, unmet preconditions and unchecked postconditions block composition.

## Answer verification policy
\`BlueprintAnswerVerifier\` records source identity/reference, content digest, capture timestamp, freshness, evidence level and conflicting records. Conflicting blueprints for the same domain/intent cause abstention. Missing, stale, future-dated or digest-mismatched provenance prevents reliable labeling. Only formally verified or independently reproduced evidence can pass the evidence-level gate.

This verifier is an evidence-policy gate, **not an external fact-checking service**. It cannot independently establish that a source is authentic, that a cited page supports every sentence, or that a formal verifier was honestly operated. The normal legacy \`respond\` path remains conservative and retrieval-only factual matches remain moderate confidence; it is not automatically promoted by the new blueprint API.

## Execution isolation
\`ZeroTrustSandboxJudge\` still performs parent-process AST checks, then executes approved candidates in a separate Python child process with timeout and POSIX CPU, address-space, file-size and file-descriptor limits. It uses an isolated interpreter mode, a temporary working directory and a restricted import/builtin map.

A child process is stronger than same-process \`exec\`, but this is **not** a container, seccomp policy, network namespace, or a proof against Python/runtime escapes. Do not run hostile multi-tenant workloads with this as the only security boundary. Production hardening should add a container/VM boundary and network denial.

## Evaluation
New tests cover dependency ordering, missing prerequisites, unchecked postconditions, formula-executor requirements, persistent legacy backfill, freshness, contradictions, unverified evidence and distinct-process execution. A separate black-box evaluation module contains synthetic negative controls.

These are regression/negative-control tests, not a large independently curated factual benchmark. A true held-out factual evaluation still needs licensed source-backed examples, independent annotators/verifiers, multilingual/adversarial slices and a frozen test set. Report factual accuracy, unsupported-answer rate, abstention quality, calibration and latency separately.

## Compatibility and remaining boundaries
The storage split is a responsibility boundary; \`DiskCrystalVault\` remains the public import and legacy rule schema remains readable. Backfill is explicit and paginated to avoid long startup migrations. The classic \`respond\` path is retained for compatibility. Crystal can still retrieve stored examples, generalize selected numeric templates, run selected deterministic CAS operations, maintain session constraints, compose explicit blueprint chains and verify bounded code candidates. It is not a general-purpose factual knowledge engine or established general intelligence.