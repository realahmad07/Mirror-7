# Reliability hardening, synthesis, security and release gates

This release makes the backend more conservative and auditable. Passing CI verifies
software contracts; it does not establish broad language, factual, or coding competence.

## 1. Trust-aware answer path

`UnifiedCognitiveOrchestrator` now keeps cross-session memory in a separate
`memory_context` field. Memory is not eligible to become an answer and is never
counted as factual evidence. Automatic extraction runs only for explicit,
non-question configuration statements.

Crystal responses are classified by evidence path. Reviewed reliable blueprints
may enter claim synthesis; exact deterministic operations may enter through their
verified-execution path. A legacy Q/A rule does not become a verified fact merely
because a sandbox flag is true.

Cerebrum math results are exposed as `computed_candidates`, not independently
verified answers. PRM score and structurally valid steps are diagnostics, not a
second proof system. Until an independent mathematical verifier issues a
certificate, an uncorroborated result remains a candidate and the final answer
abstains.

## 2. Claim-level synthesis

`EvidenceSynthesisEngine` splits verified responses into sentence-level claim
candidates, deduplicates exact claims, records their supporting sources, and blocks
direct positive/negative polarity conflicts.

An optional synthesis provider can reorder or format those claims. Its output is
accepted only when it contains exactly the supplied supported sentences. New
sentences, omitted claims, and paraphrases are rejected and the deterministic
claim composition is used instead. This is an intentionally strict baseline;
semantic paraphrase validation and broader numeric/entity contradiction detection
remain future work.

## 3. Code generation and TDD

`BackendService` accepts an optional `code_generation_provider` with contract:

- Input: the user's natural-language coding request.
- Output: source-code string, or a mapping containing `code` / `final_code`.

The new `POST /api/tdd/generate` route runs provider output through the existing
TDD engine. A result is verified only when the report itself says verified, at
least one assertion was executed, and every generated assertion passed. A missing
provider, empty output, or zero-assertion suite fails closed.

The default HTTP server does not invent or silently load a code-generation model.
The host application must inject a real model adapter. Passing generated tests
does not prove all program behavior or security.

## 4. Memory encryption and API boundary

New sensitive memory writes use AES-256-GCM via the `cryptography` package,
a per-write random nonce, authenticated associated data, and scrypt key derivation.
Set `MIRROR_MEMORY_KEY` to a strong secret of at least 32 characters before
storing credentials, tokens, or other sensitive memory. Missing/weak keys fail
closed. The old custom `enc::` format is supported for compatibility reads only;
rewrite legacy records to migrate them. Startup clears the old plaintext
`value_json` duplicate for known sensitive categories/keys.

Example deployment configuration:

```bash
export MIRROR_MEMORY_KEY='<32+ character secret from a secrets manager>'
export MIRROR7_API_TOKEN='<32+ character bearer token from a secrets manager>'
export MIRROR7_HOST='0.0.0.0'
export MIRROR7_CORS_ORIGIN='https://your-frontend.example'
python -m mirror7_backend.http_server
```

Non-loopback binding is rejected without `MIRROR7_API_TOKEN` and a specific
(non-wildcard) CORS origin. When a token is configured, all API routes except
`GET /api/health` require `Authorization: Bearer <token>`. Health exposes only
liveness/readiness fields. This is a single shared bearer-token boundary, not
per-user identity, authorization, or tenant isolation; use an authenticated
reverse proxy / identity provider for multi-user deployments.

## 5. Evaluation improvements

The source-backed factual benchmark remains blocked pending two distinct
independent reviews per atomic claim and per case. No reviewer identities or
approvals are synthesized by software. Generate a review packet with:

```bash
python -m mirror_eval.review_packet \
  --dataset mirror_eval/data/source_backed_factual_v1.jsonl \
  --manifest mirror_eval/data/source_backed_factual_v1.manifest.json \
  --output artifacts/source_backed_factual_v1_review_packet.json
```

Reviewers must independently inspect the cited source and complete the review
fields. The built-in exact-fragment judge remains a baseline, not semantic
entailment. External review is still needed before publishing factual benchmark
scores as independent evidence.

## 6. Native free-running evaluation

Use a frozen JSONL prompt file containing `prompt` (or `user_text`) and an
optional `target` (or `target_text`). Run:

```bash
python -m training.free_running_eval \
  --checkpoint path/to/mirror7_checkpoint.pt \
  --prompts path/to/frozen_prompts.jsonl \
  --seeds 17,29,43 \
  --temperature 0.0 \
  --output artifacts/native_free_running_report.json
```

The report records the checkpoint SHA-256 and size, raw prompt-file SHA-256,
selected-case fingerprint, raw outputs and response hashes, empty-output rate,
lexical prompt coverage, repetition diagnostics, UTF-8 replacement characters,
termination, and exact match when targets exist. Duplicate seed IDs are rejected. These are generation diagnostics, not factual
or semantic correctness. Compare checkpoints only with identical frozen prompts,
seeds, decoding settings, and evaluator. Use measured failure clusters to select
the next training experiment (for example, scheduled-sampling sweeps); do not
silently alter model weights or claim improvement without a baseline comparison.

## Release gate

Before a public product release, require:

1. Full CI plus security-focused tests.
2. Independent review of the source-backed factual benchmark and an expanded,
   domain-balanced held-out set.
3. An independent claim-entailment evaluator and temporal/freshness cases.
4. Code-generation benchmarks with non-vacuous hidden tests, sandbox execution,
   and regression / security tests.
5. Native-model free-running comparison against a frozen baseline across seeds.
6. Public deployment through a hardened container/VM and an identity-aware proxy.

The current changes improve trust boundaries and evaluation infrastructure. They
do not turn every memory, math result, rule, generated program, or model output
into verified truth.
