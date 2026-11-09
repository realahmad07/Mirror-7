# UOL and Capability Synthesizer v1 — Acceptance Record

**Scope:** Add a domain-neutral typed operational representation, a bounded native interpreter, structured rule integration, and example-guided capability discovery.

## Components

| Component | Responsibility |
|---|---|
| `mirror_crystal/operational_language.py` | Typed UOL AST, typed rule catalog, structured knowledge bridge, serializer, bounded interpreter and native-handler adapter. |
| `mirror_crystal/capability_synthesizer.py` | Candidate generation from primitives and supplied rules; training/held-out checks; candidate capability adapter. |
| `mirror_crystal/tests/test_operational_language.py` | UOL type, runtime, serialization, trust and resource tests. |
| `mirror_crystal/tests/test_capability_synthesizer.py` | Discovery, rule composition, holdout and admission-gate tests. |

## Focused verification

Local isolated verification:

- `python -m py_compile mirror_crystal/operational_language.py mirror_crystal/capability_synthesizer.py` — PASS
- `python -m pytest -q mirror_crystal/tests/test_operational_language.py mirror_crystal/tests/test_capability_synthesizer.py` — **15 passed**

The test environment uses small local shims for existing Mirror contracts so the two new modules can be tested independently. The full repository CI run after integration is the authoritative compatibility gate.

## Safety and trust boundary

- UOL interprets structured data; it never executes Python source or arbitrary callbacks.
- Inputs, output size, collection size, expression depth and interpreted steps are bounded.
- Rule provenance and evidence status are preserved; unknown/disputed rules are blocked.
- Synthesized programs are candidates only. The existing capability adapter always labels them `UNVERIFIED`; existing verification and promotion rules remain in force.
- Held-out examples must be disjoint from synthesis examples and all must pass.

## Research boundary

This v1 is not unrestricted source-code generation or a natural-language compiler. The UOL runtime supports a broader expression vocabulary than the v1 search grammar. The synthesizer currently discovers bounded primitive compositions and compositions using typed rules supplied to Mirror. Unseen-domain transfer and open-ended self-improvement remain separate, unaccepted research goals.
