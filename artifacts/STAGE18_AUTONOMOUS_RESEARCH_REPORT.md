# Stage 18 — Autonomous Research Acceptance Report

**Status:** PASS within the bounded adapter-backed scope  
**Code:** `mirror_crystal/autonomous_research.py`  
**Tests:** `mirror_crystal/tests/test_stage18_autonomous_research.py`

## Delivered

- Deterministic research question synthesis and optional falsifiable/null hypothesis definitions.
- Host-adapter source discovery with bounded streaming consumption.
- Stage 14 provenance, passage-binding, claim-grounding, contradiction and epistemic-status integration.
- Structured immutable research report with deterministic report fingerprint.
- Reproducible experiment design and paired multi-seed comparison with held-out manifest and environment checks.
- Resource caps for sources, text/evidence, passages, claims, objectives, hypotheses, seeds and experiment observations.

## Verification gates

The focused suite includes positive, negative and boundary tests for source digest mismatch, duplicate identifiers, discovery errors, infinite-source protection, resource ceilings, immutable report manifests, experiment pairing, minimum-effect rejection, data-split reuse and environment mismatch.

Execution commands:

```bash
python -m pytest -q mirror_crystal/tests/test_stage18_autonomous_research.py
python -m pytest -q mirror_crystal mirror_eval
```

CI results must be attached to the exact accepted commit before treating this report as independently verified by the repository workflows.

## Honest boundaries

Stage 18 does not include a built-in network crawler or literature-provider integration. The caller supplies a trusted source-discovery adapter and source records. The experiment evaluator validates caller-supplied observations; it does not run code or prove that the referenced experiment ran. Stage 14 semantic grounding remains heuristic. These gaps are explicit dependencies for Stage 19, not hidden claims of completion.

No automatic knowledge-registry promotion or self-modification occurs in this stage.
