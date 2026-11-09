# Backend integration for universal task planning and UOL synthesis

## What is connected

The default backend Q&A path invokes `UniversalIntelligencePipeline.analyze` and returns its trace under `universal_intelligence`. Normal Q&A requests also set `execute=true`: a ready plan is run only when the host has configured an allow-listed native execution engine, a separate independent verification engine, and explicit required checks. Without all three, the plan remains advisory and no step runs. The bridge performs:

1. Domain-neutral task interpretation, including intent, constraints, known inputs, context, outputs and actionable subtasks.
2. Conservative import of structured `KnowledgeRecord.content.uol_rule` entries into the UOL catalog, preserving the importer's provenance and evidence policy.
3. Typed matching against the persistent capability catalog.
4. Bounded capability composition into a dependency-ordered plan with explicit blockers and required postconditions.

This is additive to the existing Crystal retrieval, neural-symbolic reasoning, code-verification and claim-synthesis path. An advisory plan is not inserted into the verified factual evidence pool and is not presented as a completed result.

## API

### Q&A with universal trace

`POST /api/crystal/ask`

```json
{
  "prompt": "Build a calculator with addition and multiplication",
  "session_id": "studio_web",
  "universal_options": {
    "domain_hint": "mathematics",
    "known_inputs": {},
    "context": {},
    "acceptance_criteria": ["supports addition", "supports multiplication"]
  }
}
```

The response keeps the existing answer fields and adds `universal_intelligence`, containing the parsed task, imported-rule counts, capability matches, composition plan and explicit execution boundary.

### Analyze without asking the response engine

`POST /api/universal/analyze`

```json
{
  "prompt": "Design a beam for a specified load",
  "options": {
    "domain_hint": "engineering",
    "known_inputs": {"load": 1200, "area": 30}
  }
}
```

### Request bounded UOL synthesis

`POST /api/universal/synthesize`

```json
{
  "goal": {
    "name": "increment",
    "domain": "mathematics",
    "inputs": [{"name": "x", "value_type": "integer"}],
    "output_type": "integer",
    "training": [
      {"inputs": {"x": 1}, "expected": 2},
      {"inputs": {"x": 2}, "expected": 3}
    ],
    "validation": [
      {"inputs": {"x": 3}, "expected": 4}
    ],
    "max_depth": 1,
    "candidate_budget": 2000
  }
}
```

A successful request returns a typed UOL program and a candidate capability marked `UNVERIFIED`. The endpoint deliberately does not execute, verify, persist, register or promote the candidate. The HTTP form uses disjoint train/held-out examples; host-side Python callers may additionally supply the explicit oracle and probe pool required for the CEGIS loop.

## Trust and runtime boundaries

- Natural-language parsing is bounded and conservative; an inferred task contract is not proof that the request was fully understood.
- Natural-language knowledge is not silently compiled into UOL. Only explicit structured UOL rules are imported.
- Matching and composition never invent missing inputs or upgrade capability evidence.
- When configured, plan execution uses only host-registered native handlers. The engine is bounded and stops on the first failed step; it does not blindly retry operations.
- Successful execution is not enough: a separately registered verifier must pass every required check and declared postcondition before an output is marked verified.
- UOL synthesis is example-guided within the implemented finite grammar, not unrestricted program or theory invention.
- Candidate execution, independent domain verification, repeated-observation learning, explicit promotion and persistence remain separate gates.
- No arbitrary user-supplied source string is evaluated by this bridge, and this integration does not ingest or generate knowledge rules.

## Tests

`python -m pytest -q mirror7_backend/test_universal_intelligence.py`

The tests cover default-Q&A wiring, task decomposition/analysis, successful held-out synthesis, and rejection of overlapping training and validation examples.
