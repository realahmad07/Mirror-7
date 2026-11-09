# Phase 1–2: Knowledge-to-UOL Integration and Counterexample-Guided Synthesis

## Phase 1 — Knowledge-to-UOL integration

Mirror now has a conservative import boundary from the canonical KnowledgeRecord / KnowledgeRegistry layer into the Universal Operational Language:

```python
from mirror_crystal import KnowledgeRuleImporter, UOL

report = KnowledgeRuleImporter().import_records(knowledge_registry)
language = UOL(report.rules)
```

Only records with an explicit, versioned content.uol_rule object are considered. The object declares a rule id, typed input ports, a serialized UOL AST, and optional domain scope. The importer does not compile natural-language prose, infer missing operators, or turn a retrieved fact into an executable procedure.

Each import report separates imported, skipped, and rejected records. It preserves source identifiers, fingerprints, source references, licence metadata, timestamps, parent lineage, epistemic state and verification metadata in the imported rule provenance. Evidence is mapped conservatively: asserted-but-unverified records remain UNKNOWN; assumptions remain ASSUMED; observed and derived records retain those states; formal/independent verification becomes VERIFIED only when passing checks, a named verifier and non-placeholder source provenance are present. Disputed claims and conflicting claim values become DISPUTED.

Conflicting definitions reusing a UOL rule id are quarantined. A dangling rule reference or recursive dependency invalidates the imported batch instead of exposing a partially valid dependency graph. Identical definitions with multiple sources retain combined lineage and the weakest evidence state.

## Phase 2 — Counterexample-guided synthesis (CEGIS)

The synthesizer supports a bounded CEGIS loop:

1. Search for a candidate using the current training set and a separate held-out validation set.
2. Query an explicit host-supplied oracle on a disjoint probe pool.
3. When a candidate disagrees with the oracle, record the input, expected output, actual output (or execution failure), and round.
4. Add that counterexample to the training constraints and synthesize again.
5. Stop only when a candidate passes the fixed held-out set and all remaining probes, or when an explicit search/round/counterexample limit is reached.

The oracle is deliberately supplied by the host. Mirror does not infer an answer from its own candidate and call that an independent counterexample. The held-out validation set is never moved into training, and probe inputs must be disjoint from both sets. Each search round has its own fixed candidate budget, while the maximum round count caps total search work; probe count and counterexample count are bounded.

## Trust boundary

CEGIS agreement is not proof. A returned program is still only a candidate. The existing capability adapter labels it UNVERIFIED, so it is not ordinarily executable or eligible for evidence-gated promotion until Mirror's independent verification path supplies qualifying evidence. A finite probe pool establishes agreement only on those tested inputs; it does not prove correctness for every input.

## Accepted scope and remaining limits

- Structured knowledge rules now flow into the same UOL catalog used by synthesis, with trust and lineage preserved.
- Natural-language-to-AST conversion is intentionally not automatic; existing prose records need an explicit reviewed conversion into the structured schema.
- CEGIS now uses observed counterexamples to refine candidates, but its search grammar remains bounded to the currently implemented primitive/rule expressions.
- It does not yet discover unrestricted control-flow programs, arbitrary code, general scientific theories, or new verifier oracles.
- Cross-domain generalization still requires independent domain-specific evaluation.

## Example: import and synthesize

```python
from mirror_crystal import (
    CapabilitySynthesizer, Example, KnowledgeRuleImporter, SynthesisGoal,
    UOL, VariableSpec,
)

import_report = KnowledgeRuleImporter().import_records(knowledge_registry)
language = UOL(import_report.rules)
goal = SynthesisGoal(
    name="new-procedure",
    domain="mathematics",
    inputs=(VariableSpec("x", "integer"),),
    output_type="integer",
    training=(Example({"x": 1}, 2), Example({"x": 2}, 4)),
    validation=(Example({"x": 3}, 6),),
)
report = CapabilitySynthesizer(language).synthesize(
    goal,
    counterexample_inputs=({"x": 4}, {"x": 5}),
    counterexample_oracle=lambda values: independent_reference(values["x"]),
)
if report.succeeded:
    candidate = CapabilitySynthesizer(language).to_candidate_capability(report)
    assert not candidate.executable
```
