# Mirror 7 Algorithm Bundle — Algorithm Index

## Summary Table

| # | Algorithm | Purpose | Inputs | Outputs | Core Mechanism | Complexity | Memory | Failure Behavior | Integration Dependencies |
|---|-----------|---------|--------|---------|----------------|------------|--------|-----------------|-------------------------|
| 1 | Open-World Concept Discovery | Discover categories from unstructured observations | Feature dicts | Concept tree, classifications | Incremental concept tree with category utility | O(k*d) per obs, k=concepts, d=features | O(k*d) bounded | Unknown features -> partial match, uncertain | None (feeds 5, 9) |
| 2 | Compositional Program Induction | Learn reusable procedures from examples | Input-output example pairs | Synthesized programs, library | Type-guided beam search over primitive compositions | O(b^d * n) b=beam, d=depth, n=examples | O(b*d + L) L=library | No solution -> report failure with partial matches | None (fed by 11) |
| 3 | Causal Experiment Designer | Distinguish competing causal explanations | Hypotheses, observations | Experiment suggestions, posteriors | Bayesian hypothesis discrimination with EIG | O(H*E*O) H=hyp, E=experiments, O=outcomes | O(H*V^2) V=variables | All hypotheses refuted -> report, suggest new | None (feeds 10) |
| 4 | Adaptive World-Model Builder | Learn environment dynamics | State-action-state transitions | Predictions, learned rules | Rule-based transitions with surprise-driven revision | O(R*F) per step, R=rules, F=features | O(R*F) bounded | Wrong prediction -> revise rules, track surprise | None (fed by 5) |
| 5 | Semantic Memory Graph | Store knowledge with confidence and provenance | Nodes, edges, queries | Query results, contradictions | Typed knowledge graph with confidence propagation | O(V+E) traversal, O(E) contradiction check | O(V+E) bounded | Missing info -> low confidence, partial results | None (central store) |
| 6 | Long-Term Goal Manager | Manage goals, subgoals, and plans | Goals, progress updates | Action plans, next goal | Goal tree with priority scheduling | O(G log G) prioritization, G=goals | O(G) bounded | Goal failure -> replan, reduce feasibility | None (uses 4, 10) |
| 7 | Self-Debugging Algorithm | Detect and propose fixes for recurring failures | Failure records | Failure patterns, candidate patches | Statistical failure analysis + info gain root cause | O(F*C) F=failures, C=context features | O(F) bounded | No pattern found -> report uncertainty | None (monitors all) |
| 8 | Knowledge Consolidation Engine | Convert experiences to durable knowledge | Experiences (observations) | Consolidated knowledge, schemas | Two-tier memory with promotion and consistency checking | O(B*K) B=buffer, K=knowledge | O(B+K) bounded | Contradictions -> quorum voting, confidence reduction | None (feeds 5) |
| 9 | Multimodal Grounding Algorithm | Link observations across modalities | Multi-modal token streams | Grounded concepts, cross-modal bindings | Co-occurrence alignment with normalized PMI | O(T^2) per timestep, T=tokens | O(C+P) C=concepts, P=pairs | No alignment -> report low confidence | None (feeds 1, 5) |
| 10 | Autonomous Research / Learning | Decide what to learn next | Questions, facts, goals | Research actions, priorities | Uncertainty-driven active learning with epsilon-greedy | O(Q) per decision, Q=questions | O(Q+F) F=facts | No productive action -> explore randomly | None (uses 3, 6) |
| 11 | Abstract Reasoning Engine | Solve novel problems via structural reasoning | Structured problem examples | Mappings, transformations, solutions | Structure mapping + transformation discovery | O(E^2 * R) E=entities, R=relations | O(E^2) per structure | No mapping found -> report with partial score | None (feeds 2) |
| 12 | General Cognitive Workspace | Coordinate all cognitive processes | Workspace items from all modules | Focused items, processor routing | Bounded blackboard with priority scheduling | O(I log I) per tick, I=items | O(I) bounded | Processor error -> catch, log, continue | Coordinates all |

## Detailed Algorithm Cards

---

### Algorithm 01 — Open-World Concept Discovery

**Directory**: `mirror7_algorithm_01/`

**Problem**: Given a stream of observations (feature dictionaries), discover
meaningful categories without being told how many categories exist or what
features matter.

**Mechanism**: Incremental concept tree where each node stores feature
distributions. New observations are classified by traversing the tree. The
tree structure is modified (create, split, merge) to maximize category
utility — a measure of how much knowing the category improves feature
prediction.

**Key Formula**: `CU = (1/k) * sum_j[sum_i[P(f_i|C_j)^2 - P(f_i)^2]]`

**Failure Modes**: Unknown features (partial match), ambiguous observations
(multiple concepts match equally), concept drift (distribution changes).

---

### Algorithm 02 — Compositional Program Induction

**Directory**: `mirror7_algorithm_02/`

**Problem**: Given input-output examples, discover a composition of primitive
operations that explains the transformation. Build a reusable library of
learned procedures.

**Mechanism**: Type-guided beam search over compositions of typed primitives.
Programs are expression trees. Scoring is based on fraction of examples
correctly mapped. Successful programs are added to the library for future
reuse, enabling hierarchical program construction.

**Key Feature**: Library learning — discovered programs become new primitives,
enabling increasingly complex compositions.

**Failure Modes**: No solution within search budget (report partial matches),
contradictory examples (detect and report), ambiguous programs (return all
candidates with scores).

---

### Algorithm 03 — Causal Experiment Designer

**Directory**: `mirror7_algorithm_03/`

**Problem**: Given competing causal explanations for observed data, choose
experiments that most efficiently distinguish between them.

**Mechanism**: Bayesian hypothesis discrimination. Each hypothesis is a
causal graph with conditional probability tables. Posteriors are updated
via Bayes' rule. Experiments are selected by maximizing Expected Information
Gain (EIG) — the expected reduction in entropy over hypothesis posteriors.

**Key Formula**: `EIG(e) = H(hypotheses) - E[H(hypotheses | outcome of e)]`

**Failure Modes**: All hypotheses refuted (report, suggest new), hypotheses
empirically indistinguishable (report, suggest alternative experiments),
zero-likelihood data (use small epsilon for numerical stability).

---

### Algorithm 04 — Adaptive World-Model Builder

**Directory**: `mirror7_algorithm_04/`

**Problem**: Learn the dynamics of an unfamiliar environment from
state-action-state transitions and continuously improve predictions.

**Mechanism**: Rule-based transition model. Rules have conditions (feature
values), an action trigger, and effects (feature changes). Predictions
use the highest-confidence matching rule. When predictions fail, rules are
revised: confidence decreases for wrong rules, new more-specific rules
are created, and generalizations are attempted.

**Key Feature**: Surprise-driven revision — prediction errors trigger
selective rule updates rather than relearning everything.

**Failure Modes**: Stochastic environments (track frequency, report
uncertainty), rule conflicts (use confidence ranking), model collapse
(minimum rule set preserved).

---

### Algorithm 05 — Semantic Memory Graph

**Directory**: `mirror7_algorithm_05/`

**Problem**: Represent and query relationships between concepts, facts,
events, and experiences with confidence, provenance, and contradiction
awareness.

**Mechanism**: Typed knowledge graph with confidence-weighted nodes and
edges. Supports multi-hop reasoning with confidence propagation
(chain confidence = product). Detects contradictions via conflicting
property assertions. Memory bounded with importance-based eviction.

**Key Feature**: Provenance tracking — every fact records where it came
from, enabling trust assessment and selective revision.

**Failure Modes**: Missing nodes (return empty results), contradictions
(flag for resolution), memory pressure (evict lowest-utility nodes).

---

### Algorithm 06 — Long-Term Goal Manager

**Directory**: `mirror7_algorithm_06/`

**Problem**: Maintain, decompose, prioritize, and track multiple goals
over long time periods, adapting when goals fail or the world changes.

**Mechanism**: Goal tree with priority scheduling. Each goal has status,
priority, urgency, feasibility, and deadline. Effective priority combines
these factors. Goals can be decomposed into subgoals. Failure triggers
replanning: retry with lower feasibility, try alternative decomposition,
or escalate to parent.

**Key Formula**: `priority = base * urgency_factor * dependency_factor * feasibility`

**Failure Modes**: Circular dependencies (detect and break), all goals
failing (suspend, report), resource exhaustion (prioritize and defer).

---

### Algorithm 07 — Self-Debugging Algorithm

**Directory**: `mirror7_algorithm_07/`

**Problem**: Detect recurring failures in algorithmic processes, locate
likely causes, propose candidate fixes, and validate them without blindly
modifying the system.

**Mechanism**: Statistical failure analysis. Failures are logged and
clustered by (component, error_type). Information gain identifies context
features most correlated with failures. Candidate patches (parameter
adjustments, input filters, fallback rules) are tested against failing
cases, passing cases (regression), and held-out cases before adoption.

**Key Feature**: NEVER auto-applies fixes — always proposes for external
validation.

**Failure Modes**: Random failures (no pattern detected, report uncertainty),
validation failure (reject patch), multiple correlated causes (report all).

---

### Algorithm 08 — Knowledge Consolidation Engine

**Directory**: `mirror7_algorithm_08/`

**Problem**: Convert repeated experiences into durable knowledge while
resisting noise, contradictions, and memory explosion.

**Mechanism**: Two-tier memory (short-term buffer + long-term store).
Experiences are clustered by similarity. Clusters exceeding a repetition
threshold are promoted to durable knowledge. Contradictions require
stronger evidence to override existing knowledge. Schemas are extracted
from clusters with varying details.

**Key Feature**: Noise resistance — single observations don't become
durable knowledge; consensus is required.

**Failure Modes**: Equal contradictory evidence (flag ambiguity), memory
full (evict lowest-utility knowledge), schema overgeneralization (track
and refine).

---

### Algorithm 09 — Multimodal Grounding Algorithm

**Directory**: `mirror7_algorithm_09/`

**Problem**: Create common internal representations linking observations
from different modalities (language, visual, symbolic, sensor).

**Mechanism**: Cross-modal binding via temporal co-occurrence. Tokens from
different modalities that frequently co-occur receive high alignment
scores (normalized PMI). Strong alignments create grounded concepts that
link tokens across modalities. Feature representations are merged into
shared centroids.

**Key Formula**: `alignment(a,b) = co_occurrence(a,b) / sqrt(count(a) * count(b))`

**Failure Modes**: Spurious co-occurrence (require high threshold), no
grounding available (return empty with low confidence), one-to-many
mappings (track all bindings with scores).

---

### Algorithm 10 — Autonomous Research / Learning Algorithm

**Directory**: `mirror7_algorithm_10/`

**Problem**: Determine what information to pursue next to most effectively
reduce uncertainty or advance goals.

**Mechanism**: Uncertainty-driven active learning. Maintains a knowledge
state with known facts, open questions, and uncertainty map. Scores
candidate actions by: `info_gain * importance / (cost + 1)`. Uses
epsilon-greedy exploration (epsilon decays over time) to balance
exploitation of known high-value topics with exploration of novel areas.

**Key Feature**: Intrinsic motivation — the system drives its own learning
without external curriculum.

**Failure Modes**: No productive action available (explore randomly),
action fails (retry or skip, update cost estimate), conflicting results
(increase uncertainty, flag for investigation).

---

### Algorithm 11 — Abstract Reasoning Engine

**Directory**: `mirror7_algorithm_11/`

**Problem**: Solve novel problems by discovering structural relations,
transformations, analogies, and invariants rather than memorizing answers.

**Mechanism**: Structure mapping over structured representations (entities,
properties, relations). Discovers correspondences between source and target
structures. Extracts transformation rules from before-after examples.
Applies and composes transformations to predict outputs for new inputs.

**Key Feature**: Analogy by structure — transfers knowledge based on
relational similarity, not surface features.

**Failure Modes**: No consistent mapping (report partial score),
multiple valid transformations (return all with scores), large structures
(bounded search with early stopping).

---

### Algorithm 12 — General Cognitive Workspace

**Directory**: `mirror7_algorithm_12/`

**Problem**: Coordinate perception, memory, reasoning, prediction, planning,
and action through a shared bounded state.

**Mechanism**: Bounded blackboard with priority scheduling. Items (percepts,
hypotheses, goals, predictions, questions) are posted to the workspace.
Registered processors handle items by type. Each tick: expire old items,
recompute priorities, select focus, route to processors, check coherence.

**Key Feature**: The workspace is the ONLY shared state — all inter-module
communication is mediated through it, preventing tight coupling.

**Failure Modes**: Processor error (catch and log, continue), capacity
exceeded (evict lowest-priority items), no processor for item type
(item stays unprocessed), conflicting items (coherence checker reduces
weaker item's confidence).

---

## Dependency Graph (for Integration)

```
[1 Concepts] ──────> [5 Memory] <────── [8 Consolidation]
                         |
[9 Grounding] ─────>    |    <────── [4 World Model]
                         |
                    [3 Causal] ──────> [10 Research]
                         |
                    [11 Reasoning] ──> [2 Programs]
                         |
                    [6 Goals] ─────> [10 Research]
                         |
                    [7 Debugging] ──> (monitors all)
                         |
                    [12 Workspace] ── (coordinates all)
```

All arrows represent natural data flow, NOT compile-time dependencies.
Every algorithm is independently runnable.
