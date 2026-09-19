# Mirror 7 Algorithm Bundle — Integration Guide

## Overview

This document describes how the twelve Mirror 7 algorithms relate to each
other, the recommended integration order, and the interfaces for connecting
them.

---

## Dependency Map

The algorithms are designed to be **independently usable** but
**composable when integrated**.  No algorithm requires another to function.
However, some algorithms produce outputs that are natural inputs to others.

```
INDEPENDENT (no algorithm dependencies):
  Algorithm 1  - Open-World Concept Discovery
  Algorithm 2  - Compositional Program Induction
  Algorithm 3  - Causal Experiment Designer
  Algorithm 4  - Adaptive World-Model Builder
  Algorithm 5  - Semantic Memory Graph
  Algorithm 6  - Long-Term Goal Manager
  Algorithm 7  - Self-Debugging Algorithm
  Algorithm 8  - Knowledge Consolidation Engine
  Algorithm 9  - Multimodal Grounding Algorithm
  Algorithm 10 - Autonomous Research / Learning
  Algorithm 11 - Abstract Reasoning Engine
  Algorithm 12 - General Cognitive Workspace
```

All twelve are fully standalone.  The dependencies below are
**integration-time** connections, not compile-time requirements.

### Natural Data Flow Dependencies

```
Alg 1 (Concepts)      --> feeds discovered concepts to --> Alg 5 (Memory Graph)
Alg 9 (Grounding)     --> feeds grounded tokens to     --> Alg 5 (Memory Graph)
Alg 5 (Memory Graph)  --> feeds knowledge to           --> Alg 4 (World Model)
Alg 5 (Memory Graph)  --> feeds knowledge to           --> Alg 3 (Causal Exp.)
Alg 8 (Consolidation) --> feeds durable knowledge to   --> Alg 5 (Memory Graph)
Alg 4 (World Model)   --> feeds predictions to         --> Alg 6 (Goal Manager)
Alg 3 (Causal Exp.)   --> feeds experiments to         --> Alg 10 (Research)
Alg 11 (Reasoning)    --> feeds solutions to           --> Alg 2 (Programs)
Alg 6 (Goal Manager)  --> feeds goals to               --> Alg 10 (Research)
Alg 7 (Debugging)     --> monitors all                 --> Alg 12 (Workspace)
Alg 12 (Workspace)    --> coordinates all algorithms
```

---

## Recommended Integration Order

### Phase 1 — Foundation (Independent Testing)

Test each algorithm in isolation.  Verify all test suites pass.

```bash
# Run all tests
for i in 01 02 03 04 05 06 07 08 09 10 11 12; do
    echo "Testing Algorithm $i"
    python -m mirror7_algorithm_${i}.test_mirror7_algorithm
done
```

### Phase 2 — Memory Layer

Integrate the memory subsystem first, as most other algorithms benefit from
persistent knowledge storage.

```
Step 1: Alg 5 (Semantic Memory Graph)    -- central knowledge store
Step 2: Alg 8 (Knowledge Consolidation)  -- feeds durable knowledge into Alg 5
```

**Integration point**: Knowledge Consolidation's `query_knowledge()` output
maps directly to Semantic Memory Graph's `add_node()` / `add_edge()` inputs.

### Phase 3 — Perception Layer

Add the ability to extract structure from raw observations.

```
Step 3: Alg 1 (Concept Discovery)        -- discovers concepts from observations
Step 4: Alg 9 (Multimodal Grounding)     -- links cross-modal observations
```

**Integration point**: Concept Discovery's discovered concepts become nodes
in the Semantic Memory Graph.  Grounded concepts provide cross-modal edges.

### Phase 4 — Prediction Layer

Add the ability to predict future states.

```
Step 5: Alg 4 (World-Model Builder)      -- predicts transitions
Step 6: Alg 3 (Causal Experiment Designer) -- designs discriminating experiments
```

**Integration point**: World Model's transition rules can be represented as
hypotheses in the Causal Experiment Designer.  Prediction failures trigger
experiment suggestions.

### Phase 5 — Reasoning Layer

Add higher-order reasoning capabilities.

```
Step 7: Alg 11 (Abstract Reasoning)      -- structural/analogical reasoning
Step 8: Alg 2 (Program Induction)        -- procedural learning
```

**Integration point**: Abstract Reasoning discovers transformations;
Program Induction compiles them into reusable procedures stored in the
knowledge graph.

### Phase 6 — Action Layer

Add goal-directed behavior and active learning.

```
Step 9:  Alg 6 (Goal Manager)            -- goal decomposition and tracking
Step 10: Alg 10 (Research Algorithm)     -- active learning and exploration
```

**Integration point**: The Goal Manager decomposes goals into subgoals.
The Research Algorithm selects which subgoal-relevant knowledge to pursue.
Both consult the World Model for feasibility.

### Phase 7 — Meta-Cognition Layer

Add self-monitoring and self-improvement.

```
Step 11: Alg 7 (Self-Debugging)          -- failure detection and repair
```

**Integration point**: All algorithms report failures to the Self-Debugging
system.  The debugger proposes parameter adjustments or fallback rules that
are validated before adoption.

### Phase 8 — Coordination Layer

Finally, connect everything through the central workspace.

```
Step 12: Alg 12 (Cognitive Workspace)    -- coordinates all algorithms
```

**Integration point**: Each algorithm registers as a Workspace Processor.
The workspace routes items (percepts, hypotheses, goals, predictions,
questions) to the appropriate algorithm based on item type.

---

## Interface Patterns

### Pattern 1: Producer-Consumer

```python
# Algorithm A produces structured output
concepts = alg1.discover(observation)

# Algorithm B consumes it
for concept in concepts:
    alg5.add_node(concept.id, 'concept', concept.label,
                  attributes=concept.features,
                  confidence=concept.confidence)
```

### Pattern 2: Query-Response

```python
# Algorithm A queries Algorithm B
relevant_knowledge = alg5.query_by_relation(entity_id, 'causes')

# Use the response
for edge in relevant_knowledge:
    alg4.add_rule_from_knowledge(edge)
```

### Pattern 3: Event-Driven

```python
# Algorithm A detects an event
if world_model.surprise > threshold:
    # Notify Algorithm B
    alg3.add_observation(surprising_event)
    alg10.flag_uncertainty(event.topic)
```

### Pattern 4: Workspace-Mediated

```python
# All communication through the workspace
workspace.post(WorkspaceItem(
    item_type='prediction_failure',
    content={'predicted': X, 'actual': Y},
    source='world_model',
    priority=0.8
))

# Relevant processor handles it
# (Self-Debugging processor picks up prediction_failure items)
```

---

## Serialization Protocol

All algorithms support state serialization for:
- **Checkpointing**: Save/restore state between sessions
- **Debugging**: Inspect internal state
- **Transfer**: Move learned knowledge between instances
- **Testing**: Reproducible experiments

```python
# Save state
state = algorithm.to_dict()
with open('checkpoint.json', 'w') as f:
    json.dump(state, f)

# Algorithms that support from_dict can restore state
# Others can be reconstructed from serialized data
```

---

## Testing Protocol

### Unit Testing
Each algorithm has its own test suite with 7+ scenarios.

### Integration Testing
When connecting algorithms, test:
1. Data format compatibility (output of A matches input of B)
2. Error propagation (failure in A is handled gracefully by B)
3. Resource competition (two algorithms don't exhaust shared memory)
4. Circular dependency prevention (A -> B -> A doesn't loop)

### System Testing
When all algorithms are connected through the Workspace:
1. End-to-end scenario: observation -> concept -> memory -> reasoning -> action
2. Stress test: many simultaneous observations
3. Failure cascade: one algorithm fails, verify system continues
4. Memory pressure: all algorithms competing for bounded memory

---

## Common Pitfalls

1. **Tight coupling**: Don't make Algorithm A directly call Algorithm B's
   internal methods.  Use the public API or workspace-mediated communication.

2. **Unbounded growth**: When connecting algorithms, the output of one can
   trigger unbounded input to another.  Always use rate limiting or
   priority filtering at integration points.

3. **Confidence inflation**: When Algorithm A produces output with confidence
   0.8 and Algorithm B uses it as input with confidence 0.8, the resulting
   chain should have confidence <= 0.64 (not 0.8).  Propagate uncertainty.

4. **Circular updates**: A predicts X, B updates based on X, A reads B's
   update and re-predicts.  Use cycle detection or update counters.

5. **Stale data**: When Algorithm A caches data from Algorithm B, ensure
   cache invalidation when B's data changes.

---

## Performance Expectations

All algorithms are designed for CPU execution with bounded resources.
Approximate performance on a modern CPU:

| Algorithm | Observations/sec | Memory (typical) |
|-----------|----------------:|------------------:|
| 1  Concept Discovery     | 1,000+ | ~1 MB |
| 2  Program Induction     | 100+   | ~500 KB |
| 3  Causal Experiments    | 500+   | ~200 KB |
| 4  World Model           | 2,000+ | ~1 MB |
| 5  Semantic Memory       | 5,000+ | ~2 MB |
| 6  Goal Manager          | 10,000+| ~500 KB |
| 7  Self-Debugging        | 5,000+ | ~1 MB |
| 8  Knowledge Consol.     | 2,000+ | ~1 MB |
| 9  Multimodal Grounding  | 1,000+ | ~1 MB |
| 10 Research Algorithm    | 5,000+ | ~500 KB |
| 11 Abstract Reasoning    | 200+   | ~500 KB |
| 12 Cognitive Workspace   | 10,000+| ~1 MB |

These are order-of-magnitude estimates for typical workloads.

---

## Future Enhancements

### Short-Term (Same Architecture)
- Add `from_dict()` deserialization to all algorithms
- Add inter-algorithm event bus
- Add shared configuration management
- Add telemetry / observability hooks

### Medium-Term (Architecture Extension)
- GPU-accelerated similarity search for Algorithm 1 and 9
- Distributed memory for Algorithm 5
- Parallel hypothesis evaluation for Algorithm 3
- Compiled program execution for Algorithm 2

### Long-Term (Research Direction)
- Differentiable versions of key algorithms for gradient-based optimization
- Formal verification of safety properties
- Multi-agent coordination (multiple Mirror 7 instances)
- Real-world sensory integration
