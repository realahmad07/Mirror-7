# Mirror 7 — Adaptive Predictive Memory (APM)

A general-purpose sequential prediction engine for the Mirror 7 machine-intelligence project.

## What It Does

APM learns contextual patterns from streams of observations and makes probabilistic predictions about what comes next — with calibrated uncertainty and bounded resource usage.

```
observation → context window → memory lookup → weighted prediction → surprise → learning → memory management
```

## Core Idea

**Prediction is the fundamental computation of intelligence.**

An intelligent system must continuously:
1. **Observe** — receive input from the environment
2. **Predict** — anticipate what will happen next
3. **Compare** — measure surprise (prediction error)
4. **Learn** — update its model to reduce future surprise
5. **Forget** — discard low-utility memories to stay within bounds

APM implements this loop as a variable-order contextual pattern matcher with surprise-driven memory management.

## Why This Matters

| Capability | How APM Addresses It |
|---|---|
| **Learning** | Automatically extracts patterns from sequential data |
| **Prediction** | Probabilistic multi-step forecasting with confidence |
| **Memory** | Bounded pattern store with utility-based eviction |
| **Uncertainty** | Entropy-based confidence; novelty prior for unseen events |
| **Adaptation** | Surprise-driven pattern creation; optional temporal decay |
| **Generalization** | Multi-resolution context matching (short + long contexts) |
| **Efficiency** | O(k) per observation where k = context length; CPU-only |

## Quick Start

```python
from mirror7_algorithm import AdaptivePredictiveMemory

# Create engine
apm = AdaptivePredictiveMemory(max_memory=1000, max_context=8)

# Feed observations
for symbol in "ABCABCABCABC":
    result = apm.observe(symbol)
    print(f"Saw {symbol!r}, surprise={result.surprise:.2f}")

# Predict future
predictions = apm.predict(n_steps=3)
for i, p in enumerate(predictions):
    print(f"  Step {i+1}: {p.best_prediction!r} (conf={p.confidence:.3f})")
```

## Algorithm

### Data Structures

- **Pattern**: `(context_tuple, outcome_counts, total_count, recency, access_count)`
- **Context Window**: Sliding window of the last `max_context` observations
- **Prediction Result**: `(distribution, confidence, entropy, best_prediction, surprise)`

### Prediction (per observation)

1. For each context depth `d = 1, 2, ..., len(context)`:
   - Look up the pattern for the last `d` observations
   - If found, weight its distribution by `base^(d-1) × evidence_factor`
2. Normalise weighted contributions into a probability distribution
3. Reserve `novelty_prior` mass for unseen outcomes
4. Compute entropy and confidence

### Surprise

```
S(x) = -log₂ P(x)
```

Capped at `max_surprise` for stability. Unseen observations receive `novelty_prior` probability.

### Learning

- **Existing patterns**: Increment count for the observed outcome
- **New patterns**: Created when surprise exceeds a depth-dependent threshold
  - Depths 1–2: always created (cheap, foundational)
  - Deeper: require `surprise_ratio ≥ (depth - 2) × 0.1`

### Memory Management

Pattern utility:

```
U = log(1 + count) × exp(-λ × age) × log(1 + accesses)
```

When memory exceeds `max_memory`, the lowest-utility patterns are evicted.

### Non-Stationarity (optional)

Set `count_decay > 0` to apply exponential decay to all pattern counts, enabling adaptation to changing distributions.

## Configuration

| Parameter | Default | Description |
|---|---|---|
| `max_memory` | 10,000 | Maximum patterns in memory |
| `max_context` | 16 | Context window length |
| `context_weight_base` | 2.0 | Weight multiplier per context depth |
| `count_decay` | 0.0 | Temporal decay (0 = stationary) |
| `novelty_prior` | 0.01 | Prior mass for unseen observations |
| `evidence_scale` | 10.0 | Evidence needed for full confidence |
| `max_surprise` | 20.0 | Surprise cap (bits) |

## File Structure

```
mirror7_algorithm/
├── __init__.py                  # Package exports
├── mirror7_algorithm.py         # Core algorithm (~480 lines)
├── test_mirror7_algorithm.py    # 12-scenario test suite
└── README.md                    # This file
```

## Running Tests

```bash
cd mirror7_algorithm
python test_mirror7_algorithm.py
```

Or from the parent directory:

```bash
python -m mirror7_algorithm.test_mirror7_algorithm
```

## Potential Applications

APM is a **general building block**, not a single-task tool:

- **Sequence prediction**: Next-token prediction for any discrete sequence
- **Anomaly detection**: High surprise flags unexpected observations
- **Compression**: Prediction distributions can drive arithmetic coding
- **Classification**: Frame class labels as observations in context
- **Decision-making**: Predict outcomes of candidate actions
- **Novelty detection**: Track surprise trends to detect distribution shifts
- **Multi-agent systems**: Each agent maintains its own APM to model others

## Computational Complexity

| Operation | Time | Space |
|---|---|---|
| `observe()` | O(k) | O(1) amortised |
| `predict(n)` | O(n × k) | O(n) |
| Memory management | O(M log M) when triggered | — |

Where k = context length, M = number of stored patterns.

## Honest Limitations

### What It Can Do
- Learn and predict patterns in sequential discrete data
- Quantify prediction uncertainty
- Adapt to changing distributions
- Operate within strict memory bounds
- Work on any hashable observation type

### What It Cannot Do
- Process continuous/high-dimensional data directly (needs discretisation)
- Learn abstract rules or compositional structure (learns statistical associations)
- Plan or reason about goals (prediction only, not planning)
- Handle complex temporal hierarchies (flat context window)
- Replace a full reasoning engine (it is one component)

### Assumptions
- Observations are discrete and hashable
- Patterns are approximately Markov (recent context is informative)
- The observation stream has learnable regularities
- Memory budget is known in advance

### What Would Need to Be Added
- **Representation learning**: Embedding continuous inputs into discrete tokens
- **Hierarchical memory**: Multiple time-scales, chunking, abstraction
- **Goal-directed reasoning**: Using predictions to plan actions
- **Causal inference**: Distinguishing correlation from causation
- **Compositional structure**: Combining simple patterns into complex ones
- **Multi-modal integration**: Merging predictions from different sources

## Design Philosophy

> Could this same mechanism plausibly be reused on many different problems without rewriting the core algorithm?

**Yes.** APM works on any sequential data with hashable tokens. The same code handles strings, numbers, tuples, sensor readings (after discretisation), game states, or API response codes. The algorithm's parameters control the memory/accuracy tradeoff, not the problem domain.

## License

Part of the Mirror 7 experimental machine-intelligence project.
