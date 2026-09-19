"""
Mirror 7 — Adaptive Predictive Memory (APM)
============================================

A general-purpose sequential prediction engine that learns contextual patterns
from observation streams, makes probabilistic predictions with calibrated
uncertainty, and adapts through surprise-driven memory management.

Architecture
------------
The algorithm maintains a memory of contextual patterns at multiple resolution
levels (context depths 1, 2, ..., max_context).  When a new observation arrives:

  1. Current context (recent history) queries memory at all matching depths
  2. Matching patterns contribute weighted predictions (longer = higher weight)
  3. The actual observation is compared to the prediction → surprise signal
  4. Memory is updated: existing patterns reinforced, new patterns created
     selectively based on surprise
  5. Low-utility patterns are pruned to stay within bounded memory

Mathematical Foundation
-----------------------
Prediction combines evidence from contexts of different lengths:

    P(next = x) = Σ_l  w_l · P_l(x)  /  Σ_l w_l

where l indexes context depths and weights:

    w_l = base^(l-1) · (1 - exp(-evidence_l / τ))

Surprise (information content):

    S(x) = -log₂ P(x),    capped at S_max

Pattern utility for memory management:

    U = log(1 + count) · exp(-λ · age) · log(1 + accesses)

Confidence:

    confidence = (1 - H_norm) · (1 - exp(-evidence / τ))

where H_norm is Shannon entropy normalized by max entropy.

Design Goals
------------
- CPU-only, standard-library-only (no numpy/torch required)
- Bounded memory and computation
- Deterministic (no randomness unless explicitly requested)
- Works on any hashable observation type
- Reusable across sequence prediction, anomaly detection, compression,
  classification, and decision-making tasks
"""

from __future__ import annotations

import math
import json
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Hashable, Sequence


# ---------------------------------------------------------------------------
# Type alias: observations can be any hashable Python value
# ---------------------------------------------------------------------------
Observation = Hashable


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

@dataclass
class PredictionResult:
    """Result of predicting the next observation.

    Attributes
    ----------
    distribution : dict
        Mapping from possible observations to probabilities (sums to ≤ 1.0;
        remainder is implicit novelty mass for unseen observations).
    confidence : float
        Overall confidence in [0, 1].  Combines distributional certainty
        with evidence strength.
    entropy : float
        Shannon entropy of the prediction distribution, in bits.
    best_prediction : Observation or None
        Most probable next observation, or None if no prediction is possible.
    best_probability : float
        Probability assigned to ``best_prediction``.
    context_depth : int
        Length of the longest matching context that contributed.
    n_patterns_used : int
        Number of distinct memory patterns that contributed.
    surprise : float or None
        Information content of the *actual* observation once revealed.
        None before the observation is seen.
    """

    distribution: dict[Observation, float]
    confidence: float
    entropy: float
    best_prediction: Observation | None
    best_probability: float
    context_depth: int
    n_patterns_used: int
    surprise: float | None = None

    @property
    def is_uncertain(self) -> bool:
        """True when confidence is below 0.5."""
        return self.confidence < 0.5

    def __repr__(self) -> str:
        pred = self.best_prediction
        return (
            f"PredictionResult(best={pred!r}, prob={self.best_probability:.3f}, "
            f"conf={self.confidence:.3f}, surprise={self.surprise})"
        )


@dataclass
class Pattern:
    """A memory pattern linking a context tuple to outcome statistics.

    Stores a distribution over observed outcomes that followed this context,
    plus metadata for memory management (recency, access frequency).
    """

    context: tuple[Observation, ...]
    outcome_counts: dict[Observation, float] = field(default_factory=dict)
    total_count: float = 0.0
    last_accessed: int = 0
    creation_step: int = 0
    access_count: int = 0

    @property
    def depth(self) -> int:
        """Context length."""
        return len(self.context)

    def distribution(self) -> dict[Observation, float]:
        """Return normalised probability distribution over outcomes."""
        if self.total_count <= 0:
            return {}
        return {
            k: v / self.total_count
            for k, v in self.outcome_counts.items()
            if v > 0
        }

    def entropy(self) -> float:
        """Shannon entropy of the outcome distribution, in bits."""
        dist = self.distribution()
        if not dist:
            return 0.0
        return -sum(p * math.log2(p) for p in dist.values() if p > 0)

    def utility(self, current_step: int, decay_rate: float = 0.001) -> float:
        """Compute utility score for memory management.

        Combines evidence strength, recency, and access frequency.
        Low utility → candidate for eviction.
        """
        age = max(0, current_step - self.last_accessed)
        recency = math.exp(-decay_rate * age)
        evidence = math.log1p(self.total_count)
        access = math.log1p(self.access_count)
        return evidence * recency * max(access, 0.1)


@dataclass
class MemoryStats:
    """Summary statistics for the memory store."""

    n_patterns: int
    max_patterns: int
    memory_usage_ratio: float
    total_observations: int
    unique_observations: int
    mean_surprise: float
    mean_confidence: float
    patterns_created: int
    patterns_evicted: int


# ---------------------------------------------------------------------------
# Adaptive Predictive Memory — core algorithm
# ---------------------------------------------------------------------------

class AdaptivePredictiveMemory:
    """Adaptive Predictive Memory — a general-purpose prediction engine.

    Learns contextual patterns from sequential observations and makes
    probabilistic predictions with calibrated uncertainty.

    Parameters
    ----------
    max_memory : int
        Maximum number of patterns to store.  When exceeded, lowest-utility
        patterns are evicted.  Default 10 000.
    max_context : int
        Maximum context window length.  Longer contexts detect longer-range
        dependencies but use more memory.  Default 16.
    context_weight_base : float
        Exponential base for weighting deeper context matches.  Higher values
        make the algorithm rely more on specific (long) contexts when
        available.  Must be ≥ 1.0.  Default 2.0.
    count_decay : float
        Per-step multiplicative decay applied to all pattern counts.
        0.0 = stationary (no forgetting); values near 0 (e.g. 0.001) give
        exponential recency weighting for non-stationary streams.
        Must be in [0, 1).  Default 0.0.
    novelty_prior : float
        Probability mass reserved for unseen observations.  Prevents
        infinite surprise on novel inputs.  Must be in (0, 1).  Default 0.01.
    evidence_scale : float
        Number of observations needed for confidence to reach ~63% of its
        distributional maximum (1 − 1/e scaling).  Default 10.0.
    max_surprise : float
        Upper bound on surprise values (bits), for numerical stability.
        Default 20.0.
    """

    # ------------------------------------------------------------------ init
    def __init__(
        self,
        max_memory: int = 10_000,
        max_context: int = 16,
        context_weight_base: float = 2.0,
        count_decay: float = 0.0,
        novelty_prior: float = 0.01,
        evidence_scale: float = 10.0,
        max_surprise: float = 20.0,
    ) -> None:
        # --- Validate ---
        if max_memory < 1:
            raise ValueError("max_memory must be >= 1")
        if max_context < 1:
            raise ValueError("max_context must be >= 1")
        if context_weight_base < 1.0:
            raise ValueError("context_weight_base must be >= 1.0")
        if not (0.0 <= count_decay < 1.0):
            raise ValueError("count_decay must be in [0, 1)")
        if not (0.0 < novelty_prior < 1.0):
            raise ValueError("novelty_prior must be in (0, 1)")
        if evidence_scale <= 0:
            raise ValueError("evidence_scale must be > 0")
        if max_surprise <= 0:
            raise ValueError("max_surprise must be > 0")

        # --- Configuration ---
        self._max_memory: int = max_memory
        self._max_context: int = max_context
        self._context_weight_base: float = context_weight_base
        self._count_decay: float = count_decay
        self._novelty_prior: float = novelty_prior
        self._evidence_scale: float = evidence_scale
        self._max_surprise: float = max_surprise

        # --- State ---
        self._patterns: dict[tuple[Observation, ...], Pattern] = {}
        self._context: deque[Observation] = deque(maxlen=max_context)
        self._step: int = 0
        self._vocabulary: set[Observation] = set()

        # --- Tracking ---
        self._surprise_sum: float = 0.0
        self._confidence_sum: float = 0.0
        self._recent_surprises: deque[float] = deque(maxlen=200)
        self._recent_confidences: deque[float] = deque(maxlen=200)
        self._patterns_created: int = 0
        self._patterns_evicted: int = 0

    # ============================================================ PUBLIC API

    def observe(self, observation: Observation) -> PredictionResult:
        """Process one observation and return the prediction result.

        The returned ``PredictionResult`` contains what the system predicted
        *before* seeing this observation, plus the ``surprise`` value
        computed *after* seeing it.

        Parameters
        ----------
        observation : Hashable
            The observed value.

        Returns
        -------
        PredictionResult
        """
        # 1. Predict (before seeing observation)
        prediction = self._predict_from_context()

        # 2. Surprise
        surprise = self._compute_surprise(prediction.distribution, observation)
        prediction.surprise = surprise

        # 3. Learn
        self._learn(observation, surprise)

        # 4. Update context window
        self._context.append(observation)
        self._vocabulary.add(observation)
        self._step += 1

        # 5. Count decay (non-stationary mode)
        if self._count_decay > 0:
            self._apply_decay()

        # 6. Memory bounds
        self._enforce_memory_limit()

        # 7. Track running statistics
        self._surprise_sum += surprise
        self._confidence_sum += prediction.confidence
        self._recent_surprises.append(surprise)
        self._recent_confidences.append(prediction.confidence)

        return prediction

    def observe_sequence(
        self, sequence: Sequence[Observation]
    ) -> list[PredictionResult]:
        """Feed a sequence of observations.

        Equivalent to calling :meth:`observe` on each element in order.
        """
        return [self.observe(obs) for obs in sequence]

    def predict(self, n_steps: int = 1) -> list[PredictionResult]:
        """Predict the next *n_steps* observations without updating state.

        Uses autoregressive prediction: each step's best prediction becomes
        the hypothetical context for the next step.  Confidence naturally
        decays with prediction horizon.

        Parameters
        ----------
        n_steps : int
            Number of future steps to predict.

        Returns
        -------
        list[PredictionResult]
        """
        if n_steps < 1:
            return []

        saved_context = deque(self._context, maxlen=self._max_context)
        results: list[PredictionResult] = []

        for i in range(n_steps):
            pred = self._predict_from_context()
            # Confidence decays with forecast horizon
            pred.confidence *= 1.0 / (1.0 + 0.5 * i)
            results.append(pred)

            if pred.best_prediction is None:
                break
            self._context.append(pred.best_prediction)

        self._context = saved_context
        return results

    def reset_context(self) -> None:
        """Clear the context window, keeping memory intact.

        Call between unrelated sequences or episodes.
        """
        self._context.clear()

    def get_stats(self) -> MemoryStats:
        """Return summary statistics about the memory store."""
        n = self._step or 1
        return MemoryStats(
            n_patterns=len(self._patterns),
            max_patterns=self._max_memory,
            memory_usage_ratio=len(self._patterns) / self._max_memory,
            total_observations=self._step,
            unique_observations=len(self._vocabulary),
            mean_surprise=self._surprise_sum / n,
            mean_confidence=self._confidence_sum / n,
            patterns_created=self._patterns_created,
            patterns_evicted=self._patterns_evicted,
        )

    def get_recent_surprise(self, window: int = 50) -> float:
        """Mean surprise over the last *window* observations."""
        if not self._recent_surprises:
            return self._max_surprise
        recent = list(self._recent_surprises)[-window:]
        return sum(recent) / len(recent)

    def get_recent_confidence(self, window: int = 50) -> float:
        """Mean confidence over the last *window* observations."""
        if not self._recent_confidences:
            return 0.0
        recent = list(self._recent_confidences)[-window:]
        return sum(recent) / len(recent)

    def get_pattern_count(self) -> int:
        """Number of patterns currently in memory."""
        return len(self._patterns)

    def inspect_patterns(
        self, *, min_count: float = 0.0, max_results: int = 50
    ) -> list[dict[str, Any]]:
        """Return pattern summaries for debugging/analysis.

        Parameters
        ----------
        min_count : float
            Minimum ``total_count`` to include.
        max_results : int
            Maximum number of patterns to return.

        Returns
        -------
        list[dict]
            Sorted by utility, highest first.
        """
        entries: list[dict[str, Any]] = []
        for pat in self._patterns.values():
            if pat.total_count >= min_count:
                entries.append(
                    {
                        "context": pat.context,
                        "distribution": pat.distribution(),
                        "total_count": pat.total_count,
                        "depth": pat.depth,
                        "utility": pat.utility(self._step),
                        "access_count": pat.access_count,
                        "entropy": pat.entropy(),
                    }
                )
        entries.sort(key=lambda e: e["utility"], reverse=True)
        return entries[:max_results]

    def to_dict(self) -> dict[str, Any]:
        """Serialise state to a JSON-compatible dictionary.

        All observation values are converted to strings for JSON safety.
        """
        patterns_out: list[dict[str, Any]] = []
        for pat in self._patterns.values():
            patterns_out.append(
                {
                    "context": [str(c) for c in pat.context],
                    "outcome_counts": {
                        str(k): v for k, v in pat.outcome_counts.items()
                    },
                    "total_count": pat.total_count,
                    "last_accessed": pat.last_accessed,
                    "creation_step": pat.creation_step,
                    "access_count": pat.access_count,
                }
            )
        return {
            "config": {
                "max_memory": self._max_memory,
                "max_context": self._max_context,
                "context_weight_base": self._context_weight_base,
                "count_decay": self._count_decay,
                "novelty_prior": self._novelty_prior,
                "evidence_scale": self._evidence_scale,
                "max_surprise": self._max_surprise,
            },
            "context": [str(c) for c in self._context],
            "step": self._step,
            "vocabulary": sorted(str(v) for v in self._vocabulary),
            "patterns": patterns_out,
            "counters": {
                "patterns_created": self._patterns_created,
                "patterns_evicted": self._patterns_evicted,
            },
        }

    def save(self, path: str) -> None:
        """Write state to a JSON file."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    # ======================================================= INTERNAL METHODS

    def _predict_from_context(self) -> PredictionResult:
        """Generate a prediction from the current context buffer."""
        if not self._context:
            return PredictionResult(
                distribution={},
                confidence=0.0,
                entropy=0.0,
                best_prediction=None,
                best_probability=0.0,
                context_depth=0,
                n_patterns_used=0,
            )

        ctx_list = list(self._context)
        ctx_len = len(ctx_list)

        # Collect weighted predictions from all matching context depths
        weighted_counts: dict[Observation, float] = {}
        total_weight: float = 0.0
        n_used: int = 0
        max_depth: int = 0
        total_evidence: float = 0.0

        for depth in range(1, ctx_len + 1):
            context_key = tuple(ctx_list[-depth:])
            pattern = self._patterns.get(context_key)
            if pattern is None:
                continue

            # Weight: exponential in depth, scaled by evidence quality
            weight = self._context_weight_base ** (depth - 1)
            evidence_factor = 1.0 - math.exp(
                -pattern.total_count / self._evidence_scale
            )
            weight *= max(evidence_factor, 0.01)  # floor to avoid zero-weight

            dist = pattern.distribution()
            for obs, prob in dist.items():
                weighted_counts[obs] = (
                    weighted_counts.get(obs, 0.0) + weight * prob
                )

            total_weight += weight
            total_evidence += pattern.total_count
            n_used += 1
            max_depth = max(max_depth, depth)

        # --- No matching patterns ---
        if total_weight == 0 or not weighted_counts:
            return PredictionResult(
                distribution={},
                confidence=0.0,
                entropy=0.0,
                best_prediction=None,
                best_probability=0.0,
                context_depth=0,
                n_patterns_used=0,
            )

        # --- Normalise into probability distribution ---
        known_mass = 1.0 - self._novelty_prior
        distribution: dict[Observation, float] = {
            obs: known_mass * wc / total_weight
            for obs, wc in weighted_counts.items()
        }

        # --- Entropy ---
        entropy = _shannon_entropy(distribution)
        if self._novelty_prior > 0:
            entropy -= self._novelty_prior * math.log2(self._novelty_prior)

        # --- Best prediction ---
        best_obs = max(distribution, key=lambda k: distribution[k])
        best_prob = distribution[best_obs]

        # --- Confidence ---
        n_outcomes = len(distribution)
        max_ent = math.log2(n_outcomes) if n_outcomes > 1 else 1.0
        dist_entropy = _shannon_entropy(distribution)
        norm_entropy = min(1.0, dist_entropy / max_ent) if max_ent > 0 else 0.0
        certainty = 1.0 - norm_entropy
        evidence_conf = 1.0 - math.exp(-total_evidence / self._evidence_scale)
        confidence = certainty * evidence_conf

        return PredictionResult(
            distribution=distribution,
            confidence=min(confidence, 1.0),
            entropy=entropy,
            best_prediction=best_obs,
            best_probability=best_prob,
            context_depth=max_depth,
            n_patterns_used=n_used,
        )

    def _compute_surprise(
        self,
        distribution: dict[Observation, float],
        actual: Observation,
    ) -> float:
        """Information content of *actual* given the prediction distribution."""
        if not distribution:
            return self._max_surprise
        prob = distribution.get(actual, self._novelty_prior)
        if prob <= 0:
            return self._max_surprise
        return min(-math.log2(prob), self._max_surprise)

    def _learn(self, observation: Observation, surprise: float) -> None:
        """Update memory patterns given a new observation and its surprise."""
        if not self._context:
            return  # no context → nothing to condition on yet

        ctx_list = list(self._context)
        ctx_len = len(ctx_list)
        surprise_ratio = surprise / self._max_surprise  # ∈ [0, 1]

        for depth in range(1, ctx_len + 1):
            context_key = tuple(ctx_list[-depth:])
            pattern = self._patterns.get(context_key)

            if pattern is not None:
                # --- Reinforce existing pattern ---
                pattern.outcome_counts[observation] = (
                    pattern.outcome_counts.get(observation, 0.0) + 1.0
                )
                pattern.total_count += 1.0
                pattern.last_accessed = self._step
                pattern.access_count += 1
            else:
                # --- Decide whether to create a new pattern ---
                # Depth 1–2: always create (cheap, foundational)
                # Deeper: require proportionally more surprise
                if depth <= 2:
                    create = True
                else:
                    threshold = (depth - 2) * 0.1
                    create = surprise_ratio >= threshold

                if create:
                    self._patterns[context_key] = Pattern(
                        context=context_key,
                        outcome_counts={observation: 1.0},
                        total_count=1.0,
                        last_accessed=self._step,
                        creation_step=self._step,
                        access_count=1,
                    )
                    self._patterns_created += 1

    def _apply_decay(self) -> None:
        """Exponentially decay all pattern counts (non-stationary mode)."""
        factor = 1.0 - self._count_decay
        to_remove: list[tuple[Observation, ...]] = []

        for key, pattern in self._patterns.items():
            pattern.total_count *= factor
            for obs in list(pattern.outcome_counts):
                pattern.outcome_counts[obs] *= factor
                if pattern.outcome_counts[obs] < 1e-6:
                    del pattern.outcome_counts[obs]
            if pattern.total_count < 0.01 or not pattern.outcome_counts:
                to_remove.append(key)

        for key in to_remove:
            del self._patterns[key]
            self._patterns_evicted += 1

    def _enforce_memory_limit(self) -> None:
        """Evict lowest-utility patterns if memory is over budget."""
        n = len(self._patterns)
        if n <= self._max_memory:
            return

        # Sort by utility ascending (evict lowest), but protect depth-1
        # patterns since they provide foundational predictions.
        candidates = [
            (key, pat.utility(self._step))
            for key, pat in self._patterns.items()
            if pat.depth > 1
        ]
        # Only include depth-1 patterns as candidates if there aren't enough
        # deeper patterns to evict
        n_to_remove = n - self._max_memory
        if len(candidates) < n_to_remove:
            candidates.extend(
                (key, pat.utility(self._step))
                for key, pat in self._patterns.items()
                if pat.depth == 1
            )

        candidates.sort(key=lambda x: x[1])

        for i in range(min(n_to_remove, len(candidates))):
            key = candidates[i][0]
            if key in self._patterns:
                del self._patterns[key]
                self._patterns_evicted += 1


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def _shannon_entropy(dist: dict[Any, float]) -> float:
    """Shannon entropy in bits of a probability distribution."""
    return -sum(p * math.log2(p) for p in dist.values() if p > 0)
