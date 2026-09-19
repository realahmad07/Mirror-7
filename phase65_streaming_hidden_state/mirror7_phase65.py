from __future__ import annotations
from collections import defaultdict, deque
from dataclasses import dataclass
from math import sqrt
from statistics import NormalDist
from typing import Hashable, Sequence, Tuple

ObsValue = float | int | None
Observation = Tuple[ObsValue, ...]
Action = Hashable


def _obs(x: Sequence[ObsValue]) -> Observation:
    if not isinstance(x, (tuple, list)) or not x:
        raise ValueError("observation must be a non-empty tuple/list")
    out = []
    for v in x:
        if v is not None and not isinstance(v, (int, float)):
            raise ValueError("observation values must be numeric or None")
        out.append(v)
    return tuple(out)


def _visible_pairs(a: Observation, b: Observation):
    if len(a) != len(b):
        raise ValueError("observation dimensionality mismatch")
    for i, (x, y) in enumerate(zip(a, b)):
        if x is not None and y is not None:
            yield i, float(y) - float(x)


@dataclass
class RunningStat:
    n: int = 0
    mean: float = 0.0
    m2: float = 0.0

    def add(self, x: float) -> None:
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        self.m2 += delta * (x - self.mean)

    @property
    def variance(self) -> float:
        return self.m2 / (self.n - 1) if self.n > 1 else 0.0

    @property
    def std(self) -> float:
        return sqrt(max(0.0, self.variance))


@dataclass(frozen=True)
class EffectEstimate:
    action: Action
    lag: int
    dimension: int
    mean: float
    std: float
    samples: int
    confidence: float


class StochasticEffectModel:
    """Online action -> delayed effect statistics over partially observed streams."""

    def __init__(self, *, max_lag: int = 4, min_samples: int = 4):
        if max_lag < 1 or min_samples < 2:
            raise ValueError("invalid model bounds")
        self.max_lag = max_lag
        self.min_samples = min_samples
        self.effects: dict[tuple[Action, int, int], RunningStat] = defaultdict(RunningStat)
        self.baseline: dict[tuple[int, int], RunningStat] = defaultdict(RunningStat)
        self.recent_effects: dict[tuple[Action, int, int], deque[float]] = defaultdict(lambda: deque(maxlen=32))
        self.recent_baseline: dict[tuple[int, int], deque[float]] = defaultdict(lambda: deque(maxlen=32))

    def add_baseline(self, before: Observation, after: Observation) -> None:
        for dim, delta in _visible_pairs(before, after):
            self.baseline[(1, dim)].add(delta)
            self.recent_baseline[(1, dim)].append(delta)

    def add_effect(self, action: Action, before: Observation, after: Observation, lag: int) -> None:
        if lag < 1 or lag > self.max_lag:
            raise ValueError("lag outside model bounds")
        for dim, delta in _visible_pairs(before, after):
            drift = self.baseline[(1, dim)].mean if self.baseline[(1, dim)].n else 0.0
            value = delta - lag * drift
            self.effects[(action, lag, dim)].add(value)
            self.recent_effects[(action, lag, dim)].append(value)

    def estimate(self, action: Action, lag: int, dimension: int) -> EffectEstimate | None:
        stat = self.effects.get((action, lag, dimension))
        if stat is None or stat.n < self.min_samples:
            return None
        z = 1.96
        se = stat.std / sqrt(stat.n) if stat.n > 1 else float("inf")
        return EffectEstimate(
            action, lag, dimension, stat.mean, stat.std, stat.n,
            max(0.0, 1.0 - 2.0 * (1.0 - NormalDist().cdf(z))),
        )

    def best_lag(self, action: Action, dimension: int) -> int | None:
        significant = []
        fallback = []
        for lag in range(1, self.max_lag + 1):
            stat = self.effects.get((action, lag, dimension))
            if stat is None or stat.n < self.min_samples:
                continue
            se = stat.std / sqrt(stat.n) if stat.n > 1 else float("inf")
            z = abs(stat.mean) / (se + 1e-9)
            fallback.append((z, abs(stat.mean), -lag, lag))
            if z >= 3.0 and abs(stat.mean) >= 0.20:
                significant.append(lag)
        if significant:
            return min(significant)
        if fallback:
            return max(fallback)[-1]
        return None

    def recent_model(self, window: int = 24) -> "StochasticEffectModel":
        clone = StochasticEffectModel(max_lag=self.max_lag, min_samples=self.min_samples)
        clone.effects = defaultdict(RunningStat)
        clone.baseline = defaultdict(RunningStat)
        for key, vals in self.recent_effects.items():
            stat = clone.effects[key]
            for value in list(vals)[-window:]:
                stat.add(value)
        for key, vals in self.recent_baseline.items():
            stat = clone.baseline[key]
            for value in list(vals)[-window:]:
                stat.add(value)
        return clone

    def signature(self, actions: Sequence[Action], dims: int) -> dict[tuple[Action, int, int], float]:
        out = {}
        for action in actions:
            for lag in range(1, self.max_lag + 1):
                for dim in range(dims):
                    est = self.estimate(action, lag, dim)
                    if est is not None:
                        out[(action, lag, dim)] = est.mean
        return out

    def predict(self, obs: Observation, action: Action, horizon: int) -> Observation | None:
        out = list(obs)
        found = False
        for dim, x in enumerate(obs):
            if x is None:
                continue
            lag = self.best_lag(action, dim)
            if lag is None or lag > horizon:
                continue
            est = self.estimate(action, lag, dim)
            if est is None:
                continue
            out[dim] = float(x) + est.mean
            found = True
        return tuple(out) if found else None


@dataclass
class PendingAction:
    action: Action
    start: Observation
    age: int = 0


@dataclass(frozen=True)
class RegimeSnapshot:
    regime_id: int
    support: int
    signature: Tuple[Tuple[tuple, float], ...]


class StreamingHypothesis:
    def __init__(self, regime_id: int, *, max_lag: int, min_samples: int, merge_threshold: float = 0.90):
        self.regime_id = regime_id
        self.model = StochasticEffectModel(max_lag=max_lag, min_samples=min_samples)
        self.support = 0
        self.merge_threshold = merge_threshold

    def ingest_model(self, source: StochasticEffectModel) -> None:
        for key, stat in source.effects.items():
            target = self.model.effects[key]
            if stat.n <= 0:
                continue
            if target.n == 0:
                target.n, target.mean, target.m2 = stat.n, stat.mean, stat.m2
            else:
                n1, n2 = target.n, stat.n
                delta = stat.mean - target.mean
                total = n1 + n2
                target.mean = (n1 * target.mean + n2 * stat.mean) / total
                target.m2 = target.m2 + stat.m2 + delta * delta * n1 * n2 / total
                target.n = total
        self.support += 1

    def similarity(self, source: StochasticEffectModel) -> tuple[float, int]:
        matches = 0.0
        overlap = 0
        for key, stat in source.effects.items():
            if stat.n < source.min_samples:
                continue
            target = self.model.effects.get(key)
            if target is None or target.n < self.model.min_samples:
                continue
            scale = max(1.0, abs(target.mean), abs(stat.mean), target.std + stat.std)
            matches += max(0.0, 1.0 - abs(target.mean - stat.mean) / scale)
            overlap += 1
        return (matches / overlap if overlap else 0.0), overlap

    def snapshot(self) -> RegimeSnapshot:
        sig = []
        for key, stat in sorted(self.model.effects.items(), key=lambda x: repr(x[0])):
            if stat.n >= self.model.min_samples:
                sig.append((key, round(stat.mean, 6)))
        return RegimeSnapshot(self.regime_id, self.support, tuple(sig))


class StreamingRegimeBank:
    """Online regime discovery from unlabeled stochastic effect models."""

    def __init__(self, *, max_lag: int = 4, min_samples: int = 4, merge_threshold: float = 0.90):
        if not (0.0 < merge_threshold <= 1.0):
            raise ValueError("merge_threshold must be in (0,1]")
        self.max_lag = max_lag
        self.min_samples = min_samples
        self.merge_threshold = merge_threshold
        self._hyps: list[StreamingHypothesis] = []

    @property
    def hypotheses(self) -> tuple[StreamingHypothesis, ...]:
        return tuple(self._hyps)

    def learn(self, model: StochasticEffectModel) -> int:
        if not any(stat.n >= model.min_samples for stat in model.effects.values()):
            raise ValueError("model has insufficient mature evidence")
        best = None
        best_key = (-1.0, -1)
        for i, h in enumerate(self._hyps):
            score, overlap = h.similarity(model)
            key = (score, overlap)
            if key > best_key:
                best_key = key
                best = i
        if best is not None and best_key[0] >= self.merge_threshold and best_key[1] >= 1:
            self._hyps[best].ingest_model(model)
            return best
        h = StreamingHypothesis(
            len(self._hyps),
            max_lag=self.max_lag,
            min_samples=self.min_samples,
            merge_threshold=self.merge_threshold,
        )
        h.ingest_model(model)
        self._hyps.append(h)
        return h.regime_id

    def identify(self, model: StochasticEffectModel) -> int | None:
        if not self._hyps:
            return None
        ranked = []
        for h in self._hyps:
            score, overlap = h.similarity(model)
            ranked.append((score, overlap, h.support, h.regime_id))
        ranked.sort(reverse=True)
        best = ranked[0]
        if best[0] < self.merge_threshold or best[1] < 1:
            return None
        second = ranked[1] if len(ranked) > 1 else (0.0, 0, 0, -1)
        if best[0] == second[0] and best[1] == second[1]:
            return None
        return best[3]

    def snapshots(self) -> tuple[RegimeSnapshot, ...]:
        return tuple(h.snapshot() for h in self._hyps)


@dataclass(frozen=True)
class ExperimentPlan:
    sequence: Tuple[Action, ...]
    focal_actions: Tuple[Action, ...]
    score: float
    disagreement: float
    reason: str


class Phase65Agent:
    """No-reset streaming learner with delayed/stochastic/partial observation handling."""

    def __init__(self, *, max_lag: int = 4, min_samples: int = 4, experiment_budget: int = 12, mismatch_window: int = 3):
        if min_samples < 2 or max_lag < 1 or experiment_budget < 1 or mismatch_window < 1:
            raise ValueError("invalid agent bounds")
        self.max_lag = max_lag
        self.min_samples = min_samples
        self.experiment_budget = experiment_budget
        self.mismatch_window = mismatch_window
        self.bank = StreamingRegimeBank(max_lag=max_lag, min_samples=min_samples, merge_threshold=0.90)
        self.model = StochasticEffectModel(max_lag=max_lag, min_samples=min_samples)
        self.active_regime: int | None = None
        self._buffer: deque[tuple[Action, Observation]] = deque(maxlen=max_lag + 2)
        self._last_obs: Observation | None = None
        self._pending: list[PendingAction] = []
        self._recent_models: deque[StochasticEffectModel] = deque(maxlen=mismatch_window)
        self._window_size = max(6, mismatch_window * 3)
        self._mismatch_streak = 0
        self._seen_experiments: set[tuple[Action, ...]] = set()
        self.experiment_count = 0
        self.revision_events = 0
        self.invalid_actions = 0
        self._goal: Observation | None = None

    def reset_context(self, goal: Sequence[ObsValue]) -> None:
        self._goal = _obs(goal)
        self._buffer.clear()
        self._mismatch_streak = 0
        self._pending.clear()
        self._recent_models.clear()
        self._seen_experiments.clear()
        self.experiment_count = 0
        self.revision_events = 0
        self.invalid_actions = 0
        self.active_regime = None
        self.model = StochasticEffectModel(max_lag=self.max_lag, min_samples=self.min_samples)

    def observe(self, action: Action, before: Observation, after: Observation) -> None:
        before, after = _obs(before), _obs(after)
        had_pending = bool(self._pending)
        if action == "__noop__":
            if not had_pending:
                self.model.add_baseline(before, after)
        else:
            self._pending.append(PendingAction(action, before, 0))
        for p in list(self._pending):
            p.age += 1
            if p.age <= self.max_lag:
                self.model.add_effect(p.action, p.start, after, p.age)
            if p.age >= self.max_lag:
                self._pending.remove(p)
        self._last_obs = after
        self._buffer.append((action, after))
        self._maybe_revision()

    def _maybe_revision(self) -> None:
        snap = self.model.recent_model(self._window_size)
        mature = False
        for stat in snap.effects.values():
            if stat.n < self.min_samples:
                continue
            se = stat.std / sqrt(stat.n) if stat.n > 1 else float("inf")
            if abs(stat.mean) >= 0.20 and abs(stat.mean) / (se + 1e-9) >= 3.0:
                mature = True
                break
        if not mature:
            return
        self._recent_models.append(snap)
        if self.active_regime is None:
            self.active_regime = self.bank.learn(snap)
            self.revision_events += 1
            return
        h = self.bank.hypotheses[self.active_regime]
        score, overlap = h.similarity(snap)
        if overlap >= 1 and score < h.merge_threshold:
            self._mismatch_streak += 1
        else:
            self._mismatch_streak = 0
        if self._mismatch_streak >= self.mismatch_window:
            self.revision_events += 1
            self.active_regime = self.bank.learn(snap)
            self._mismatch_streak = 0

    def design_experiment(self, legal: Sequence[Action], *, slots: int | None = None) -> ExperimentPlan | None:
        legal = tuple(dict.fromkeys(legal))
        if not legal or self.experiment_count >= self.experiment_budget:
            return None
        noop = "__noop__" if "__noop__" in legal else None
        active = self.bank.hypotheses
        horizon_slots = min(self.max_lag, slots or self.max_lag)
        candidates = []
        for focal in legal:
            if focal == noop:
                continue
            disagreement = 0.0
            if active:
                for i in range(len(active)):
                    for j in range(i + 1, len(active)):
                        h1, h2 = active[i], active[j]
                        for lag in range(1, horizon_slots + 1):
                            for dim in range(8):
                                e1 = h1.model.estimate(focal, lag, dim)
                                e2 = h2.model.estimate(focal, lag, dim)
                                if e1 and e2:
                                    disagreement += abs(e1.mean - e2.mean)
                uncertainty = 0.0
                for h in active:
                    for lag in range(1, horizon_slots + 1):
                        for dim in range(8):
                            e = h.model.estimate(focal, lag, dim)
                            if e:
                                uncertainty += e.std / sqrt(max(1, e.samples))
                score = disagreement * 10.0 + uncertainty + (
                    20.0 if not any(h.model.estimate(focal, 1, d) for h in active for d in range(8)) else 0.0
                )
            else:
                score = 100.0 if not any(
                    k[0] == focal and s.n >= self.min_samples for k, s in self.model.effects.items()
                ) else 1.0
            lengths = range(1, horizon_slots + 1) if noop else range(1, 2)
            for length in lengths:
                seq = [focal]
                if noop:
                    seq.extend([noop] * (length - 1))
                sequence = tuple(seq)
                if sequence in self._seen_experiments:
                    continue
                length_bonus = 2.0 if length > 1 else 0.0
                candidates.append((score + length_bonus, repr(sequence), sequence, focal, disagreement))
        if not candidates:
            return None
        candidates.sort(reverse=True)
        score, _, sequence, focal, disagreement = candidates[0]
        self._seen_experiments.add(sequence)
        self.experiment_count += 1
        return ExperimentPlan(sequence, (focal,), score, disagreement, "disagreement-plus-uncertainty-pulse")

    def predict(self, observation: Sequence[ObsValue], action: Action, horizon: int) -> Observation | None:
        obs = _obs(observation)
        if self.active_regime is not None and self.active_regime < len(self.bank.hypotheses):
            return self.bank.hypotheses[self.active_regime].model.predict(obs, action, horizon)
        return self.model.predict(obs, action, horizon)

    def fail_closed(self) -> dict:
        return {
            "active_regime": self.active_regime,
            "experiments": self.experiment_count,
            "revisions": self.revision_events,
            "hypotheses": len(self.bank.hypotheses),
        }

    @staticmethod
    def _goal_distance(obs: Observation, goal: Observation) -> float:
        if len(obs) != len(goal):
            raise ValueError("observation/goal dimensionality mismatch")
        total = 0.0
        seen = False
        for x, g in zip(obs, goal):
            if g is None:
                continue
            if x is None:
                return float("inf")
            total += abs(float(x) - float(g))
            seen = True
        return total if seen else float("inf")

    @staticmethod
    def _goal_reached(obs: Observation, goal: Observation, tolerance: float = 0.25) -> bool:
        if len(obs) != len(goal):
            raise ValueError("observation/goal dimensionality mismatch")
        required = False
        for x, g in zip(obs, goal):
            if g is None:
                continue
            required = True
            if x is None or abs(float(x) - float(g)) > tolerance:
                return False
        return required

    def run_stream(self, env, *, goal: Sequence[ObsValue] | None = None, max_steps: int = 64):
        """Execute in one continuous stream; never calls env.reset()."""
        if max_steps < 1:
            raise ValueError("max_steps must be positive")
        target = _obs(goal) if goal is not None else self._goal
        self._goal = target
        current = _obs(env.observe())
        actions = []
        experiments_used = 0
        for _ in range(max_steps):
            if target is not None and self._goal_reached(current, target):
                return {
                    "solved": True, "steps": len(actions), "actions": tuple(actions),
                    "experiments": self.experiment_count, "hypotheses": len(self.bank.hypotheses),
                    "active_regime": self.active_regime, "invalid_actions": self.invalid_actions,
                }
            legal = tuple(env.legal_actions())
            if not legal:
                return {
                    "solved": False, "steps": len(actions), "actions": tuple(actions),
                    "experiments": self.experiment_count, "hypotheses": len(self.bank.hypotheses),
                    "active_regime": self.active_regime, "invalid_actions": self.invalid_actions,
                }

            if experiments_used < self.experiment_budget and self.active_regime is None:
                plan = self.design_experiment(legal)
                if plan is not None:
                    experiments_used += 1
                    for action in plan.sequence:
                        legal_now = tuple(env.legal_actions())
                        if action not in legal_now:
                            self.invalid_actions += 1
                            return {
                                "solved": False, "steps": len(actions), "actions": tuple(actions),
                                "experiments": self.experiment_count, "hypotheses": len(self.bank.hypotheses),
                                "active_regime": self.active_regime, "invalid_actions": self.invalid_actions,
                            }
                        before = current
                        current = _obs(env.step(action))
                        self.observe(action, before, current)
                        actions.append(action)
                        if len(actions) >= max_steps:
                            break
                    continue

            chosen = None
            if target is not None:
                ranked = []
                for action in legal:
                    pred = self.predict(current, action, self.max_lag)
                    if pred is None:
                        continue
                    ranked.append((self._goal_distance(pred, target), repr(action), action))
                if ranked:
                    ranked.sort()
                    chosen = ranked[0][-1]
            if chosen is None:
                supported = [a for a in legal if self.predict(current, a, self.max_lag) is not None]
                if not supported:
                    return {
                        "solved": False, "steps": len(actions), "actions": tuple(actions),
                        "experiments": self.experiment_count, "hypotheses": len(self.bank.hypotheses),
                        "active_regime": self.active_regime, "invalid_actions": self.invalid_actions,
                    }
                chosen = sorted(supported, key=repr)[0]
            if chosen not in legal:
                self.invalid_actions += 1
                return {
                    "solved": False, "steps": len(actions), "actions": tuple(actions),
                    "experiments": self.experiment_count, "hypotheses": len(self.bank.hypotheses),
                    "active_regime": self.active_regime, "invalid_actions": self.invalid_actions,
                }
            before = current
            current = _obs(env.step(chosen))
            self.observe(chosen, before, current)
            actions.append(chosen)

        solved = target is not None and self._goal_reached(current, target)
        return {
            "solved": bool(solved), "steps": len(actions), "actions": tuple(actions),
            "experiments": self.experiment_count, "hypotheses": len(self.bank.hypotheses),
            "active_regime": self.active_regime, "invalid_actions": self.invalid_actions,
        }


__all__ = [
    "EffectEstimate",
    "ExperimentPlan",
    "Observation",
    "Phase65Agent",
    "RunningStat",
    "StochasticEffectModel",
    "StreamingRegimeBank",
    "StreamingHypothesis",
]
