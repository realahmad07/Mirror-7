"""Mirror 7 Phase 63: nonstationary world + autonomous experiment design.

Learns multiple regime-specific transition hypotheses, detects repeated model
mismatch as possible dynamics drift, and selects experiments by predicted
outcome disagreement plus goal relevance. Exploration is budgeted and legal
actions are enforced.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Hashable, Sequence, Tuple

State = Tuple[Hashable, ...]
Action = Hashable


def _state(x: Sequence[Hashable]) -> State:
    if not isinstance(x, (tuple, list)) or not x:
        raise ValueError("state must be a non-empty tuple/list")
    return tuple(x)


def _dist(a: State, b: State) -> float:
    if len(a) != len(b):
        raise ValueError("state/goal dimensionality mismatch")
    total = 0.0
    for x, y in zip(a, b):
        if isinstance(x, (int, float)) and isinstance(y, (int, float)):
            total += abs(x - y)
        else:
            total += 0.0 if x == y else 1.0
    return total


@dataclass(frozen=True)
class Experiment:
    action: Action
    score: float
    hypotheses: int
    expected_outcomes: int
    reason: str


class RegimeAwareModel:
    def __init__(self, min_support: int = 2, drift_threshold: int = 2):
        if min_support < 2 or drift_threshold < 1:
            raise ValueError("invalid model bounds")
        self.min_support = min_support
        self.drift_threshold = drift_threshold
        self._evidence = defaultdict(Counter)
        self._regime_evidence = defaultdict(lambda: defaultdict(Counter))
        self._drift = Counter()
        self.stale = False
        self.drift_events = 0

    def update(self, state: State, action: Action, next_state: State) -> bool:
        state, next_state = _state(state), _state(next_state)
        key = (state, action)
        pred = self.predict(state, action)
        mismatch = pred is not None and pred != next_state
        if mismatch:
            self._drift[key] += 1
            if self._drift[key] >= self.drift_threshold:
                self.stale = True
                self.drift_events += 1
        self._evidence[key][next_state] += 1
        return mismatch

    def predict(self, state: State, action: Action) -> State | None:
        counts = self._evidence.get((_state(state), action))
        if not counts:
            return None
        best = counts.most_common()
        if best[0][1] < self.min_support:
            return None
        if len(best) > 1 and best[0][1] <= best[1][1]:
            return None
        return best[0][0]

    def add_regime_profile(
        self,
        regime: Hashable,
        transitions: Sequence[tuple[State, Action, State]],
    ) -> None:
        if not transitions:
            raise ValueError("transitions cannot be empty")
        bucket = self._regime_evidence[regime]
        for s, a, n in transitions:
            s, n = _state(s), _state(n)
            bucket[(s, a)][n] += 1
            self._evidence[(s, a)][n] += 1

    def regime_prediction(
        self, regime: Hashable, state: State, action: Action
    ) -> State | None:
        counts = self._regime_evidence.get(regime, {}).get((_state(state), action))
        if not counts:
            return None
        best = counts.most_common()
        if best[0][1] < self.min_support:
            return None
        if len(best) > 1 and best[0][1] <= best[1][1]:
            return None
        return best[0][0]

    def clear_stale(self) -> None:
        self.stale = False
        self._drift.clear()

    def candidate_regimes(self) -> tuple[Hashable, ...]:
        return tuple(sorted(self._regime_evidence.keys(), key=repr))


class NonstationaryAgent:
    def __init__(
        self,
        *,
        experiment_budget: int = 4,
        drift_threshold: int = 2,
        min_support: int = 2,
    ):
        if experiment_budget < 1:
            raise ValueError("experiment_budget must be positive")
        self.experiment_budget = experiment_budget
        self.model = RegimeAwareModel(
            min_support=min_support, drift_threshold=drift_threshold
        )
        self.active_regime = None
        self._experiment_trials = Counter()
        self._observations = Counter()
        self._known_actions: set[Action] = set()
        self._experiments = 0
        self._replans = 0
        self._drift_events = 0
        self._invalid_actions = 0
        self._current: State | None = None
        self._goal: State | None = None

    def reset(self, state: Sequence[Hashable], goal: Sequence[Hashable]) -> None:
        self._current, self._goal = _state(state), _state(goal)
        if len(self._current) != len(self._goal):
            raise ValueError("state/goal dimensionality mismatch")
        self.active_regime = None
        self._experiment_trials.clear()
        self._observations.clear()
        self._experiments = 0
        self._replans = 0
        self._drift_events = 0
        self._invalid_actions = 0

    @property
    def state(self) -> State:
        if self._current is None:
            raise RuntimeError("agent is not reset")
        return self._current

    @property
    def goal(self) -> State:
        if self._goal is None:
            raise RuntimeError("agent is not reset")
        return self._goal

    def learn_regime(
        self,
        regime: Hashable,
        transitions: Sequence[tuple[State, Action, State]],
    ) -> None:
        if not transitions:
            raise ValueError("transitions cannot be empty")
        self.model.add_regime_profile(regime, transitions)
        for _, a, _ in transitions:
            self._known_actions.add(a)

    def design_experiment(
        self,
        state: Sequence[Hashable],
        legal: Sequence[Action],
        goal: Sequence[Hashable],
    ) -> Experiment | None:
        state, goal = _state(state), _state(goal)
        regimes = self.model.candidate_regimes()
        if len(regimes) < 2 or self._experiments >= self.experiment_budget:
            return None

        current_distance = _dist(state, goal)
        candidates = []
        for action in dict.fromkeys(legal):
            if self._experiment_trials[(state, action)] >= 1:
                continue
            outcomes = []
            for regime in regimes:
                pred = self.model.regime_prediction(regime, state, action)
                if pred is not None:
                    outcomes.append(pred)
            uniq = {repr(x): x for x in outcomes}
            if len(uniq) < 2:
                continue

            goal_gain = 0.0
            for outcome in uniq.values():
                goal_gain += max(
                    0.0, current_distance - _dist(outcome, goal)
                )
            score = (
                len(uniq) * 100.0
                + goal_gain
                - self._experiment_trials[(state, action)]
            )
            candidates.append(
                (score, repr(action), action, len(regimes), len(uniq))
            )

        if not candidates:
            return None
        candidates.sort(reverse=True)
        score, _, action, hypotheses, outcomes = candidates[0]
        return Experiment(
            action, score, hypotheses, outcomes, "max-disagreement"
        )

    def identify_regime(
        self, state: State, action: Action, outcome: State
    ) -> Hashable | None:
        matches = []
        for regime in self.model.candidate_regimes():
            pred = self.model.regime_prediction(regime, state, action)
            if pred is not None and pred == outcome:
                matches.append(regime)
        if len(matches) == 1:
            self.active_regime = matches[0]
            self.model.clear_stale()
            self._replans += 1
            return matches[0]
        return None

    def decide(
        self,
        state: Sequence[Hashable],
        goal: Sequence[Hashable],
        legal: Sequence[Action],
    ) -> tuple[Action, str]:
        state, goal = _state(state), _state(goal)
        legal = tuple(dict.fromkeys(legal))
        if not legal:
            raise ValueError("legal actions cannot be empty")
        self._current, self._goal = state, goal
        if state == goal:
            raise RuntimeError("goal already satisfied")

        exp = self.design_experiment(state, legal, goal)
        if exp is not None and (self.model.stale or self.active_regime is None):
            self._experiments += 1
            self._experiment_trials[(state, exp.action)] += 1
            return exp.action, "experiment"

        candidates = []
        regimes = (
            [self.active_regime]
            if self.active_regime is not None
            else list(self.model.candidate_regimes())
        )
        for action in legal:
            nexts = []
            for regime in regimes:
                pred = self.model.regime_prediction(regime, state, action)
                if pred is not None:
                    nexts.append(pred)
            if len(nexts) == 1:
                candidates.append((_dist(nexts[0], goal), repr(action), action))
            else:
                pred = self.model.predict(state, action)
                if pred is not None:
                    candidates.append((_dist(pred, goal), repr(action), action))

        if candidates:
            candidates.sort()
            return candidates[0][2], "model"

        for action in sorted(legal, key=repr):
            if self._observations[(state, action)] == 0:
                self._observations[(state, action)] += 1
                return action, "probe"

        raise RuntimeError("no supported action remains")

    def observe(
        self,
        action: Action,
        next_state: Sequence[Hashable],
        *,
        legal: Sequence[Action],
    ) -> dict:
        if self._current is None:
            raise RuntimeError("agent is not reset")
        if action not in set(legal):
            self._invalid_actions += 1
            raise ValueError("illegal action")

        before = self._current
        nxt = _state(next_state)
        predicted = self.model.predict(before, action)
        mismatch = self.model.update(before, action, nxt)
        if mismatch:
            self._drift_events += 1
            self._replans += 1

        regime = self.identify_regime(before, action, nxt)
        self._current = nxt
        self._known_actions.add(action)

        return {
            "predicted": predicted,
            "observed": nxt,
            "mismatch": mismatch,
            "regime": regime,
        }

    def run(self, env, *, start: Sequence[Hashable], goal: Sequence[Hashable], max_steps: int):
        if max_steps < 1:
            raise ValueError("max_steps must be positive")
        env.reset(start)
        self.reset(start, goal)
        actions = []
        for _ in range(max_steps):
            if self.state == self.goal:
                break
            legal = env.legal_actions(self.state)
            try:
                action, _ = self.decide(self.state, self.goal, legal)
            except RuntimeError:
                break
            if action not in legal:
                self._invalid_actions += 1
                raise AssertionError("agent selected illegal action")
            nxt = env.step(action)
            self.observe(action, nxt, legal=legal)
            actions.append(action)

        return {
            "solved": self.state == self.goal,
            "steps": len(actions),
            "invalid_actions": self._invalid_actions,
            "experiments": self._experiments,
            "drift_events": self._drift_events,
            "replans": self._replans,
            "final_state": self.state,
            "actions": tuple(actions),
        }
