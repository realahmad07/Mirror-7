from __future__ import annotations
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import permutations
from typing import Hashable, Sequence, Tuple

State = Tuple[Hashable, ...]
Action = Hashable
Effect = Tuple[Tuple, ...]


def _state(x: Sequence[Hashable]) -> State:
    if not isinstance(x, (tuple, list)) or not x:
        raise ValueError("state must be a non-empty tuple/list")
    return tuple(x)


def _distance(a: State, b: State) -> float:
    if len(a) != len(b):
        raise ValueError("state/goal dimensionality mismatch")
    total = 0.0
    for x, y in zip(a, b):
        if isinstance(x, (int, float)) and isinstance(y, (int, float)):
            total += abs(x - y)
        else:
            total += 0.0 if x == y else 1.0
    return total


def _effect(before: State, after: State) -> Effect:
    if len(before) != len(after):
        raise ValueError("state dimensionality mismatch")
    out = []
    for x, y in zip(before, after):
        if isinstance(x, (int, float)) and isinstance(y, (int, float)):
            out.append(("d", y - x))
        else:
            out.append(("c", x, y))
    return tuple(out)


def _apply(state: State, effect: Effect) -> State | None:
    if len(state) != len(effect):
        return None
    out = []
    for x, e in zip(state, effect):
        if not e:
            return None
        if e[0] == "d":
            if not isinstance(x, (int, float)):
                return None
            out.append(x + e[1])
        elif e[0] == "c":
            if x != e[1]:
                return None
            out.append(e[2])
        else:
            return None
    return tuple(out)


@dataclass(frozen=True)
class Experiment:
    sequence: Tuple[Action, ...]
    score: float
    predicted_partitions: int
    hypotheses: int
    reason: str


@dataclass(frozen=True)
class RegimeRecord:
    regime_id: int
    support: int
    action_count: int
    signature: Tuple[Tuple[Action, Effect], ...]


class Hypothesis:
    def __init__(self, regime_id: int, min_support: int = 2):
        self.regime_id = regime_id
        self.min_support = min_support
        self._effects: dict[Action, Counter[Effect]] = defaultdict(Counter)
        self.support = 0

    def add_profile(self, profile: dict[Action, Effect]) -> None:
        self.support += 1
        for action, effect in profile.items():
            self._effects[action][effect] += 1

    def action_effect(self, action: Action, *, require_support: bool = True) -> Effect | None:
        counts = self._effects.get(action)
        if not counts:
            return None
        best = counts.most_common()
        if require_support and best[0][1] < self.min_support:
            return None
        if len(best) > 1 and best[0][1] <= best[1][1]:
            return None
        return best[0][0]

    def predict(self, state: State, action: Action, *, require_support: bool = True) -> State | None:
        effect = self.action_effect(action, require_support=require_support)
        return None if effect is None else _apply(state, effect)

    def similarity(self, profile: dict[Action, Effect]) -> tuple[float, int]:
        if not profile:
            return 0.0, 0
        matches = 0
        overlap = 0
        for action, effect in profile.items():
            learned = self.action_effect(action, require_support=False)
            if learned is None:
                continue
            overlap += 1
            if learned == effect:
                matches += 1
        return (matches / overlap if overlap else 0.0), overlap

    def record(self) -> RegimeRecord:
        sig = []
        for action in sorted(self._effects, key=repr):
            effect = self.action_effect(action)
            if effect is not None:
                sig.append((action, effect))
        return RegimeRecord(self.regime_id, self.support, len(sig), tuple(sig))


class UnknownRegimeBank:
    """Discovers regimes from unlabeled experiment profiles."""

    def __init__(self, *, min_support: int = 2, merge_threshold: float = 0.8):
        if min_support < 1 or not (0.0 < merge_threshold <= 1.0):
            raise ValueError("invalid hypothesis-bank bounds")
        self.min_support = min_support
        self.merge_threshold = merge_threshold
        self._hypotheses: list[Hypothesis] = []

    @property
    def hypotheses(self) -> tuple[Hypothesis, ...]:
        return tuple(self._hypotheses)

    def add_experiment(self, transitions: Sequence[tuple[State, Action, State]]) -> int:
        if not transitions:
            raise ValueError("experiment transitions cannot be empty")
        profile = {}
        for before, action, after in transitions:
            profile[action] = _effect(_state(before), _state(after))
        best = None
        best_key = (-1.0, -1)
        for index, hypothesis in enumerate(self._hypotheses):
            score, overlap = hypothesis.similarity(profile)
            key = (score, overlap)
            if key > best_key:
                best_key = key
                best = index
        if best is not None and best_key[0] >= self.merge_threshold and best_key[1] >= 1:
            self._hypotheses[best].add_profile(profile)
            return best
        hypothesis = Hypothesis(len(self._hypotheses), self.min_support)
        hypothesis.add_profile(profile)
        self._hypotheses.append(hypothesis)
        return hypothesis.regime_id

    def identify(self, transitions: Sequence[tuple[State, Action, State]]) -> int | None:
        if not transitions or not self._hypotheses:
            return None
        profile = {action: _effect(_state(before), _state(after)) for before, action, after in transitions}
        ranked = []
        for hypothesis in self._hypotheses:
            score, overlap = hypothesis.similarity(profile)
            ranked.append((score, overlap, hypothesis.regime_id))
        ranked.sort(reverse=True)
        if not ranked:
            return None
        best = ranked[0]
        second = ranked[1] if len(ranked) > 1 else (0.0, 0, -1)
        if (
            self._hypotheses[best[2]].support >= 2
            and best[0] >= self.merge_threshold
            and best[1] >= 1
            and (best[0] > second[0] or best[1] > second[1])
        ):
            return best[2]
        return None

    def records(self) -> tuple[RegimeRecord, ...]:
        return tuple(h.record() for h in self._hypotheses)


class UnknownRegimeAgent:
    def __init__(self, *, max_experiment_depth: int = 3, experiment_budget: int = 12):
        if max_experiment_depth < 1 or experiment_budget < 1:
            raise ValueError("invalid agent bounds")
        self.max_experiment_depth = max_experiment_depth
        self.experiment_budget = experiment_budget
        self.bank = UnknownRegimeBank(min_support=2)
        self.active_regime: int | None = None
        self._seen_sequences: set[tuple[Action, ...]] = set()
        self._experiment_count = 0
        self._goal: State | None = None
        self._last_experiment: tuple[Action, ...] | None = None
        self.replans = 0
        self.invalid_actions = 0
        self.discovery_events = 0

    @property
    def experiment_count(self) -> int:
        return self._experiment_count

    def reset_context(self, goal: Sequence[Hashable]) -> None:
        self._goal = _state(goal)
        self.active_regime = None
        self._seen_sequences.clear()
        self._experiment_count = 0
        self.replans = 0
        self.invalid_actions = 0
        self.discovery_events = 0

    def _simulate(self, hypothesis: Hypothesis, start: State, sequence: Sequence[Action]):
        state = start
        for action in sequence:
            state = hypothesis.predict(state, action, require_support=False)
            if state is None:
                return None
        return state

    def design_experiment(self, state: State, goal: State, legal: Sequence[Action]) -> Experiment | None:
        legal = tuple(dict.fromkeys(legal))
        if not legal or self._experiment_count >= self.experiment_budget:
            return None
        sequences = []
        discovery_depth = min(self.max_experiment_depth, len(legal))
        if discovery_depth < 1:
            return None
        sequences.extend(permutations(legal, discovery_depth))
        candidates = []
        hyps = self.bank.hypotheses
        for sequence in sequences:
            if sequence in self._seen_sequences:
                continue
            predictions = []
            trajectories = []
            for hypothesis in hyps:
                cursor = state
                path = []
                for action in sequence:
                    nxt = hypothesis.predict(cursor, action, require_support=False)
                    if nxt is None:
                        break
                    cursor = nxt
                    path.append(cursor)
                trajectories.append(path)
                predictions.append(cursor if len(path) == len(sequence) else None)

            partitions = 1
            for step in range(max((len(path) for path in trajectories), default=0)):
                states = [path[step] for path in trajectories if len(path) > step]
                if len(states) >= 2:
                    partitions = max(partitions, len({repr(x) for x in states}))
            disagreement = (partitions - 1) if hyps else 0
            novelty = len(set(sequence))
            supported_actions = set()
            for hypothesis in hyps:
                for action in sequence:
                    if hypothesis.action_effect(action, require_support=True) is not None:
                        supported_actions.add(action)
            unknown_action_count = sum(1 for action in set(sequence) if action not in supported_actions)
            weak_action_count = sum(
                1 for action in set(sequence)
                if action not in supported_actions
                and any(h.action_effect(action, require_support=False) is not None for h in hyps)
            )
            goal_gain = 0.0
            for final in predictions:
                if final is not None:
                    goal_gain += max(0.0, _distance(state, goal) - _distance(final, goal))
            sequence_coverage = len(set(sequence)) / max(1, len(legal))
            coverage_bonus = (
                100.0 * unknown_action_count
                + 150.0 * weak_action_count
                + 200.0 * sequence_coverage
                if hyps
                else 120.0 * sequence_coverage
            )
            score = (
                500.0 * disagreement
                + 20.0 * novelty
                + goal_gain
                + coverage_bonus
                - len(sequence)
            )
            candidates.append((score, -len(sequence), repr(sequence), sequence, partitions))
        if not candidates:
            return None
        candidates.sort(reverse=True)
        score, _, _, sequence, partitions = candidates[0]
        self._seen_sequences.add(sequence)
        self._experiment_count += 1
        self._last_experiment = sequence
        return Experiment(sequence, score, partitions, len(hyps), "max-disagreement-novelty")

    def observe_experiment(self, transitions: Sequence[tuple[State, Action, State]]) -> int | None:
        if not transitions:
            raise ValueError("experiment transitions cannot be empty")
        before = len(self.bank.hypotheses)
        regime_id = self.bank.add_experiment(transitions)
        after = len(self.bank.hypotheses)
        if after > before:
            self.discovery_events += 1
        identified = self.bank.identify(transitions)
        self.active_regime = identified
        self.replans += 1
        return self.active_regime

    def plan_to_goal(self, state: State, goal: State, legal: Sequence[Action], max_depth: int = 16) -> list[Action]:
        if self.active_regime is None:
            return []
        if self.active_regime >= len(self.bank.hypotheses):
            return []
        hypothesis = self.bank.hypotheses[self.active_regime]
        q = [(state, [])]
        seen = {state}
        legal = tuple(dict.fromkeys(legal))
        while q:
            cur, path = q.pop(0)
            if len(path) >= max_depth:
                continue
            for action in sorted(legal, key=repr):
                nxt = hypothesis.predict(cur, action, require_support=True)
                if nxt is None or nxt in seen:
                    continue
                p = path + [action]
                if nxt == goal:
                    return p
                seen.add(nxt)
                q.append((nxt, p))
        return []

    def run(self, env, *, start: Sequence[Hashable], goal: Sequence[Hashable], max_steps: int):
        if max_steps < 1:
            raise ValueError("max_steps must be positive")
        start = _state(start)
        goal = _state(goal)
        self.reset_context(goal)
        actions_executed = []
        current = start

        while self.active_regime is None and self._experiment_count < self.experiment_budget:
            env.reset(start)
            current = start
            legal = tuple(env.legal_actions(current))
            experiment = self.design_experiment(current, goal, legal)
            if experiment is None:
                break
            trace = []
            for action in experiment.sequence:
                legal = tuple(env.legal_actions(current))
                if action not in legal:
                    self.invalid_actions += 1
                    raise AssertionError("designed non-legal experiment action")
                nxt = _state(env.step(action))
                trace.append((current, action, nxt))
                current = nxt
            self.observe_experiment(trace)

        env.reset(start)
        current = start
        if self.active_regime is None:
            return {
                "solved": current == goal,
                "steps": 0,
                "invalid_actions": self.invalid_actions,
                "experiments": self._experiment_count,
                "discovered_regimes": len(self.bank.hypotheses),
                "replans": self.replans,
                "final_state": current,
                "actions": tuple(actions_executed),
            }

        for _ in range(max_steps):
            if current == goal:
                break
            legal = tuple(env.legal_actions(current))
            plan = self.plan_to_goal(current, goal, legal)
            if not plan:
                candidates = [a for a in legal if a not in self._seen_sequences]
                if not candidates:
                    candidates = list(legal)
                action = sorted(candidates, key=repr)[0]
                trace = [(current, action, _state(env.step(action)))]
                actions_executed.append(action)
                current = trace[0][2]
                self.observe_experiment(trace)
                continue
            action = plan[0]
            if action not in legal:
                self.invalid_actions += 1
                raise AssertionError("planned non-legal action")
            before = current
            current = _state(env.step(action))
            actions_executed.append(action)
            predicted = self.bank.hypotheses[self.active_regime].predict(before, action, require_support=True)
            if predicted is not None and predicted != current:
                self.replans += 1
                self.active_regime = None
                self.observe_experiment([(before, action, current)])

        return {
            "solved": current == goal,
            "steps": len(actions_executed),
            "invalid_actions": self.invalid_actions,
            "experiments": self._experiment_count,
            "discovered_regimes": len(self.bank.hypotheses),
            "replans": self.replans,
            "final_state": current,
            "actions": tuple(actions_executed),
    }
