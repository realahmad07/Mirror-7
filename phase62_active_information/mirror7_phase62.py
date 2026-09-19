"""Mirror 7 Phase 62: partial observability and active information seeking.

The agent operates on partial observations, infers which legal actions are
information-producing from their observed effect, prioritizes information
about goal-relevant hidden state, and only then plans over a learned full-state
transition model. Unknown actions are explored once per state and stale or
contradictory observations invalidate the relevant model evidence.
"""
from __future__ import annotations

from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from typing import Hashable, Sequence, Tuple

State = Tuple[Hashable, ...]
Action = Hashable
Observation = Tuple[Hashable | None, ...]


@dataclass(frozen=True)
class Decision:
    action: Action
    reason: str
    information_gain: int
    predicted_state: State | None


@dataclass(frozen=True)
class EpisodeResult:
    solved: bool
    steps: int
    invalid_actions: int
    information_steps: int
    replans: int
    contradictions: int
    final_observation: Observation
    actions: Tuple[Action, ...]


def _obs(x: Sequence[Hashable | None]) -> Observation:
    if not isinstance(x, (tuple, list)) or not x:
        raise ValueError("observation must be a non-empty tuple/list")
    return tuple(x)


def _state(x: Sequence[Hashable]) -> State:
    out = _obs(x)
    if any(v is None for v in out):
        raise ValueError("full state cannot contain None")
    return out  # type: ignore[return-value]


def _visible_count(obs: Observation) -> int:
    return sum(v is not None for v in obs)


def _satisfies(obs: Observation, goal: State) -> bool:
    return len(obs) == len(goal) and all(v is not None and v == g for v, g in zip(obs, goal))


def _distance(state: State, goal: State) -> float:
    if len(state) != len(goal):
        raise ValueError("state/goal dimensionality mismatch")
    total = 0.0
    for x, g in zip(state, goal):
        if isinstance(x, (int, float)) and isinstance(g, (int, float)):
            total += abs(x - g)
        else:
            total += 0.0 if x == g else 1.0
    return total


class PartialWorldModel:
    """Exact + factorized numeric-delta transition evidence over full states."""

    def __init__(self, min_support: int = 2):
        if min_support < 2:
            raise ValueError("min_support must be >= 2")
        self.min_support = min_support
        self._exact = defaultdict(Counter)
        self._feature = defaultdict(lambda: defaultdict(Counter))
        self._delta = defaultdict(Counter)
        self._uncertain = set()

    def update(self, state: State, action: Action, nxt: State) -> None:
        if len(state) != len(nxt):
            raise ValueError("state dimensionality mismatch")
        self._exact[(state, action)][nxt] += 1
        for i, (x, y) in enumerate(zip(state, nxt)):
            self._feature[(action, i)][x][y] += 1
            if isinstance(x, (int, float)) and isinstance(y, (int, float)):
                self._delta[(action, i)][y - x] += 1
        self._recompute_uncertainty()

    def _recompute_uncertainty(self) -> None:
        self._uncertain.clear()
        for key, by_old in self._feature.items():
            for old, counts in by_old.items():
                best = counts.most_common()
                if len(best) > 1 and best[0][1] <= best[1][1]:
                    self._uncertain.add((key, old))

    def predict(self, state: State, action: Action) -> State | None:
        exact = self._exact.get((state, action))
        if exact:
            best = exact.most_common()
            if best[0][1] >= self.min_support and (len(best) == 1 or best[0][1] > best[1][1]):
                return best[0][0]
        out = []
        for i, x in enumerate(state):
            key = (action, i)
            counts = self._feature.get(key, {}).get(x)
            if counts and (key, x) not in self._uncertain:
                best = counts.most_common()
                if best[0][1] >= self.min_support and (len(best) == 1 or best[0][1] > best[1][1]):
                    out.append(best[0][0])
                    continue
            deltas = self._delta.get(key)
            if deltas:
                best = deltas.most_common()
                if best[0][1] >= self.min_support and (len(best) == 1 or best[0][1] > best[1][1]):
                    out.append(x + best[0][0])
                    continue
            return None
        return tuple(out)  # type: ignore[return-value]


class PartialObservationAgent:
    def __init__(self, *, max_plan_depth: int = 8, max_exploration: int = 64):
        if max_plan_depth < 1 or max_exploration < 1:
            raise ValueError("invalid agent bounds")
        self.max_plan_depth = max_plan_depth
        self.max_exploration = max_exploration
        self.model = PartialWorldModel(min_support=2)
        self.current: Observation | None = None
        self.goal: State | None = None
        self._tried = Counter()
        self._global_trials = Counter()
        self._info_gain = Counter()
        self._info_hits = defaultdict(Counter)
        self._info_actions: set[Action] = set()
        self._bad_info_actions: set[Action] = set()
        self._replans = 0
        self._contradictions = 0
        self._information_steps = 0
        self._invalid_actions = 0
        self._actions: list[Action] = []

    def reset(self, observation: Sequence[Hashable | None], goal: Sequence[Hashable]) -> None:
        self.current = _obs(observation)
        self.goal = _state(goal)
        if len(self.current) != len(self.goal):
            raise ValueError("observation/goal dimensionality mismatch")
        self._tried.clear()
        self._replans = 0
        self._contradictions = 0
        self._information_steps = 0
        self._invalid_actions = 0
        self._actions = []

    def _choose_information_action(
        self, legal: Sequence[Action], obs: Observation, goal: State
    ) -> Decision | None:
        hidden = {i for i, v in enumerate(obs) if v is None}
        if not hidden:
            return None
        candidates = []
        for action in legal:
            if action in self._bad_info_actions or action not in self._info_actions:
                continue
            gains = self._info_hits[action]
            expected_total = self._info_gain[action] / max(1, sum(gains.values()))
            relevant = sum(1 for i in hidden if self._info_hits[action][i] > 0 and goal[i] is not None)
            score = relevant * 100.0 + expected_total * 10.0 - self._tried[(obs, action)]
            candidates.append((score, action))
        if not candidates:
            return None
        candidates.sort(key=lambda item: (-item[0], repr(item[1])))
        _, action = candidates[0]
        return Decision(action, "active-information", 0, None)

    def _plan(self, obs: Observation, goal: State, legal: Sequence[Action]) -> list[Action]:
        if any(v is None for v in obs):
            return []
        start = _state(obs)
        if start == goal:
            return []
        q = deque([(start, [])])
        seen = {start}
        legal = tuple(dict.fromkeys(legal))
        while q:
            state, path = q.popleft()
            if len(path) >= self.max_plan_depth:
                continue
            for action in sorted(legal, key=repr):
                if action in self._info_actions:
                    continue
                nxt = self.model.predict(state, action)
                if nxt is None or nxt in seen:
                    continue
                p = path + [action]
                if nxt == goal:
                    return p
                seen.add(nxt)
                q.append((nxt, p))
        return []

    def decide(
        self,
        observation: Sequence[Hashable | None],
        goal: Sequence[Hashable],
        legal_actions: Sequence[Action],
    ) -> Decision:
        obs = _obs(observation)
        g = _state(goal)
        legal = tuple(dict.fromkeys(legal_actions))
        if not legal:
            raise ValueError("legal_actions cannot be empty")
        if len(obs) != len(g):
            raise ValueError("observation/goal dimensionality mismatch")
        self.current, self.goal = obs, g
        if _satisfies(obs, g):
            raise RuntimeError("goal already satisfied")

        info = self._choose_information_action(legal, obs, g)
        if info is not None:
            return info

        if any(v is None for v in obs):
            unprofiled = [
                a for a in legal
                if self._global_trials[a] == 0 and a not in self._bad_info_actions
            ]
            if unprofiled:
                return Decision(sorted(unprofiled, key=repr)[0], "profile-action", 0, None)

        plan = self._plan(obs, g, legal)
        if plan:
            nxt = self.model.predict(_state(obs), plan[0])
            return Decision(plan[0], "model-plan", 0, nxt)

        unknown = [a for a in legal if self._tried[(obs, a)] == 0]
        if unknown:
            return Decision(sorted(unknown, key=repr)[0], "bounded-exploration", 0, None)

        if all(v is not None for v in obs):
            s = _state(obs)
            current_distance = _distance(s, g)
            candidates = []
            for action in legal:
                if action in self._info_actions:
                    continue
                nxt = self.model.predict(s, action)
                if nxt is not None:
                    candidates.append((_distance(nxt, g), repr(action), action, nxt))
            if candidates:
                candidates.sort()
                d, _, action, nxt = candidates[0]
                if d <= current_distance:
                    return Decision(action, "model-greedy", 0, nxt)

        raise RuntimeError("no safe information or model-supported action remains")

    def observe(
        self,
        action: Action,
        next_observation: Sequence[Hashable | None],
        *,
        legal_actions: Sequence[Action] | None = None,
    ) -> dict:
        current = self.current
        if current is None:
            raise RuntimeError("agent is not reset")
        nxt = _obs(next_observation)
        if len(nxt) != len(current):
            raise ValueError("observation dimensionality mismatch")
        if legal_actions is not None and action not in set(legal_actions):
            self._invalid_actions += 1
            raise ValueError("attempted action was not legal")

        self._tried[(current, action)] += 1
        self._global_trials[action] += 1
        before_visible = _visible_count(current)
        after_visible = _visible_count(nxt)
        gain = max(0, after_visible - before_visible)
        revealed = {i for i, (a, b) in enumerate(zip(current, nxt)) if a is None and b is not None}
        known_changed = any(a is not None and b is not None and a != b for a, b in zip(current, nxt))

        if gain > 0 and not known_changed:
            self._info_actions.add(action)
            self._info_gain[action] += gain
            for i in revealed:
                self._info_hits[action][i] += 1
            self._information_steps += 1

        predicted = None
        if all(v is not None for v in current) and all(v is not None for v in nxt):
            predicted = self.model.predict(_state(current), action)
            if predicted is not None and predicted != _state(nxt):
                self._contradictions += 1
                self._replans += 1
            self.model.update(_state(current), action, _state(nxt))
        elif action in self._info_actions and gain == 0 and not known_changed:
            self._bad_info_actions.add(action)
            self._info_actions.discard(action)

        self.current = nxt
        self._actions.append(action)
        return {
            "information_gain": gain,
            "revealed_indices": tuple(sorted(revealed)),
            "contradiction": predicted is not None and predicted != _state(nxt) if predicted is not None else False,
        }

    def run(self, env, *, goal: Sequence[Hashable], max_steps: int) -> EpisodeResult:
        if max_steps < 1:
            raise ValueError("max_steps must be positive")
        observation = _obs(env.reset())
        self.reset(observation, goal)
        actions = []
        for _ in range(max_steps):
            if _satisfies(self.current, self.goal):
                return EpisodeResult(
                    True, len(actions), self._invalid_actions,
                    self._information_steps, self._replans, self._contradictions,
                    self.current, tuple(actions)
                )
            legal = env.legal_actions(self.current)
            try:
                decision = self.decide(self.current, self.goal, legal)
            except RuntimeError:
                return EpisodeResult(
                    False, len(actions), self._invalid_actions,
                    self._information_steps, self._replans, self._contradictions,
                    self.current, tuple(actions)
                )
            if decision.action not in legal:
                self._invalid_actions += 1
                raise AssertionError("agent selected non-legal action")
            nxt = env.step(decision.action)
            self.observe(decision.action, nxt, legal_actions=legal)
            actions.append(decision.action)
        return EpisodeResult(
            _satisfies(self.current, self.goal), len(actions), self._invalid_actions,
            self._information_steps, self._replans, self._contradictions,
            self.current, tuple(actions)
        )

    def revealed_state(self) -> State:
        if self.current is None or any(v is None for v in self.current):
            raise RuntimeError("full state is not currently revealed")
        return _state(self.current)
