"""Mirror 7 Phase 61: predictive closed-loop autonomy.

Combines Phase 58 hierarchy discovery, Phase 59 concept prediction, and Phase
60 structured world modeling into a bounded goal-directed control loop.

The agent receives only the current state, goal, and legal actions. It learns
from observed transitions, builds short model-based plans when evidence allows,
uses successful action histories as a predictive prior, replans after observed
model discrepancy, and explores unknown legal actions when necessary.
"""
from __future__ import annotations
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from typing import Hashable, Sequence, Tuple

from phase58_hierarchical_abstraction.mirror7_phase58 import discover_hierarchy
from phase59_predictive_concepts.mirror7_phase59 import PredictiveConceptModel
from phase60_long_horizon_world_model.mirror7_phase60 import LongHorizonWorldModel

State = Tuple[Hashable, ...]
Action = Hashable

@dataclass(frozen=True)
class Decision:
    action: Action
    predicted_state: State | None
    score: float
    reason: str
    planned: bool

@dataclass(frozen=True)
class EpisodeResult:
    solved: bool
    steps: int
    invalid_actions: int
    discrepancies: int
    replans: int
    exploratory_steps: int
    final_state: State
    actions: Tuple[Action, ...]


def _state(state: Sequence[Hashable]) -> State:
    if not isinstance(state, (tuple, list)) or not state:
        raise ValueError("state must be a non-empty tuple/list")
    return tuple(state)


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


class PredictiveClosedLoopAgent:
    def __init__(self, *, max_plan_depth: int = 8, max_exploration: int = 64):
        if max_plan_depth < 1 or max_exploration < 1:
            raise ValueError("invalid agent bounds")
        self.max_plan_depth = max_plan_depth
        self.max_exploration = max_exploration
        self.world = LongHorizonWorldModel(min_support=2)
        self._fitted = False
        self._memory: dict[tuple[State, Action], Counter[State]] = defaultdict(Counter)
        self._tries: Counter[tuple[State, Action]] = Counter()
        self._bad: Counter[tuple[State, Action]] = Counter()
        self._plan: deque[Action] = deque()
        self._current: State | None = None
        self._goal: State | None = None
        self._trace: list[Action] = []
        self.success_traces: list[Tuple[Action, ...]] = []
        self.hierarchy = None
        self.predictor = None
        self.replans = 0
        self.discrepancies = 0
        self.exploration_steps = 0
        self.invalid_actions = 0

    def reset(self, state: Sequence[Hashable], goal: Sequence[Hashable]) -> None:
        self._current = _state(state)
        self._goal = _state(goal)
        if len(self._current) != len(self._goal):
            raise ValueError("state/goal dimensionality mismatch")
        self._plan.clear()
        self._trace.clear()
        self.replans = 0
        self.discrepancies = 0
        self.exploration_steps = 0
        self.invalid_actions = 0

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

    def learn_demonstration(
        self,
        transitions: Sequence[tuple[State, Action, State]],
        actions: Sequence[Action],
    ) -> None:
        if len(transitions) != len(actions) or not transitions:
            raise ValueError("demonstration must contain aligned transitions and actions")
        for (s, a, n), action in zip(transitions, actions):
            s, n = _state(s), _state(n)
            if a != action:
                raise ValueError("transition/action mismatch")
            self._memory[(s, a)][n] += 1
            self.world.update(s, a, n)
        self._fitted = True
        self.success_traces.append(tuple(actions))
        self._refresh_abstraction()

    def _refresh_abstraction(self) -> None:
        if len(self.success_traces) < 2:
            return
        self.hierarchy = discover_hierarchy(
            self.success_traces,
            max_depth=2,
            min_len=2,
            max_len=4,
            min_episode_support=2,
        )
        corpus = (
            self.hierarchy.transformed_episodes
            if self.hierarchy.transformed_episodes
            else tuple(self.success_traces)
        )
        if corpus and all(corpus):
            self.predictor = PredictiveConceptModel(
                max_order=3, min_support=2
            ).fit(corpus)

    def _predict_known(self, state: State, action: Action) -> State | None:
        counts = self._memory.get((state, action))
        if counts:
            best = counts.most_common()
            if len(best) == 1 or best[0][1] > best[1][1]:
                return best[0][0]
        if self._fitted:
            result = self.world.predict(state, action)
            if result.known:
                return result.next_state
        return None

    def _find_plan(self, legal: Sequence[Action]) -> list[Action]:
        start, goal = self.state, self.goal
        if start == goal:
            return []
        queue = deque([(start, [])])
        seen = {start}
        while queue:
            cur, path = queue.popleft()
            if len(path) >= self.max_plan_depth:
                continue
            for action in sorted(set(legal), key=repr):
                nxt = self._predict_known(cur, action)
                if nxt is None or nxt in seen:
                    continue
                new_path = path + [action]
                if nxt == goal:
                    return new_path
                seen.add(nxt)
                queue.append((nxt, new_path))
        return []

    def _prior_action(self, legal: Sequence[Action]) -> Action | None:
        if self.predictor is None or not self._trace:
            return None
        prediction = self.predictor.predict(self._trace)
        if prediction.value in legal:
            return prediction.value
        return None

    def decide(
        self,
        state: Sequence[Hashable],
        goal: Sequence[Hashable],
        legal_actions: Sequence[Action],
    ) -> Decision:
        state, goal = _state(state), _state(goal)
        legal = tuple(dict.fromkeys(legal_actions))
        if not legal:
            raise ValueError("legal_actions cannot be empty")
        if len(state) != len(goal):
            raise ValueError("state/goal dimensionality mismatch")
        if self._current is None:
            self.reset(state, goal)
        else:
            self._current, self._goal = state, goal
        if state == goal:
            raise RuntimeError("goal already satisfied")

        path = self._find_plan(legal)
        if path:
            self._plan = deque(path[1:])
            action = path[0]
            nxt = self._predict_known(state, action)
            return Decision(
                action,
                nxt,
                -_distance(nxt, goal) if nxt is not None else 0.0,
                "model-plan",
                True,
            )

        self.replans += 1 if self._plan else 0
        prior = self._prior_action(legal)
        current_distance = _distance(state, goal)
        candidates = []
        improving_known = []
        unknown = []

        for action in legal:
            nxt = self._predict_known(state, action)
            if nxt is not None:
                d = _distance(nxt, goal)
                score = (
                    -(d * 10.0)
                    + (0.5 if prior == action else 0.0)
                    - 3.0 * self._bad[(state, action)]
                )
                if d > current_distance:
                    score -= 1.0
                if nxt == state:
                    score -= 2.0
                item = (
                    score,
                    0,
                    -self._tries[(state, action)],
                    repr(action),
                    action,
                    nxt,
                )
                candidates.append(item)
                if d < current_distance:
                    improving_known.append(item)
            elif self._tries[(state, action)] < 1:
                # When no learned action improves the goal, probe each legal
                # unknown action once before repeating a non-progress action.
                score = -20.0 + (0.5 if prior == action else 0.0)
                item = (
                    score,
                    1,
                    -self._tries[(state, action)],
                    repr(action),
                    action,
                    None,
                )
                candidates.append(item)
                unknown.append(item)

        if improving_known:
            candidates = improving_known
        elif unknown:
            candidates = unknown
        elif not candidates:
            # Every legal action was already probed and remains uncertain.
            # Choose the least harmful known option rather than inventing one.
            for action in legal:
                nxt = self._predict_known(state, action)
                if nxt is not None:
                    d = _distance(nxt, goal)
                    candidates.append(
                        (
                            -(d * 10.0) - 3.0 * self._bad[(state, action)],
                            0,
                            -self._tries[(state, action)],
                            repr(action),
                            action,
                            nxt,
                        )
                    )

        if not candidates:
            raise RuntimeError("no safe or known legal action remains")

        candidates.sort(reverse=True)
        _, kind, _, _, action, nxt = candidates[0]
        if kind == 1:
            self.exploration_steps += 1
        return Decision(
            action,
            nxt,
            candidates[0][0],
            "explore" if kind else "model-greedy",
            False,
        )

    def observe(
        self,
        action: Action,
        next_state: Sequence[Hashable],
        *,
        legal_actions: Sequence[Action] | None = None,
    ) -> dict:
        if self._current is None:
            raise RuntimeError("agent is not reset")
        next_state = _state(next_state)
        if legal_actions is not None and action not in set(legal_actions):
            self.invalid_actions += 1
            raise ValueError("attempted action was not legal")

        self._tries[(self.state, action)] += 1
        predicted = self._predict_known(self.state, action)
        discrepancy = predicted is not None and predicted != next_state
        if discrepancy:
            self.discrepancies += 1
            self.replans += 1
            self._bad[(self.state, action)] += 1
            self._plan.clear()

        self._memory[(self.state, action)][next_state] += 1
        self.world.update(self.state, action, next_state)
        self._fitted = True
        self._trace.append(action)
        self._current = next_state

        return {
            "discrepancy": discrepancy,
            "predicted": predicted,
            "observed": next_state,
        }

    def finish(self, solved: bool) -> None:
        if solved and self._trace:
            self.success_traces.append(tuple(self._trace))
            self._refresh_abstraction()

    def run(
        self,
        env,
        *,
        start: Sequence[Hashable],
        goal: Sequence[Hashable],
        max_steps: int,
    ) -> EpisodeResult:
        if hasattr(env, "reset"):
            try:
                env.reset(start)
            except TypeError:
                env.reset()

        self.reset(start, goal)
        actions = []

        for _ in range(max_steps):
            if self.state == self.goal:
                self.finish(True)
                return EpisodeResult(
                    True,
                    len(actions),
                    self.invalid_actions,
                    self.discrepancies,
                    self.replans,
                    self.exploration_steps,
                    self.state,
                    tuple(actions),
                )

            legal = env.legal_actions(self.state)
            decision = self.decide(self.state, self.goal, legal)
            if decision.action not in legal:
                self.invalid_actions += 1
                raise AssertionError("agent selected non-legal action")

            nxt = env.step(decision.action)
            actions.append(decision.action)
            self.observe(decision.action, nxt, legal_actions=legal)

        solved = self.state == self.goal
        self.finish(solved)
        return EpisodeResult(
            solved,
            len(actions),
            self.invalid_actions,
            self.discrepancies,
            self.replans,
            self.exploration_steps,
            self.state,
            tuple(actions),
        )
