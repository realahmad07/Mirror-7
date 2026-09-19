"""Mirror 7 Phase 34: goal-directed reasoning and planning.

The module consumes an explicit transition model learned from observations and
searches predicted states to satisfy a goal. It does not contain task-specific
answers or hidden environment semantics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from heapq import heappop, heappush
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

State = Tuple[Tuple[str, Any], ...]


def freeze_state(state: Mapping[str, Any]) -> State:
    return tuple(sorted(state.items(), key=lambda kv: kv[0]))


def thaw_state(state: State) -> Dict[str, Any]:
    return dict(state)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


@dataclass(frozen=True)
class Goal:
    requirements: Tuple[Tuple[str, Any], ...]

    @classmethod
    def from_mapping(cls, requirements: Mapping[str, Any]) -> "Goal":
        return cls(tuple(sorted(requirements.items(), key=lambda kv: kv[0])))

    def satisfied_by(self, state: State) -> bool:
        current = thaw_state(state)
        return all(current.get(k) == v for k, v in self.requirements)

    def distance(self, state: State) -> int:
        current = thaw_state(state)
        return sum(current.get(k) != v for k, v in self.requirements)


@dataclass(frozen=True)
class Observation:
    state_before: State
    action: str
    state_after: State


@dataclass
class ActionRule:
    action: str
    observations: List[Observation] = field(default_factory=list)
    numeric_deltas: Dict[str, float] = field(default_factory=dict)
    numeric_sources: Dict[str, Tuple[Any, ...]] = field(default_factory=dict)
    categorical_map: Dict[str, Dict[Any, Any]] = field(default_factory=dict)

    def learn(self, observation: Observation) -> None:
        self.observations.append(observation)
        before = thaw_state(observation.state_before)
        after = thaw_state(observation.state_after)
        for key in sorted(set(before) | set(after)):
            old = before.get(key)
            new = after.get(key)
            if _is_number(old) and _is_number(new):
                deltas = []
                for prior_obs in self.observations:
                    prior_before = thaw_state(prior_obs.state_before).get(key)
                    prior_after = thaw_state(prior_obs.state_after).get(key)
                    if _is_number(prior_before) and _is_number(prior_after):
                        deltas.append(prior_after - prior_before)
                if len(deltas) >= 2 and all(d == deltas[0] for d in deltas):
                    self.numeric_deltas[key] = deltas[0]
                    self.numeric_sources[key] = tuple(sorted(
                        thaw_state(obs.state_before).get(key)
                        for obs in self.observations
                        if _is_number(thaw_state(obs.state_before).get(key))
                        and _is_number(thaw_state(obs.state_after).get(key))
                    ))
                else:
                    self.numeric_deltas.pop(key, None)
                    self.numeric_sources.pop(key, None)
            elif old != new:
                mapping = self.categorical_map.setdefault(key, {})
                prior = mapping.get(old)
                if prior is None:
                    mapping[old] = new
                elif prior != new:
                    mapping.pop(old, None)

    def predict(self, state: State) -> Optional[State]:
        current = thaw_state(state)
        exact = [o for o in reversed(self.observations) if o.state_before == state]
        if exact:
            return exact[0].state_after

        changed = False
        predicted = dict(current)
        for key, delta in self.numeric_deltas.items():
            if key in predicted and predicted[key] in self.numeric_sources.get(key, ()):
                predicted[key] = predicted[key] + delta
                changed = True
        for key, mapping in self.categorical_map.items():
            old = predicted.get(key)
            if old in mapping:
                predicted[key] = mapping[old]
                changed = True
        return freeze_state(predicted) if changed else None

    @property
    def sample_count(self) -> int:
        return len(self.observations)


class TransitionModel:
    def __init__(self) -> None:
        self._rules: Dict[str, ActionRule] = {}

    def observe(self, state_before: Mapping[str, Any], action: str, state_after: Mapping[str, Any]) -> None:
        obs = Observation(freeze_state(state_before), action, freeze_state(state_after))
        self._rules.setdefault(action, ActionRule(action)).learn(obs)

    def actions(self) -> Tuple[str, ...]:
        return tuple(sorted(self._rules))

    def predict(self, state: State, action: str) -> Optional[State]:
        rule = self._rules.get(action)
        return rule.predict(state) if rule else None

    def evidence(self, action: str) -> int:
        rule = self._rules.get(action)
        return rule.sample_count if rule else 0


@dataclass(frozen=True)
class PlanStep:
    action: str
    state_before: State
    predicted_state_after: State
    goal_distance_after: int


@dataclass(frozen=True)
class Plan:
    steps: Tuple[PlanStep, ...]
    expanded_nodes: int
    goal_distance: int

    @property
    def actions(self) -> Tuple[str, ...]:
        return tuple(step.action for step in self.steps)


class GoalPlanner:
    def __init__(self, transition_model: TransitionModel) -> None:
        self.model = transition_model
        self.trace: List[Dict[str, Any]] = []

    def plan(self, start: Mapping[str, Any], goal: Goal, max_depth: int = 12, max_nodes: int = 1000) -> Optional[Plan]:
        start_state = freeze_state(start)
        self.trace = [{"event": "start", "state": start_state, "goal": goal.requirements}]
        if goal.satisfied_by(start_state):
            return Plan(steps=tuple(), expanded_nodes=0, goal_distance=0)

        queue: List[Tuple[int, int, State, Tuple[PlanStep, ...]]] = []
        counter = 0
        heappush(queue, (goal.distance(start_state), counter, start_state, tuple()))
        best_depth: Dict[State, int] = {start_state: 0}
        expanded = 0

        while queue and expanded < max_nodes:
            _, _, state, steps = heappop(queue)
            depth = len(steps)
            expanded += 1
            self.trace.append({"event": "expand", "state": state, "depth": depth})

            if goal.satisfied_by(state):
                return Plan(tuple(steps), expanded, 0)
            if depth >= max_depth:
                continue

            for action in self.model.actions():
                predicted = self.model.predict(state, action)
                if predicted is None or predicted == state:
                    self.trace.append({"event": "reject", "action": action, "reason": "no_progress_or_unknown"})
                    continue
                next_depth = depth + 1
                if predicted in best_depth and best_depth[predicted] <= next_depth:
                    self.trace.append({"event": "reject", "action": action, "reason": "cycle_or_dominated"})
                    continue
                best_depth[predicted] = next_depth
                distance = goal.distance(predicted)
                step = PlanStep(action, state, predicted, distance)
                counter += 1
                priority = next_depth + distance
                heappush(queue, (priority, counter, predicted, steps + (step,)))
                self.trace.append({"event": "enqueue", "action": action, "state": predicted, "distance": distance})

        self.trace.append({"event": "failure", "reason": "search_exhausted", "expanded": expanded})
        return None

    def validate_plan(self, plan: Plan, start: Mapping[str, Any], goal: Goal) -> bool:
        state = freeze_state(start)
        for step in plan.steps:
            if step.state_before != state:
                return False
            predicted = self.model.predict(state, step.action)
            if predicted != step.predicted_state_after:
                return False
            state = predicted
        return goal.satisfied_by(state)


def train_model_from_trajectory(
    model: TransitionModel,
    trajectory: Sequence[Tuple[Mapping[str, Any], str, Mapping[str, Any]]],
) -> None:
    for before, action, after in trajectory:
        model.observe(before, action, after)
