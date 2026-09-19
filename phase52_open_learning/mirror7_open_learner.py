from __future__ import annotations

"""
Phase 52: open-ended black-box learner.

The learner receives only observations, goals, legal actions, and transition
feedback. It discovers action consequences online and replans from its
learned transition model without task-family labels or hidden parameters.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

State = Dict[str, Any]


def state_key(state: State) -> Tuple[Tuple[str, Any], ...]:
    return tuple(sorted(state.items(), key=lambda x: x[0]))


def goal_cost(state: State, goal: State) -> float:
    """Numerical closeness when possible, exact mismatch otherwise."""
    cost = 0.0
    for key, target in goal.items():
        actual = state.get(key)
        if isinstance(actual, (int, float)) and isinstance(target, (int, float)):
            cost += abs(actual - target)
        else:
            cost += 0.0 if actual == target else 1.0
    return cost


@dataclass
class TransitionSample:
    action: str
    before: State
    after: State
    ok: bool
    changed: Dict[str, Tuple[Any, Any]]


@dataclass
class ActionSchema:
    action: str
    samples: List[TransitionSample] = field(default_factory=list)
    last_effect: Dict[str, Tuple[Any, Any]] = field(default_factory=dict)

    def observe(self, sample: TransitionSample) -> None:
        self.samples.append(sample)
        self.last_effect = dict(sample.changed)

    def predict(self, state: State) -> Optional[State]:
        if not self.samples:
            return None

        result = dict(state)
        for key, (old, new) in self.last_effect.items():
            if key not in state:
                return None
            if (
                isinstance(old, int)
                and isinstance(new, int)
                and {old, new} == {0, 1}
                and state[key] in {0, 1}
            ):
                # Binary observed effect: infer a toggle, not an unbounded +1.
                result[key] = 1 - state[key]
            elif isinstance(old, (int, float)) and isinstance(new, (int, float)):
                result[key] = state[key] + (new - old)
            else:
                result[key] = new if state[key] == old else old
        return result


@dataclass
class EpisodeMemory:
    current: State
    goal: State
    actions: List[str]
    schemas: Dict[str, ActionSchema] = field(default_factory=dict)
    tried_at_state: Dict[Tuple[Tuple[str, Any], ...], List[str]] = field(default_factory=dict)
    last_action: Optional[str] = None

    def schema(self, action: str) -> ActionSchema:
        if action not in self.schemas:
            self.schemas[action] = ActionSchema(action)
        return self.schemas[action]

    def remember_try(self, action: str) -> None:
        key = state_key(self.current)
        self.tried_at_state.setdefault(key, []).append(action)

    def tried(self, action: str) -> bool:
        return action in self.tried_at_state.get(state_key(self.current), [])


class OpenEndedLearner:
    """General adaptive state/action learner used by the Phase 53 protocol."""

    def __init__(self) -> None:
        self.episodes_seen = 0
        self.successes = 0
        self.global_effect_stats: List[Dict[str, Any]] = []
        self.episode: Optional[EpisodeMemory] = None

    def _start(self, msg: dict) -> None:
        self.episodes_seen += 1
        self.episode = EpisodeMemory(
            current=dict(msg["observation"]),
            goal=dict(msg["goal"]),
            actions=list(msg["legal_actions"]),
        )

    def _observe(self, msg: dict) -> None:
        assert self.episode is not None
        ep = self.episode
        before = dict(ep.current)
        after = dict(msg["observation"])
        action = ep.last_action

        if action is not None:
            changed = dict(msg.get("changed", {}))
            sample = TransitionSample(
                action=action,
                before=before,
                after=after,
                ok=bool(msg.get("ok", False)),
                changed=changed,
            )
            if sample.ok:
                ep.schema(action).observe(sample)
                self.global_effect_stats.append(changed)

        ep.current = after

    def _known_predictions(self) -> List[Tuple[float, int, str, State]]:
        assert self.episode is not None
        ep = self.episode
        candidates: List[Tuple[float, int, str, State]] = []

        for index, action in enumerate(ep.actions):
            schema = ep.schemas.get(action)
            if schema is None or not schema.samples:
                continue
            predicted = schema.predict(ep.current)
            if predicted is None:
                continue
            candidates.append((
                goal_cost(predicted, ep.goal),
                -index,
                action,
                predicted,
            ))

        candidates.sort(key=lambda item: (item[0], item[1]))
        return candidates

    def choose_action(self) -> str:
        assert self.episode is not None
        ep = self.episode
        current_cost = goal_cost(ep.current, ep.goal)
        known = self._known_predictions()

        # Prefer a learned action that strictly improves the goal cost.
        improving = [c for c in known if c[0] < current_cost]
        if improving:
            return improving[0][2]

        # When no learned action improves, probe one previously untried action.
        # The evaluator provides only legal actions; the learner has no
        # access to hidden preconditions or family labels.
        unknown = [a for a in ep.actions if a not in ep.schemas]
        if unknown:
            return unknown[0]

        # If everything is known, take the least-cost predicted successor,
        # even when currently neutral. This permits state-dependent rules
        # to reveal a new direction after replanning.
        if known:
            return known[0][2]

        return ep.actions[0]

    def handle(self, msg: dict) -> dict:
        kind = msg.get("type")
        if kind == "episode_start":
            self._start(msg)
        elif kind == "transition":
            self._observe(msg)
        else:
            raise ValueError(f"Unsupported protocol message: {kind!r}")

        assert self.episode is not None
        action = self.choose_action()
        self.episode.remember_try(action)
        self.episode.last_action = action
        return {"action": action}
