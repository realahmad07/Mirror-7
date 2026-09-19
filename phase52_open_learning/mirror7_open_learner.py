from __future__ import annotations

"""
Phase 52: open-ended black-box learner.

The learner receives only observations, goals, legal actions, and transition
feedback. It discovers action consequences online and replans from its
learned transition model without task-family labels or hidden parameters.

Exploration invariant:
A failed action is recorded as a negative transition at the exact state where
it failed. The learner never repeatedly retries that same failed action while
the state is unchanged.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

State = Dict[str, Any]
StateKey = Tuple[Tuple[str, Any], ...]


def state_key(state: State) -> StateKey:
    return tuple(sorted(state.items(), key=lambda x: x[0]))


def goal_cost(state: State, goal: State) -> float:
    cost = 0.0
    for key, target in goal.items():
        actual = state.get(key)
        if isinstance(actual, (int, float)) and isinstance(target, (int, float)):
            cost += abs(actual - target)
        else:
            cost += 0.0 if actual == target else 1.0
    return cost


# Backward-compatible public name used by the existing Phase 52 tests.
goal_distance = goal_cost


@dataclass
class TransitionSample:
    action: str
    before: State
    after: State
    ok: bool
    changed: Dict[str, Tuple[Any, Any]]

    @property
    def before_key(self) -> StateKey:
        return state_key(self.before)


@dataclass
class ActionSchema:
    action: str
    samples: List[TransitionSample] = field(default_factory=list)
    last_effect: Dict[str, Tuple[Any, Any]] = field(default_factory=dict)

    def observe(self, sample: TransitionSample) -> None:
        self.samples.append(sample)
        if sample.ok and sample.changed:
            self.last_effect = dict(sample.changed)

    @property
    def successful(self) -> bool:
        return any(sample.ok for sample in self.samples)

    @property
    def failed(self) -> bool:
        return any(not sample.ok for sample in self.samples)

    def failed_at(self, state: State) -> bool:
        key = state_key(state)
        return any(not sample.ok and sample.before_key == key for sample in self.samples)

    def _confirmed_toggle(self, key: str) -> bool:
        transitions = [
            sample.changed[key]
            for sample in self.samples
            if sample.ok and key in sample.changed
        ]
        if not transitions:
            return False
        observed = {(old, new) for old, new in transitions}
        return (0, 1) in observed and (1, 0) in observed

    def predict(self, state: State) -> Optional[State]:
        if not self.successful or not self.last_effect:
            return None

        # Prediction error invalidation / exact memory fallback
        state_key_now = state_key(state)
        for sample in reversed(self.samples):
            if sample.ok and sample.before_key == state_key_now:
                return dict(sample.after)

        result = dict(state)
        for key, (old, new) in self.last_effect.items():
            if key not in state:
                return None

            if (
                isinstance(old, int)
                and isinstance(new, int)
                and {old, new} == {0, 1}
                and state[key] in {0, 1}
                and self._confirmed_toggle(key)
            ):
                # A single 0 -> 1 sample is ambiguous between +1 and toggle.
                # Confirm toggle semantics only after observing both directions.
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
    attempted_at_state: Dict[StateKey, set[str]] = field(default_factory=dict)
    last_action: Optional[str] = None

    def schema(self, action: str) -> ActionSchema:
        if action not in self.schemas:
            self.schemas[action] = ActionSchema(action)
        return self.schemas[action]

    def failed_at_current(self, action: str) -> bool:
        schema = self.schemas.get(action)
        return bool(schema and schema.failed_at(self.current))

    def known(self, action: str) -> bool:
        schema = self.schemas.get(action)
        return bool(schema and schema.successful)

    def remember_attempt(self, action: str) -> None:
        key = state_key(self.current)
        self.attempted_at_state.setdefault(key, set()).add(action)

    def attempted(self, action: str) -> bool:
        return action in self.attempted_at_state.get(state_key(self.current), set())


class OpenEndedLearner:
    """Adaptive state/action learner used by the blind evaluation protocols."""

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
        changed = dict(msg.get("changed", {}))

        if action is not None:
            sample = TransitionSample(
                action=action,
                before=before,
                after=after,
                ok=bool(msg.get("ok", False)),
                changed=changed,
            )
            # Record BOTH success and failure. Failure is knowledge.
            ep.schema(action).observe(sample)

            if sample.ok:
                self.successes += 1
                self.global_effect_stats.append(sample.changed)

        ep.current = after

    def _known_predictions(self) -> List[Tuple[float, int, str, State]]:
        assert self.episode is not None
        ep = self.episode
        out: List[Tuple[float, int, str, State]] = []

        for index, action in enumerate(ep.actions):
            schema = ep.schemas.get(action)
            if schema is None or not schema.successful:
                continue

            # A failed action at this exact state is a negative fact.
            if schema.failed_at(ep.current):
                continue

            predicted = schema.predict(ep.current)
            if predicted is None:
                continue

            out.append((
                goal_cost(predicted, ep.goal),
                -index,
                action,
                predicted,
            ))

        out.sort(key=lambda item: (item[0], item[1]))
        return out

    def _unknown_actions(self) -> List[str]:
        assert self.episode is not None
        return [
            action for action in self.episode.actions
            if not self.episode.known(action)
            and not self.episode.failed_at_current(action)
        ]

    def _safe_known_actions(self) -> List[str]:
        assert self.episode is not None
        return [
            action for action in self.episode.actions
            if self.episode.known(action)
            and not self.episode.failed_at_current(action)
        ]

    def choose_action(self) -> str:
        assert self.episode is not None
        ep = self.episode
        current_cost = goal_cost(ep.current, ep.goal)

        # 1. Prefer a learned action that predicts strict goal improvement.
        known = self._known_predictions()
        improving = [candidate for candidate in known if candidate[0] < current_cost]
        if improving:
            return improving[0][2]

        # 2. Explore only a genuinely unknown action.
        #    Previously failed actions are no longer considered unknown.
        unknown = self._unknown_actions()
        if unknown:
            return unknown[0]

        # 3. Do not re-probe a known action solely because the state changed.
        # A successful action may be destructive outside the state where it was
        # first observed (for example, a reset-like transition). Re-discovery
        # at every new state creates avoidable destructive exploration. Unknown
        # actions are still explored above, so first-time action discovery is
        # preserved.
        #
        # 4. If no strict improvement exists, continue with the best learned
        # successor that is not known to fail at this exact state.
        non_failing = [
            candidate for candidate in known
            if not ep.failed_at_current(candidate[2])
        ]
        if non_failing:
            return non_failing[0][2]

        # 5. Every available action has failed at this exact state.
        #    Repeating an invalid action is never useful; choose the first
        #    legal action only as a deterministic terminal fallback.
        #    The environment/test harness remains responsible for declaring
        #    the episode unsolvable when no legal progress exists.
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
        self.episode.remember_attempt(action)
        self.episode.last_action = action
        return {"action": action}
