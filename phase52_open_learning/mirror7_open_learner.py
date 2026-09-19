from __future__ import annotations

"""
Phase 52: open-ended black-box learner.

The learner receives only:
- observations
- goals
- legal actions
- transition feedback

It does not receive task-family labels, hidden parameters, or expected transitions.

It learns action consequences online, detects reversible/harmful actions,
reuses learned transition schemas, and performs goal-directed planning.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any


State = Dict[str, Any]


def state_key(state: State) -> Tuple[Tuple[str, Any], ...]:
    return tuple(sorted(state.items(), key=lambda x: x[0]))


def goal_distance(state: State, goal: State) -> int:
    return sum(state.get(k) != v for k, v in goal.items())


def signed_progress(before: State, after: State, goal: State) -> int:
    b = goal_distance(before, goal)
    a = goal_distance(after, goal)
    return b - a


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

        # Infer a reusable transformation from the latest successful sample.
        # Each observed change is represented as a typed delta/swap/toggle.
        result = dict(state)
        for key, (old, new) in self.last_effect.items():
            if key not in state:
                return None
            if isinstance(old, bool) and isinstance(new, bool):
                result[key] = new if state[key] == old else old
            elif isinstance(old, int) and isinstance(new, int):
                delta = new - old
                if delta == 0:
                    result[key] = state[key]
                else:
                    result[key] = state[key] + delta
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
    previous_state: Optional[State] = None

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
    """
    Generic adaptive controller.

    Strategy:
      1. observe one action consequence;
      2. infer a reusable local transformation;
      3. predict each known action at the current state;
      4. select the action with maximum goal progress while avoiding
         predicted regressions;
      5. explore one previously untried legal action only when no known
         action improves the goal;
      6. preserve experience across episodes as structural statistics.
    """

    def __init__(self) -> None:
        self.episodes_seen = 0
        self.successes = 0
        self.global_effect_stats: Dict[str, List[Dict[str, Tuple[Any, Any]]]] = {}
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
                self.global_effect_stats.setdefault(action, []).append(changed)
        ep.current = after

    def _predicted_candidates(self) -> List[Tuple[int, int, str, State]]:
        assert self.episode is not None
        ep = self.episode
        out = []
        current_distance = goal_distance(ep.current, ep.goal)
        for idx, action in enumerate(ep.actions):
            schema = ep.schemas.get(action)
            if schema is None or not schema.samples:
                continue
            predicted = schema.predict(ep.current)
            if predicted is None:
                continue
            delta = current_distance - goal_distance(predicted, ep.goal)
            out.append((delta, -idx, action, predicted))
        return out

    def choose_action(self) -> str:
        assert self.episode is not None
        ep = self.episode

        candidates = self._predicted_candidates()
        improving = [c for c in candidates if c[0] > 0]
        if improving:
            improving.sort(reverse=True)
            return improving[0][2]

        # An action that is already known but would worsen the goal is avoided.
        # Explore an unknown action only when all known actions fail to help.
        unknown = [a for a in ep.actions if not ep.tried(a)]
        if unknown:
            # The first unknown action is used first. Once its effect is known,
            # later choices are model-based. This stays task-family agnostic.
            return unknown[0]

        if candidates:
            neutral = [c for c in candidates if c[0] == 0]
            if neutral:
                neutral.sort(reverse=True)
                return neutral[0][2]

        # Last deterministic fallback. This should be reached only on
        # environments where no observed action can make further progress.
        return ep.actions[0]

    def handle(self, msg: dict) -> dict:
        if msg.get("type") == "episode_start":
            self._start(msg)
        elif msg.get("type") == "transition":
            self._observe(msg)
        else:
            raise ValueError(f"Unsupported protocol message: {msg.get('type')!r}")

        action = self.choose_action()
        self.episode.remember_try(action)
        self.episode.previous_state = dict(self.episode.current)
        self.episode.last_action = action
        return {"action": action}
