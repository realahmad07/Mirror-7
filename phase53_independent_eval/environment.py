from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from typing import Any, Dict, Tuple


@dataclass(frozen=True)
class HiddenTaskSpec:
    task_id: str
    family: str
    seed: int
    split: str
    field_names: Tuple[str, ...]
    action_names: Tuple[str, ...]
    initial: Dict[str, int]
    goal: Dict[str, int]
    max_steps: int
    hidden_parameters: Dict[str, Any]


class HiddenRuleEnvironment:
    """Hidden-rule environment. Secret rule data never enters the public protocol."""

    def __init__(self, spec: HiddenTaskSpec):
        self.spec = spec
        self.state = dict(spec.initial)
        self.steps = 0

    def public_init(self) -> dict:
        return {
            "type": "episode_start",
            "episode_id": self.spec.task_id,
            "observation": dict(self.state),
            "goal": dict(self.spec.goal),
            "legal_actions": list(self.spec.action_names),
            "step_limit": self.spec.max_steps,
        }

    def step(self, action: str) -> dict:
        self.steps += 1
        ok = False
        old = dict(self.state)
        hp = self.spec.hidden_parameters

        if action not in self.spec.action_names:
            ok = False
        elif self.spec.family == "linear" and action == hp["action"]:
            self.state[hp["field"]] += hp["delta"]
            ok = True
        elif self.spec.family == "swap" and action == hp["action"]:
            a, b = hp["fields"]
            self.state[a], self.state[b] = self.state[b], self.state[a]
            ok = True
        elif self.spec.family == "gate":
            if action == hp["toggle_action"]:
                self.state[hp["gate_field"]] ^= 1
                ok = True
            elif action == hp["work_action"] and self.state[hp["gate_field"]] == 1:
                self.state[hp["value_field"]] += hp["delta"]
                ok = True
        elif self.spec.family == "conditional" and action == hp["action"]:
            sign = 1 if self.state[hp["condition_field"]] >= 0 else -1
            self.state[hp["value_field"]] += sign * hp["delta"]
            ok = True
        elif self.spec.family == "composition":
            if action == hp["action_a"]:
                self.state[hp["field_a"]] += hp["delta_a"]
                ok = True
            elif action == hp["action_b"]:
                self.state[hp["field_b"]] += hp["delta_b"]
                ok = True

        terminal = self.state_matches_goal() or self.steps >= self.spec.max_steps
        return {
            "type": "transition",
            "ok": ok,
            "observation": dict(self.state),
            "terminal": terminal,
            "reward": 1 if self.state_matches_goal() else 0,
            "steps": self.steps,
            "changed": {k: (old[k], self.state[k]) for k in self.state if old[k] != self.state[k]},
        }

    def state_matches_goal(self) -> bool:
        return all(self.state.get(k) == v for k, v in self.spec.goal.items())


def make_task(family: str, seed: int, split: str) -> HiddenTaskSpec:
    digest = hashlib.sha256(f"mirror7-phase53|{split}|{family}|{seed}".encode()).digest()
    rng = random.Random(int.from_bytes(digest[:8], "big"))

    fields = []
    while len(fields) < 3:
        x = f"z{rng.randrange(1000):03d}"
        if x not in fields:
            fields.append(x)

    actions = []
    while len(actions) < 3:
        x = f"a{rng.randrange(10000):04d}"
        if x not in actions:
            actions.append(x)

    state = {f: 0 for f in fields}
    hidden: dict[str, Any] = {}

    if family == "linear":
        field, action = fields[seed % 3], actions[0]
        delta = (seed % 5) + 1
        state[field] = -(seed % 2)
        goal = {field: state[field] + delta * (2 + seed % 2)}
        hidden.update(field=field, action=action, delta=delta)
    elif family == "swap":
        a, b = fields[:2]
        state[a], state[b] = seed % 3, -(seed % 2)
        goal = {a: state[b], b: state[a]}
        hidden.update(action=actions[0], fields=(a, b))
    elif family == "gate":
        gate_field, value_field = fields[:2]
        state[gate_field] = 0
        state[value_field] = seed % 2
        delta = 1 + seed % 3
        goal = {gate_field: 1, value_field: state[value_field] + delta}
        hidden.update(toggle_action=actions[0], work_action=actions[1],
                     gate_field=gate_field, value_field=value_field, delta=delta)
    elif family == "conditional":
        cond, value = fields[:2]
        state[cond] = -1 if seed % 2 else 1
        state[value] = 0
        delta = 1 + seed % 2
        goal = {value: state[value] + (delta if state[cond] >= 0 else -delta) * 3}
        hidden.update(action=actions[0], condition_field=cond, value_field=value, delta=delta)
    elif family == "composition":
        a, b = fields[:2]
        state[a], state[b] = seed % 2, seed % 3
        goal = {a: state[a] + 2, b: state[b] + 3}
        hidden.update(action_a=actions[0], action_b=actions[1],
                     field_a=a, field_b=b, delta_a=1, delta_b=1)
    else:
        raise ValueError(family)

    opaque_id = "ep-" + hashlib.sha256(f"{split}|{family}|{seed}".encode()).hexdigest()[:16]
    return HiddenTaskSpec(
        task_id=opaque_id, family=family, seed=seed, split=split,
        field_names=tuple(fields), action_names=tuple(actions),
        initial=state, goal=goal, max_steps=40, hidden_parameters=hidden,
    )
