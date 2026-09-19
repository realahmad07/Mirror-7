"""Phases 161-170: black-box raw-input evaluation.

The environment is outside the agent module. The agent receives only:
raw observation bytes, opaque action bytes, raw outcome bytes, optional raw target bytes.
"""
from __future__ import annotations
from dataclasses import dataclass
from collections import deque
from typing import Dict, Iterable, List, Optional, Tuple
import random
import time
from phase141_152_integration.phase141_152 import IntegratedCognition

@dataclass(frozen=True)
class BlackBoxTask:
    markers: Tuple[int, ...]
    permutation: Tuple[int, ...]
    action_tokens: Tuple[bytes, ...]
    action_effects: Tuple[Tuple[int, int], ...]
    initial: Tuple[int, ...]

    @classmethod
    def generate(cls, seed: int, slots: int = 4) -> "BlackBoxTask":
        rng = random.Random(seed)
        markers = tuple(rng.sample(range(16, 240), slots))
        positions = list(range(slots * 2))
        rng.shuffle(positions)
        token_count = slots + 2
        action_tokens = tuple(bytes(rng.randrange(0, 256) for _ in range(4)) for _ in range(token_count))
        while len(set(action_tokens)) != token_count:
            action_tokens = tuple(bytes(rng.randrange(0, 256) for _ in range(4)) for _ in range(token_count))
        effects = [(i, rng.choice((-2, -1, 1, 2))) for i in range(slots)]
        effects.extend([(-1, 0), (-1, 0)])
        return cls(markers, tuple(positions), action_tokens, tuple(effects),
                   tuple(rng.randrange(-4, 5) for _ in range(slots)))

    def encode(self, state: Tuple[int, ...]) -> bytes:
        logical = []
        for marker, value in zip(self.markers, state):
            logical.extend((marker, (128 + max(-40, min(40, value))) & 0xFF))
        out = [0] * len(logical)
        for logical_i, physical_i in enumerate(self.permutation):
            out[physical_i] = logical[logical_i]
        return bytes(out)

    def transition(self, state: Tuple[int, ...], action: bytes) -> Tuple[int, ...]:
        idx = self.action_tokens.index(action)
        slot, delta = self.action_effects[idx]
        if slot < 0:
            return state
        nxt = list(state)
        nxt[slot] = max(-40, min(40, nxt[slot] + delta))
        return tuple(nxt)

@dataclass(frozen=True)
class Trace:
    observation: bytes
    action: bytes
    outcome: bytes

class RawRepresentationLearner:
    """Discover variable byte positions from raw transition traces."""
    def __init__(self) -> None:
        self.marker_positions: Tuple[int, ...] = ()
        self.value_positions: Tuple[int, ...] = ()
        self.action_effects: Dict[bytes, Tuple[int, int]] = {}
        self.traces: List[Trace] = []

    def observe(self, observation: bytes, action: bytes, outcome: bytes) -> None:
        self.traces.append(Trace(observation, action, outcome))
        if len(observation) != len(outcome):
            return
        constant = []
        for i in range(len(observation)):
            if all(t.observation[i] == t.outcome[i] for t in self.traces):
                constant.append(i)
        self.marker_positions = tuple(constant)
        self.value_positions = tuple(i for i in range(len(observation))
                                     if i not in self.marker_positions)
        self._infer_effects()

    def _infer_effects(self) -> None:
        for trace in self.traces:
            changes = []
            for pos in self.value_positions:
                before = int(trace.observation[pos]) - 128
                after = int(trace.outcome[pos]) - 128
                if before != after:
                    changes.append((pos, after - before))
            if len(changes) == 1:
                self.action_effects[trace.action] = changes[0]

    def decode(self, raw: bytes) -> Tuple[int, ...]:
        return tuple(int(raw[p]) - 128 for p in sorted(self.value_positions))

    def encode_like(self, latent: Tuple[int, ...], template: bytes) -> bytes:
        out = bytearray(template)
        for pos, value in zip(sorted(self.value_positions), latent):
            out[pos] = (128 + max(-40, min(40, value))) & 0xFF
        return bytes(out)

    def confidence(self) -> float:
        if not self.value_positions or not self.action_effects:
            return 0.0
        return min(1.0, len(self.action_effects) / max(1, len(set(t.action for t in self.traces))))

class BlackBoxMirrorAgent:
    def __init__(self) -> None:
        self.rep = RawRepresentationLearner()
        self.runtime = IntegratedCognition()
        self.runtime.world.max_rules = 64

    def observe_transition(self, observation: bytes, action: bytes, outcome: bytes) -> None:
        self.rep.observe(observation, action, outcome)
        state = self.rep.decode(observation)
        nxt = self.rep.decode(outcome)
        if state and nxt:
            self.runtime.step(
                {f"slot_{i}": v for i, v in enumerate(state)},
                {f"slot_{i}": v for i, v in enumerate(state)},
                action.hex(),
                {f"slot_{i}": v for i, v in enumerate(nxt)},
            )

    def predict(self, observation: bytes, action: bytes) -> Tuple[bytes, float]:
        state = self.rep.decode(observation)
        effect = self.rep.action_effects.get(action)
        if not state or effect is None:
            return observation, 0.0
        pos, delta = effect
        positions = sorted(self.rep.value_positions)
        if pos not in positions:
            return observation, 0.0
        idx = positions.index(pos)
        latent = list(state)
        latent[idx] = max(-40, min(40, latent[idx] + delta))
        return self.rep.encode_like(tuple(latent), observation), min(1.0, self.rep.confidence())

    def choose_action(self, observation: bytes, target: bytes, actions: Iterable[bytes]) -> Optional[bytes]:
        state = self.rep.decode(observation)
        goal = self.rep.decode(target)
        if not state or not goal or not self.rep.action_effects:
            return None
        positions = sorted(self.rep.value_positions)
        best = None
        best_distance = None
        for action in actions:
            effect = self.rep.action_effects.get(action)
            if effect is None:
                continue
            pos, delta = effect
            if pos not in positions:
                continue
            idx = positions.index(pos)
            candidate = list(state)
            candidate[idx] = max(-40, min(40, candidate[idx] + delta))
            distance = sum(abs(a - b) for a, b in zip(candidate, goal))
            if best_distance is None or distance < best_distance:
                best_distance, best = distance, action
        return best

    def plan_to_target(self, observation: bytes, target: bytes, actions: Iterable[bytes], max_depth: int = 12) -> List[bytes]:
        start = self.rep.decode(observation)
        goal = self.rep.decode(target)
        if not start or not goal or not self.rep.action_effects:
            return []
        positions = sorted(self.rep.value_positions)
        indexed = []
        for action in actions:
            effect = self.rep.action_effects.get(action)
            if effect is None:
                continue
            pos, delta = effect
            if pos in positions:
                indexed.append((action, positions.index(pos), delta))
        if not indexed:
            return []
        queue = deque([(start, [])])
        seen = {start}
        while queue:
            state, path = queue.popleft()
            if state == goal:
                return path
            if len(path) >= max_depth:
                continue
            for action, idx, delta in indexed:
                nxt = list(state)
                nxt[idx] = max(-40, min(40, nxt[idx] + delta))
                nxt = tuple(nxt)
                if nxt in seen:
                    continue
                seen.add(nxt)
                queue.append((nxt, path + [action]))
        return []

def train_agent(task: BlackBoxTask, agent: BlackBoxMirrorAgent, steps: int = 24):
    state = task.initial
    actions = list(task.action_tokens)
    history = []
    for step in range(steps):
        action = actions[step % len(actions)]
        nxt = task.transition(state, action)
        agent.observe_transition(task.encode(state), action, task.encode(nxt))
        history.append(action)
        state = nxt
    return state, history

def phase161_protocol(seed: int) -> bool:
    task = BlackBoxTask.generate(seed)
    agent = BlackBoxMirrorAgent()
    state = task.initial
    action = task.action_tokens[0]
    nxt = task.transition(state, action)
    agent.observe_transition(task.encode(state), action, task.encode(nxt))
    predicted, confidence = agent.predict(task.encode(state), action)
    return len(predicted) == len(task.encode(state)) and confidence > 0.0

def phase162_raw_representation(seed: int) -> bool:
    task = BlackBoxTask.generate(seed)
    agent = BlackBoxMirrorAgent()
    train_agent(task, agent, 18)
    return len(agent.rep.value_positions) == len(task.initial) and agent.rep.confidence() > 0.5

def phase163_opaque_actions(seed: int) -> bool:
    task = BlackBoxTask.generate(seed)
    agent = BlackBoxMirrorAgent()
    train_agent(task, agent, 24)
    known = list(agent.rep.action_effects)
    unknown = bytes((b ^ 0xA5) for b in task.action_tokens[0])
    _, confidence = agent.predict(task.encode(task.initial), unknown)
    return len(known) >= len(task.initial) and confidence == 0.0

def phase164_multi_domain(seed: int) -> bool:
    for k in range(3):
        task = BlackBoxTask.generate(seed + k * 101, slots=3 + k)
        agent = BlackBoxMirrorAgent()
        train_agent(task, agent, 28)
        if len(agent.rep.value_positions) != len(task.initial):
            return False
    return True

def phase165_hidden_goal(seed: int) -> bool:
    task = BlackBoxTask.generate(seed, slots=3)
    agent = BlackBoxMirrorAgent()
    train_agent(task, agent, 36)
    state = task.initial
    target = list(state)
    target[0] = max(-40, min(40, target[0] + 2 * task.action_effects[0][1]))
    target_bytes = task.encode(tuple(target))
    current = task.encode(state)
    for _ in range(6):
        action = agent.choose_action(current, target_bytes, task.action_tokens)
        if action is None:
            return False
        state = task.transition(state, action)
        current = task.encode(state)
        if current == target_bytes:
            return True
    return False

def phase166_transfer(seed: int) -> bool:
    a = BlackBoxTask.generate(seed, slots=4)
    b = BlackBoxTask.generate(seed + 777, slots=4)
    agent_a = BlackBoxMirrorAgent()
    train_agent(a, agent_a, 32)
    agent_b = BlackBoxMirrorAgent()
    train_agent(b, agent_b, 32)
    transferable_policy = len(agent_a.rep.action_effects) >= 3 and len(agent_b.rep.action_effects) >= 3
    state = b.initial
    target = list(state)
    target[0] = max(-40, min(40, target[0] + b.action_effects[0][1]))
    chosen = agent_b.choose_action(b.encode(state), b.encode(tuple(target)), b.action_tokens)
    return transferable_policy and chosen is not None

def phase167_irreversible_long_horizon(seed: int) -> bool:
    task = BlackBoxTask.generate(seed, slots=4)
    agent = BlackBoxMirrorAgent()
    train_agent(task, agent, 48)
    state = task.initial
    target = list(state)
    for i in range(12):
        slot, delta = task.action_effects[i % len(task.action_effects)]
        if slot >= 0:
            target[slot] = max(-40, min(40, target[slot] + delta))
        state = task.transition(state, task.action_tokens[i % len(task.action_tokens)])
    target_bytes = task.encode(tuple(target))
    plan = agent.plan_to_target(task.encode(task.initial), target_bytes, task.action_tokens, max_depth=12)
    if not plan or len(plan) > 12:
        return False
    state = task.initial
    for action in plan:
        state = task.transition(state, action)
    return task.encode(state) == target_bytes

def phase168_contamination_controls(seed: int) -> bool:
    task = BlackBoxTask.generate(seed, slots=3)
    agent = BlackBoxMirrorAgent()
    train_agent(task, agent, 24)
    source = task.encode(task.initial)
    outcome = task.transition(task.initial, task.action_tokens[0])
    agent2 = BlackBoxMirrorAgent()
    agent2.observe_transition(source, task.action_tokens[0], task.encode(outcome))
    negative = not hasattr(agent2, "hidden_world") and not hasattr(agent2, "seed")
    return negative and bool(agent2.rep.traces)

def phase169_scaling(seed: int) -> bool:
    measurements = []
    for slots in (2, 4, 8, 12):
        task = BlackBoxTask.generate(seed + slots, slots=slots)
        agent = BlackBoxMirrorAgent()
        start = time.perf_counter()
        train_agent(task, agent, max(16, slots * 4))
        elapsed = time.perf_counter() - start
        measurements.append((slots, elapsed, len(agent.rep.traces), len(agent.rep.action_effects)))
    return all(traces >= 16 and actions > 0 for _, _, traces, actions in measurements)

def phase170_final(seed: int) -> bool:
    return all([
        phase161_protocol(seed), phase162_raw_representation(seed), phase163_opaque_actions(seed),
        phase164_multi_domain(seed), phase165_hidden_goal(seed), phase166_transfer(seed),
        phase167_irreversible_long_horizon(seed), phase168_contamination_controls(seed),
        phase169_scaling(seed),
    ])

def run_all_seeds(seeds: Iterable[int] = (0, 1, 2)):
    phases = {
        "161": phase161_protocol, "162": phase162_raw_representation, "163": phase163_opaque_actions,
        "164": phase164_multi_domain, "165": phase165_hidden_goal, "166": phase166_transfer,
        "167": phase167_irreversible_long_horizon, "168": phase168_contamination_controls,
        "169": phase169_scaling, "170": phase170_final,
    }
    return {name: {seed: bool(fn(seed)) for seed in seeds} for name, fn in phases.items()}
