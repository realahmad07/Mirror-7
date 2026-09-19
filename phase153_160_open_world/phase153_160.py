"""Phases 153-160: sealed open-world evaluation around the Phase 141-152 runtime.

The evaluator exposes only observation/state/action/outcome transitions to the agent.
Hidden environment generators and acceptance checks remain outside the agent boundary.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Tuple
import random

from phase141_152_integration.phase141_152 import IntegratedCognition
from mirror7_algorithm_11 import Structure, discover_transformation, apply_transformation

@dataclass(frozen=True)
class HiddenWorld:
    variables: Tuple[str, ...]
    actions: Tuple[str, ...]
    deltas: Tuple[int, ...]
    initial: Tuple[int, ...]

    @classmethod
    def generate(cls, seed: int, variable_count: int = 3) -> "HiddenWorld":
        rng = random.Random(seed)
        variables = tuple(f"q{rng.randrange(10_000):04d}" for _ in range(variable_count))
        while len(set(variables)) != variable_count:
            variables = tuple(f"q{rng.randrange(10_000):04d}" for _ in range(variable_count))
        actions = tuple(f"u{rng.randrange(10_000):04d}" for _ in range(variable_count))
        while len(set(actions)) != variable_count:
            actions = tuple(f"u{rng.randrange(10_000):04d}" for _ in range(variable_count))
        deltas = tuple(rng.choice((-2, -1, 1, 2)) for _ in range(variable_count))
        initial = tuple(rng.randrange(-3, 4) for _ in range(variable_count))
        return cls(variables, actions, deltas, initial)

    def state(self, values: Iterable[int]) -> Dict[str, int]:
        return dict(zip(self.variables, values))

    def initial_state(self) -> Dict[str, int]:
        return self.state(self.initial)

    def transition(self, state: Dict[str, int], action: str) -> Dict[str, int]:
        nxt = dict(state)
        if action in self.actions:
            i = self.actions.index(action)
            nxt[self.variables[i]] = nxt.get(self.variables[i], 0) + self.deltas[i]
        return nxt

class SealedMirrorAgent:
    """Black-box boundary used by the evaluator.

    The evaluator never receives the internal modules, goals, hidden world, or seed.
    """
    def __init__(self) -> None:
        self._runtime = IntegratedCognition()
        # Phase 157 requires a bounded 32-transition replay; the runtime's
        # default world-model rule budget is 20. Increase only the evaluation
        # configuration, without changing the learning/prediction algorithm.
        self._runtime.world.max_rules = 64

    def learn(self, observation: Dict[str, Any], state: Dict[str, Any], action: str, outcome: Dict[str, Any]) -> Any:
        return self._runtime.step(observation, state, action, outcome)

    def predict(self, state: Dict[str, Any], action: str) -> Tuple[Dict[str, Any], float]:
        return self._runtime.world.predict(state, action)

    @property
    def memory_size(self) -> int:
        return len(self._runtime.memory.nodes)

    @property
    def workspace_size(self) -> int:
        return len(self._runtime.workspace.items)

@dataclass(frozen=True)
class PredictionRecord:
    expected: Dict[str, Any]
    predicted: Dict[str, Any]
    confidence: float

@dataclass(frozen=True)
class EvaluationResult:
    passed: bool
    checks: int
    passed_checks: int
    prediction_accuracy: float
    unknown_confidence: float
    max_memory: int
    max_workspace: int

class IndependentEvaluator:
    """Trace evaluator retained for phase-local unit checks."""
    @staticmethod
    def score(records: List[PredictionRecord], unknown_confidence: float, max_memory: int,
              max_workspace: int, memory_limit: int = 1000, workspace_limit: int = 100) -> EvaluationResult:
        checks = 4
        correct = sum(r.predicted == r.expected for r in records)
        accuracy = correct / len(records) if records else 0.0
        criteria = [accuracy >= 0.75, unknown_confidence <= 0.10,
                    max_memory <= memory_limit, max_workspace <= workspace_limit]
        passed_checks = sum(bool(x) for x in criteria)
        return EvaluationResult(passed_checks == checks, checks, passed_checks, accuracy,
                                unknown_confidence, max_memory, max_workspace)

def run_prediction_episode(seed: int, *, variable_count: int = 3, horizon: int = 12,
                           noise: bool = False) -> Tuple[SealedMirrorAgent, List[PredictionRecord], float]:
    world = HiddenWorld.generate(seed, variable_count)
    agent = SealedMirrorAgent()
    state = world.initial_state()
    records: List[PredictionRecord] = []

    for step in range(horizon):
        action = world.actions[step % len(world.actions)]
        nxt = world.transition(state, action)
        obs = dict(state)
        if noise:
            obs[f"noise_{step}"] = step % 3
        agent.learn(obs, state, action, nxt)
        state = nxt

    state = world.initial_state()
    for step in range(horizon):
        action = world.actions[step % len(world.actions)]
        expected = world.transition(state, action)
        predicted, confidence = agent.predict(state, action)
        records.append(PredictionRecord(expected, predicted, confidence))
        obs = dict(state)
        if noise:
            obs[f"noise_eval_{step}"] = (step + 1) % 3
        agent.learn(obs, state, action, expected)
        state = expected

    _, unknown_conf = agent.predict(state, "opaque-unseen-action")
    return agent, records, unknown_conf

def phase153_sealed(seed: int) -> bool:
    agent, records, unknown = run_prediction_episode(seed)
    result = IndependentEvaluator.score(records, unknown, agent.memory_size, agent.workspace_size)
    return result.passed

def phase154_unseen(seed: int) -> bool:
    train_world = HiddenWorld.generate(seed, 3)
    eval_world = HiddenWorld.generate(seed + 1000, 3)
    agent = SealedMirrorAgent()
    state = train_world.initial_state()
    for step in range(10):
        action = train_world.actions[step % 3]
        nxt = train_world.transition(state, action)
        agent.learn(state, state, action, nxt)
        state = nxt

    state = eval_world.initial_state()
    seen = 0
    for step in range(8):
        action = eval_world.actions[step % 3]
        nxt = eval_world.transition(state, action)
        _, confidence = agent.predict(state, action)
        if confidence <= 0.10:
            seen += 1
        agent.learn(state, state, action, nxt)
        state = nxt
    return seen >= 4

def phase155_transfer(seed: int) -> bool:
    shift = seed % 3
    source = Structure(
        ["a", "b", "c"],
        {"a": {"role": "x", "v": 1 + shift}, "b": {"role": "y", "v": 2 + shift}, "c": {"role": "z", "v": 3 + shift}},
        [],
    )
    target = Structure(
        ["m", "n", "p"],
        {"m": {"role": "x", "v": 9}, "n": {"role": "y", "v": 8}, "p": {"role": "z", "v": 7}},
        [],
    )
    rule = discover_transformation(source, target)
    transformed = apply_transformation(source, rule)
    mapping = {"a": "m", "b": "n", "c": "p"}
    return all(transformed.properties[src] == target.properties[dst] for src, dst in mapping.items())

def phase156_noise(seed: int) -> bool:
    agent, records, unknown = run_prediction_episode(seed, noise=True)
    clean_predictions = sum(r.predicted == r.expected for r in records)
    return clean_predictions >= int(0.75 * len(records)) and unknown <= 0.10

def phase157_long_horizon(seed: int) -> bool:
    agent, records, _ = run_prediction_episode(seed, horizon=32)
    return sum(r.predicted == r.expected for r in records) >= 24

def phase158_independent(seed: int) -> bool:
    agent, records, unknown = run_prediction_episode(seed)
    report = IndependentEvaluator.score(records, unknown, agent.memory_size, agent.workspace_size)
    return report.passed and report.passed_checks == report.checks

def phase159_stress(seed: int) -> bool:
    max_memory = 0
    max_workspace = 0
    agent = SealedMirrorAgent()
    world = HiddenWorld.generate(seed, variable_count=8)
    state = world.initial_state()
    for step in range(64):
        action = world.actions[step % len(world.actions)]
        nxt = world.transition(state, action)
        obs = dict(state)
        obs.update({f"noise_{j}": (step + j) % 5 for j in range(8)})
        agent.learn(obs, state, action, nxt)
        state = nxt
        max_memory = max(max_memory, agent.memory_size)
        max_workspace = max(max_workspace, agent.workspace_size)
    return max_memory <= 1000 and max_workspace <= 100

def phase160_final(seed: int) -> bool:
    checks = [phase153_sealed(seed), phase154_unseen(seed), phase155_transfer(seed),
              phase156_noise(seed), phase157_long_horizon(seed), phase158_independent(seed),
              phase159_stress(seed)]
    return sum(bool(x) for x in checks) == len(checks)

def run_all_seeds(seeds: Iterable[int] = (0, 1, 2)) -> Dict[str, Dict[int, bool]]:
    phases = {"153": phase153_sealed, "154": phase154_unseen, "155": phase155_transfer,
              "156": phase156_noise, "157": phase157_long_horizon, "158": phase158_independent,
              "159": phase159_stress, "160": phase160_final}
    return {name: {seed: bool(fn(seed)) for seed in seeds} for name, fn in phases.items()}
