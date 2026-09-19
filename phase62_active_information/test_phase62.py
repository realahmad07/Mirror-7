import random
import pytest
from .mirror7_phase62 import PartialObservationAgent


class HiddenSensorEnv:
    def __init__(self, seed, mode=0, sensors=True):
        acts = ["inc", "dec", "toggle", "sense_a", "sense_b"]
        random.Random(seed).shuffle(acts)
        self.actions = tuple(acts)
        self.mode = mode
        self.sensors = sensors
        self.hidden = (mode % 2, (mode // 2) % 2)
        self.state = (0, *self.hidden)
        self.reveal = set()

    def reset(self):
        self.state = (0, *self.hidden)
        self.reveal = set()
        return self._obs()

    def _obs(self):
        x, a, b = self.state
        return (x, a if 1 in self.reveal else None, b if 2 in self.reveal else None)

    def legal_actions(self, obs):
        return self.actions if self.sensors else tuple(a for a in self.actions if not a.startswith("sense"))

    def step(self, action):
        x, a, b = self.state
        if action == "inc":
            x += 1
        elif action == "dec":
            x = max(0, x - 1) if a == 1 else x
        elif action == "toggle":
            x += b
        elif action == "sense_a":
            self.reveal.add(1)
        elif action == "sense_b":
            self.reveal.add(2)
        else:
            raise ValueError(action)
        self.state = (x, a, b)
        return self._obs()


def test_progressive_partial_observability_3x3():
    for family in range(3):
        for seed in range(3):
            env = HiddenSensorEnv(100 + seed, family)
            agent = PartialObservationAgent(max_plan_depth=8, max_exploration=20)
            result = agent.run(env, goal=(3, env.hidden[0], env.hidden[1]), max_steps=20)
            assert result.solved
            assert result.invalid_actions == 0
            assert result.information_steps >= 1


def test_heldout_opaque_hidden_configuration():
    env = HiddenSensorEnv(9001, mode=3)
    agent = PartialObservationAgent(max_plan_depth=10, max_exploration=30)
    result = agent.run(env, goal=(4, 1, 1), max_steps=30)
    assert result.solved and result.invalid_actions == 0


def test_active_information_selection_prefers_relevant_sensor():
    env = HiddenSensorEnv(77, mode=1)
    agent = PartialObservationAgent()
    assert agent.run(env, goal=(0, 1, 0), max_steps=12).information_steps >= 1
    agent.reset((0, 1, None), (3, 1, 1))
    legal = env.legal_actions(agent.current)
    decision = agent.decide(agent.current, agent.goal, legal)
    assert decision.action == "sense_b"


def test_sensor_dropout_uses_remaining_information_path():
    env = HiddenSensorEnv(33, mode=2, sensors=True)
    agent = PartialObservationAgent(max_plan_depth=8, max_exploration=20)
    first = agent.run(env, goal=(2, 0, 1), max_steps=20)
    assert first.solved

    env2 = HiddenSensorEnv(34, mode=2, sensors=False)
    agent2 = PartialObservationAgent(max_plan_depth=8, max_exploration=10)
    result = agent2.run(env2, goal=(1, 0, 1), max_steps=10)
    assert result.invalid_actions == 0
    assert not result.solved


def test_contradictory_model_evidence_forces_replan_counter():
    agent = PartialObservationAgent()
    agent.reset((0, 1, 1), (2, 1, 1))
    agent.model.update((0, 1, 1), "inc", (1, 1, 1))
    agent.model.update((0, 1, 1), "inc", (1, 1, 1))
    agent.observe("inc", (9, 1, 1), legal_actions=["inc"])
    assert agent._contradictions == 1 and agent._replans >= 1


def test_unknown_actions_are_not_repeated_at_same_partial_state():
    env = HiddenSensorEnv(66, mode=3)
    agent = PartialObservationAgent(max_exploration=10)
    agent.reset(env.reset(), (1, 1, 1))
    legal = env.legal_actions(agent.current)
    seen = set()
    for _ in range(min(3, len(legal))):
        state = agent.current
        d = agent.decide(state, agent.goal, legal)
        pair = (state, d.action)
        assert pair not in seen
        seen.add(pair)
        agent.observe(d.action, env.step(d.action), legal_actions=legal)


def test_full_state_model_learns_from_revealed_transitions():
    env = HiddenSensorEnv(88, mode=3)
    agent = PartialObservationAgent()
    agent.run(env, goal=(0, 1, 1), max_steps=12)
    env.reveal.update((1, 2))
    agent.current = env._obs()
    nxt = env.step("inc")
    agent.observe("inc", nxt, legal_actions=env.legal_actions(agent.current))
    agent.current = (0, 1, 1)
    agent.observe("inc", nxt, legal_actions=env.legal_actions(agent.current))
    assert agent.model.predict((0, 1, 1), "inc") == (1, 1, 1)


def test_phase61_handoff_contract():
    env = HiddenSensorEnv(101, mode=3)
    agent = PartialObservationAgent()
    agent.run(env, goal=(2, 1, 1), max_steps=20)
    state = agent.revealed_state()
    assert state == env.state
    assert all(v is not None for v in state)


def test_malformed():
    with pytest.raises(ValueError):
        PartialObservationAgent(max_plan_depth=0)
    with pytest.raises(ValueError):
        PartialObservationAgent().reset([], (0,))
    with pytest.raises(ValueError):
        PartialObservationAgent().decide((0,), (0, 1), ["a"])
    with pytest.raises(ValueError):
        PartialObservationAgent().run(HiddenSensorEnv(1), goal=(0, 0, 0), max_steps=0)
