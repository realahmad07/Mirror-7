import random
import pytest

from .mirror7_phase61 import PredictiveClosedLoopAgent


class HiddenEnv:
    def __init__(self, seed, family):
        actions = ["a", "b", "c", "d", "e"]
        random.Random(seed).shuffle(actions)
        self.actions = actions[:4]
        if family == 0:
            self.ops = {
                self.actions[0]: ("inc", 0),
                self.actions[1]: ("toggle", 1),
                self.actions[2]: ("noop", 0),
                self.actions[3]: ("dec", 0),
            }
        elif family == 1:
            self.ops = {
                self.actions[0]: ("inc", 0),
                self.actions[1]: ("toggle", 1),
                self.actions[2]: ("dec", 0),
                self.actions[3]: ("noop", 0),
            }
        else:
            self.ops = {
                self.actions[0]: ("inc", 0),
                self.actions[1]: ("toggle", 1),
                self.actions[2]: ("noop", 0),
                self.actions[3]: ("dec", 0),
            }
        self.state = (0, 0)

    def reset(self, s=(0, 0)):
        self.state = s

    def legal_actions(self, state):
        return tuple(self.actions)

    def step(self, action):
        op, _ = self.ops[action]
        x, y = self.state
        if op == "inc":
            x += 1
        elif op == "dec":
            x = max(0, x - 1)
        elif op == "toggle":
            y ^= 1
        self.state = (x, y)
        return self.state


class GateEnv:
    def __init__(self, seed):
        actions = ["open", "move", "bad", "idle"]
        random.Random(seed).shuffle(actions)
        self.actions = actions
        self.state = (0, 0)
        self.ops = {
            "open": lambda s: (s[0], 1),
            "move": lambda s: (s[0] + 1, s[1]) if s[1] else s,
            "bad": lambda s: (max(0, s[0] - 1), s[1]),
            "idle": lambda s: s,
        }

    def reset(self):
        self.state = (0, 0)

    def legal_actions(self, state):
        return tuple(self.actions)

    def step(self, action):
        self.state = self.ops[action](self.state)
        return self.state


def test_progressive_goal_pursuit_3x3():
    for family in range(3):
        for seed in range(3):
            env = HiddenEnv(100 + seed, family)
            agent = PredictiveClosedLoopAgent(
                max_plan_depth=6, max_exploration=40
            )
            result = agent.run(
                env, start=(0, 0), goal=(5, 1), max_steps=40
            )
            assert (
                result.solved
                and result.invalid_actions == 0
                and result.steps <= 40
            )


def test_heldout_opaque_environment():
    env = HiddenEnv(9001, 2)
    agent = PredictiveClosedLoopAgent(max_plan_depth=8, max_exploration=48)
    result = agent.run(
        env, start=(0, 0), goal=(7, 1), max_steps=48
    )
    assert result.solved and result.invalid_actions == 0


def test_gate_requires_multi_step_composition():
    env = GateEnv(12)
    agent = PredictiveClosedLoopAgent(
        max_plan_depth=8, max_exploration=40
    )
    result = agent.run(
        env, start=(0, 0), goal=(4, 1), max_steps=40
    )
    assert result.solved and result.steps <= 40


def test_discrepancy_replans_and_clears_plan():
    env = HiddenEnv(33, 0)
    agent = PredictiveClosedLoopAgent(max_plan_depth=6)
    action = env.actions[0]

    agent.learn_demonstration(
        [
            ((0, 0), action, (1, 0)),
            ((1, 0), action, (2, 0)),
        ],
        [action, action],
    )
    agent.reset((0, 0), (3, 0))
    decision = agent.decide(
        (0, 0), (3, 0), env.legal_actions((0, 0))
    )
    assert decision.action == action

    env.ops[action] = (0, 1)
    agent.observe(
        action,
        env.step(action),
        legal_actions=env.legal_actions((0, 0)),
    )
    assert agent.discrepancies == 1 and agent.replans >= 1


def test_unknown_actions_explored_once_before_repetition():
    env = GateEnv(5)
    agent = PredictiveClosedLoopAgent(max_exploration=20)
    agent.reset((0, 0), (10, 10))
    legal = env.legal_actions(env.state)

    seen = []
    for _ in range(len(legal)):
        state = env.state
        decision = agent.decide(state, agent.goal, legal)
        seen.append((state, decision.action))
        agent.observe(
            decision.action,
            env.step(decision.action),
            legal_actions=legal,
        )

    assert len(seen) == len(set(seen))


def test_phase58_59_policy_memory_integration():
    agent = PredictiveClosedLoopAgent()
    traces = [
        ("open", "move", "move"),
        ("open", "move", "move"),
    ]

    for trace in traces:
        state = (0, 0)
        transitions = []
        for action in trace:
            next_state = (
                (state[0], 1)
                if action == "open"
                else (state[0] + 1, state[1])
            )
            transitions.append((state, action, next_state))
            state = next_state
        agent.learn_demonstration(transitions, trace)

    assert agent.hierarchy is not None
    assert len(agent.hierarchy.levels) >= 1
    assert agent.predictor is not None


def test_long_horizon_32_steps_from_learned_delta():
    env = HiddenEnv(71, 0)
    agent = PredictiveClosedLoopAgent(
        max_plan_depth=8, max_exploration=20
    )
    advance, toggle = env.actions[0], env.actions[1]
    demos = []

    for start in (0, 10):
        demos.append(
            [
                ((start, 0), advance, (start + 1, 0)),
                ((start + 1, 0), advance, (start + 2, 0)),
                ((start + 2, 0), toggle, (start + 2, 1)),
            ]
        )

    for episode in demos:
        agent.learn_demonstration(
            episode, [advance, advance, toggle]
        )

    result = agent.run(
        env, start=(20, 0), goal=(52, 1), max_steps=40
    )
    assert result.solved and result.steps <= 40


def test_no_invalid_action_and_fail_closed():
    agent = PredictiveClosedLoopAgent()
    agent.reset((0, 0), (1, 0))

    with pytest.raises(ValueError):
        agent.decide((0, 0), (1, 0), [])

    env = HiddenEnv(3, 0)
    result = agent.run(
        env, start=(0, 0), goal=(2, 0), max_steps=20
    )
    assert result.invalid_actions == 0


def test_malformed():
    with pytest.raises(ValueError):
        PredictiveClosedLoopAgent(max_plan_depth=0)
