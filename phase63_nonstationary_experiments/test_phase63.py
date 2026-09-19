import random
import pytest

from .mirror7_phase63 import NonstationaryAgent, RegimeAwareModel


class DriftEnv:
    def __init__(self, seed, switch=5, mode=0):
        acts = ["drive", "calibrate", "idle", "reverse"]
        random.Random(seed).shuffle(acts)
        self.actions = tuple(acts)
        self.mode = mode
        self.switch = switch
        self.t = 0
        self.state = (0,)

    def reset(self, s=(0,)):
        self.t = 0
        self.state = tuple(s)
        return self.state

    def legal_actions(self, state):
        return self.actions

    def step(self, action):
        x = self.state[0]
        if action == "drive":
            x += 1 if self.t < self.switch else 2
        elif action == "reverse":
            x -= 1
        self.t += 1
        self.state = (x,)
        return self.state


class RegimeEnv:
    def __init__(self, mode):
        self.mode = mode
        self.state = (0,)
        self.actions = ("drive", "probe", "idle")

    def reset(self, s=(0,)):
        self.state = tuple(s)
        return self.state

    def legal_actions(self, state):
        return self.actions

    def step(self, action):
        x = self.state[0]
        if action == "drive":
            x += 1 if self.mode == 0 else 3
        elif action == "probe":
            x += 0 if self.mode == 0 else 10
        self.state = (x,)
        return self.state


def regime_demo():
    return (
        [((0,), "drive", (1,)), ((1,), "drive", (2,)), ((0,), "probe", (0,))],
        [((0,), "drive", (3,)), ((3,), "drive", (6,)), ((0,), "probe", (10,))],
    )


def test_progressive_nonstationary_3x3():
    for switch in (3, 5, 8):
        for seed in range(3):
            env = DriftEnv(100 + seed, switch=switch)
            agent = NonstationaryAgent(experiment_budget=4, drift_threshold=2)
            stable = [((i,), "drive", (i + 1,)) for i in range(31) for _ in range(2)]
            changed = [((i,), "drive", (i + 2,)) for i in range(31) for _ in range(2)]
            agent.learn_regime("stable", stable)
            agent.learn_regime("changed", changed)
            result = agent.run(
                env,
                start=(0,),
                goal=(switch + 12,),
                max_steps=switch + 8,
            )
            assert result["invalid_actions"] == 0 and result["solved"]
            assert result["experiments"] >= 1 and agent.active_regime == "changed"


def test_experiment_selection_max_disagreement():
    agent = NonstationaryAgent()
    r0, r1 = regime_demo()
    agent.learn_regime(0, r0 * 2)
    agent.learn_regime(1, r1 * 2)
    experiment = agent.design_experiment(
        (0,), ("drive", "probe", "idle"), (10,)
    )
    assert experiment is not None
    assert experiment.action == "probe"
    assert experiment.expected_outcomes == 2


def test_drift_detection_requires_repeated_mismatch():
    agent = NonstationaryAgent(drift_threshold=2)
    agent.learn_regime(
        0,
        [
            ((0,), "drive", (1,)),
            ((0,), "drive", (1,)),
            ((1,), "drive", (2,)),
            ((1,), "drive", (2,)),
        ],
    )
    agent.reset((0,), (5,))
    assert agent.model.update((0,), "drive", (9,)) is True
    assert not agent.model.stale
    assert agent.model.update((0,), "drive", (9,)) is True
    assert agent.model.stale and agent.model.drift_events == 1


def test_no_false_drift_on_consistent_regime():
    agent = NonstationaryAgent(drift_threshold=2)
    agent.learn_regime(
        0,
        [
            ((0,), "drive", (1,)),
            ((0,), "drive", (1,)),
            ((1,), "drive", (2,)),
            ((1,), "drive", (2,)),
        ],
    )
    for i in range(5):
        agent.model.update((i,), "drive", (i + 1,))
    assert not agent.model.stale


def test_regime_identification_after_experiment():
    agent = NonstationaryAgent(experiment_budget=3)
    r0, r1 = regime_demo()
    agent.learn_regime(0, r0 * 2)
    agent.learn_regime(1, r1 * 2)
    agent.reset((0,), (6,))
    agent.observe("probe", (10,), legal=("drive", "probe", "idle"))
    assert agent.active_regime == 1


def test_goal_reuse_after_drift():
    env = RegimeEnv(1)
    agent = NonstationaryAgent(experiment_budget=3, drift_threshold=1)
    r0, r1 = regime_demo()
    agent.learn_regime(0, r0 * 2)
    agent.learn_regime(1, r1 * 2)
    result = agent.run(
        env, start=(0,), goal=(31,), max_steps=8
    )
    assert result["solved"]
    assert result["invalid_actions"] == 0
    assert result["experiments"] >= 1


def test_experiment_budget_is_bounded():
    agent = NonstationaryAgent(experiment_budget=2)
    r0, r1 = regime_demo()
    agent.learn_regime(0, r0 * 2)
    agent.learn_regime(1, r1 * 2)
    agent.reset((0,), (99,))
    for _ in range(5):
        try:
            agent.decide((0,), (99,), ("drive", "probe", "idle"))
        except RuntimeError:
            pass
    assert agent._experiments <= 2


def test_phase62_handoff_contract():
    from phase62_active_information.mirror7_phase62 import PartialObservationAgent

    agent = PartialObservationAgent()
    agent.reset((0, None), (3, 1))
    assert agent.current == (0, None)
    assert agent.goal == (3, 1)


def test_malformed():
    with pytest.raises(ValueError):
        NonstationaryAgent(experiment_budget=0)
    with pytest.raises(ValueError):
        RegimeAwareModel(min_support=1)
    with pytest.raises(ValueError):
        NonstationaryAgent().reset([], (0,))
