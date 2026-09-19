from .mirror7_open_learner import OpenEndedLearner, goal_distance


def test_linear_discovery():
    learner = OpenEndedLearner()
    first = learner.handle({
        "type": "episode_start",
        "episode_id": "opaque",
        "observation": {"z": 0},
        "goal": {"z": 3},
        "legal_actions": ["a0", "a1"],
        "step_limit": 10,
    })
    assert first["action"] == "a0"
    second = learner.handle({
        "type": "transition",
        "ok": True,
        "observation": {"z": 1},
        "changed": {"z": (0, 1)},
        "terminal": False,
        "reward": 0,
        "steps": 1,
    })
    assert second["action"] == "a0"
    third = learner.handle({
        "type": "transition",
        "ok": True,
        "observation": {"z": 2},
        "changed": {"z": (1, 2)},
        "terminal": False,
        "reward": 0,
        "steps": 2,
    })
    assert third["action"] == "a0"


def test_switches_after_predicted_overshoot():
    learner = OpenEndedLearner()
    learner.handle({
        "type": "episode_start",
        "episode_id": "opaque",
        "observation": {"g": 0, "v": 0},
        "goal": {"g": 1, "v": 1},
        "legal_actions": ["toggle", "work", "noop"],
        "step_limit": 10,
    })
    nxt = learner.handle({
        "type": "transition",
        "ok": True,
        "observation": {"g": 1, "v": 0},
        "changed": {"g": (0, 1)},
        "terminal": False,
        "reward": 0,
        "steps": 1,
    })
    assert nxt["action"] == "work"


def test_composition_switches_to_second_action():
    learner = OpenEndedLearner()
    learner.handle({
        "type": "episode_start",
        "episode_id": "opaque",
        "observation": {"x": 0, "y": 0},
        "goal": {"x": 2, "y": 1},
        "legal_actions": ["ax", "ay", "noop"],
        "step_limit": 10,
    })
    msg = learner.handle({
        "type": "transition", "ok": True,
        "observation": {"x": 1, "y": 0},
        "changed": {"x": (0, 1)},
        "terminal": False, "reward": 0, "steps": 1,
    })
    assert msg["action"] == "ax"
    msg = learner.handle({
        "type": "transition", "ok": True,
        "observation": {"x": 2, "y": 0},
        "changed": {"x": (1, 2)},
        "terminal": False, "reward": 0, "steps": 2,
    })
    assert msg["action"] == "ay"
