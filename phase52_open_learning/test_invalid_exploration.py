from phase52_open_learning.mirror7_open_learner import OpenEndedLearner


def test_failed_action_is_not_retried_at_same_state():
    learner = OpenEndedLearner()

    first = learner.handle({
        "type": "episode_start",
        "episode_id": "opaque",
        "observation": {"x": 0},
        "goal": {"x": 1},
        "legal_actions": ["bad", "good"],
        "step_limit": 8,
    })
    assert first["action"] == "bad"

    second = learner.handle({
        "type": "transition",
        "ok": False,
        "observation": {"x": 0},
        "changed": {},
        "terminal": False,
        "reward": 0,
        "steps": 1,
    })

    assert second["action"] == "good"


def test_failed_action_does_not_override_successful_model_at_other_states():
    learner = OpenEndedLearner()

    learner.handle({
        "type": "episode_start",
        "episode_id": "opaque",
        "observation": {"x": 0},
        "goal": {"x": 1},
        "legal_actions": ["advance", "other"],
        "step_limit": 8,
    })

    next_action = learner.handle({
        "type": "transition",
        "ok": True,
        "observation": {"x": 1},
        "changed": {"x": (0, 1)},
        "terminal": False,
        "reward": 0,
        "steps": 1,
    })
    assert next_action["action"] == "other"

    # The next transition is now for the action actually returned above.
    retry = learner.handle({
        "type": "transition",
        "ok": False,
        "observation": {"x": 1},
        "changed": {},
        "terminal": False,
        "reward": 0,
        "steps": 2,
    })
    assert retry["action"] == "advance"

    assert next_action["action"] == "other"


def test_failure_memory_is_state_local():
    learner = OpenEndedLearner()

    learner.handle({
        "type": "episode_start",
        "episode_id": "opaque",
        "observation": {"x": 0},
        "goal": {"x": 1},
        "legal_actions": ["toggle", "other"],
        "step_limit": 8,
    })

    first = learner.handle({
        "type": "transition",
        "ok": False,
        "observation": {"x": 0},
        "changed": {},
        "terminal": False,
        "reward": 0,
        "steps": 1,
    })
    assert first["action"] == "other"

    # Move to a different state through the other action.
    second = learner.handle({
        "type": "transition",
        "ok": True,
        "observation": {"x": 1},
        "changed": {"x": (0, 1)},
        "terminal": False,
        "reward": 0,
        "steps": 2,
    })

    # toggle is now allowed to be explored again because its failure was
    # tied specifically to x=0.
    assert second["action"] == "toggle"
