import random

from .mirror7_phase34 import Goal, GoalPlanner, TransitionModel, train_model_from_trajectory


def train_linear(seed):
    rng = random.Random(seed)
    model = TransitionModel()
    rows = []
    for x in range(-3, 6):
        rows.append(({"x": x}, "inc", {"x": x + 1}))
        rows.append(({"x": x}, "dec", {"x": x - 1}))
    rng.shuffle(rows)
    train_model_from_trajectory(model, rows)
    return model


def train_two_axis(seed):
    rng = random.Random(seed)
    model = TransitionModel()
    rows = []
    for x in range(0, 5):
        rows.append(({"x": x, "y": 0}, "east", {"x": x + 1, "y": 0}))
        rows.append(({"x": x, "y": 1}, "east", {"x": x + 1, "y": 1}))
    for y in range(0, 4):
        rows.append(({"x": 0, "y": y}, "north", {"x": 0, "y": y + 1}))
        rows.append(({"x": 1, "y": y}, "north", {"x": 1, "y": y + 1}))
    rows += [
        ({"x": 2, "y": 0}, "reset", {"x": 0, "y": 0}),
        ({"x": 4, "y": 3}, "reset", {"x": 0, "y": 0}),
    ]
    rng.shuffle(rows)
    train_model_from_trajectory(model, rows)
    return model


def train_gate(seed):
    rng = random.Random(seed)
    model = TransitionModel()
    rows = []
    for x in range(0, 5):
        rows.append(({"x": x, "gate": 0}, "step", {"x": x + 1, "gate": 0}))
        rows.append(({"x": x, "gate": 1}, "step", {"x": x + 1, "gate": 1}))
    rows += [
        ({"x": 2, "gate": 0}, "open", {"x": 2, "gate": 1}),
        ({"x": 2, "gate": 1}, "close", {"x": 2, "gate": 0}),
        ({"x": 0, "gate": 1}, "finish", {"x": 0, "gate": 2}),
        ({"x": 2, "gate": 1}, "finish", {"x": 2, "gate": 2}),
    ]
    rng.shuffle(rows)
    train_model_from_trajectory(model, rows)
    return model


def assert_plan(model, start, goal, max_depth=12):
    planner = GoalPlanner(model)
    plan = planner.plan(start, goal, max_depth=max_depth)
    assert plan is not None, planner.trace
    assert planner.validate_plan(plan, start, goal)
    return plan


def test_progressive_level_1():
    for seed in range(3):
        plan = assert_plan(train_linear(seed), {"x": 0}, Goal.from_mapping({"x": 4}), max_depth=6)
        assert plan.actions == ("inc", "inc", "inc", "inc")


def test_progressive_level_2():
    for seed in range(3):
        plan = assert_plan(train_two_axis(seed), {"x": 1, "y": 0}, Goal.from_mapping({"x": 4, "y": 3}), max_depth=8)
        assert len(plan.actions) == 6
        assert plan.actions.count("east") == 3
        assert plan.actions.count("north") == 3


def test_progressive_level_3():
    for seed in range(3):
        plan = assert_plan(train_gate(seed), {"x": 0, "gate": 0}, Goal.from_mapping({"x": 4, "gate": 2}), max_depth=10)
        assert "open" in plan.actions
        assert "step" in plan.actions
        assert plan.actions.count("step") >= 2


def test_held_out_cases():
    cases = [
        (train_linear(17), {"x": -2}, {"x": 5}, 9),
        (train_two_axis(19), {"x": 2, "y": 0}, {"x": 4, "y": 3}, 8),
        (train_gate(23), {"x": 1, "gate": 0}, {"x": 4, "gate": 2}, 10),
    ]
    for model, start, target, depth in cases:
        plan = assert_plan(model, start, Goal.from_mapping(target), depth)
        assert len(plan.actions) > 0


def test_adversarial_unsatisfiable_goal():
    model = train_linear(31)
    planner = GoalPlanner(model)
    plan = planner.plan({"x": 0}, Goal.from_mapping({"z": 99}), max_depth=5, max_nodes=100)
    assert plan is None
    assert planner.trace[-1]["event"] == "failure"


def test_adversarial_cycle_control():
    model = TransitionModel()
    model.observe({"x": 0}, "inc", {"x": 1})
    model.observe({"x": 1}, "dec", {"x": 0})
    planner = GoalPlanner(model)
    plan = planner.plan({"x": 0}, Goal.from_mapping({"x": 2}), max_depth=6, max_nodes=50)
    assert plan is None


def test_adversarial_unknown_action_rejected():
    model = train_linear(41)
    planner = GoalPlanner(model)
    plan = planner.plan({"x": 0}, Goal.from_mapping({"x": 2, "y": 1}), max_depth=6)
    assert plan is None
    assert model.predict((('x', 0),), "ghost") is None


def test_exact_evidence_precedes_generalization():
    model = TransitionModel()
    model.observe({"x": 0}, "jump", {"x": 5})
    model.observe({"x": 1}, "jump", {"x": 6})
    model.observe({"x": 2}, "jump", {"x": 7})
    model.observe({"x": 3}, "jump", {"x": 8})
    assert model.predict((('x', 0),), "jump") == (('x', 5),)
    assert model.predict((('x', 4),), "jump") is None
