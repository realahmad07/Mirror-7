from __future__ import annotations

from phase52_open_learning.mirror7_open_learner import OpenEndedLearner
from phase53_independent_eval.environment import HiddenRuleEnvironment
from phase53_independent_eval.evaluator import build_blind_suite


def run_protocol_task(spec):
    learner = OpenEndedLearner()
    env = HiddenRuleEnvironment(spec)
    msg = env.public_init()
    invalid = 0

    for _ in range(spec.max_steps):
        response = learner.handle(msg)
        action = response.get("action")
        result = env.step(action)
        invalid += int(not result["ok"])

        if result["terminal"]:
            return result["reward"] == 1, env.steps, invalid

        msg = result

    return False, env.steps, invalid


def test_phase53_unchanged_black_box_suite():
    scores = [run_protocol_task(spec) for spec in build_blind_suite()]
    assert len(scores) == 21
    assert all(solved for solved, _, _ in scores)
    assert all(invalid == 0 for _, _, invalid in scores)
