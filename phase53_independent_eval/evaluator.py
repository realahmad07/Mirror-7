from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from typing import Iterable

from .environment import HiddenRuleEnvironment, make_task

TRAIN_TASKS = (
    ("linear", 1), ("linear", 2), ("linear", 3),
    ("swap", 1), ("swap", 2), ("swap", 3),
    ("gate", 1), ("gate", 2), ("gate", 3),
)
HELD_OUT_TASKS = (
    ("linear", 11), ("linear", 17), ("linear", 23),
    ("swap", 11), ("swap", 17), ("swap", 23),
    ("conditional", 11), ("conditional", 17), ("conditional", 23),
    ("composition", 11), ("composition", 17), ("composition", 23),
)


@dataclass
class EpisodeScore:
    task_id: str
    family: str
    seed: int
    split: str
    solved: bool
    steps: int
    invalid_actions: int
    timed_out: bool


class ProtocolError(RuntimeError):
    pass


def _launch_agent(agent_cmd: list[str]):
    return subprocess.Popen(
        agent_cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, text=True, bufsize=1,
        env={**os.environ, "MIRROR7_PHASE53": "blind-black-box"},
    )


def _send(p, msg: dict) -> dict:
    assert p.stdin and p.stdout
    p.stdin.write(json.dumps(msg, sort_keys=True) + "\n")
    p.stdin.flush()
    line = p.stdout.readline()
    if not line:
        raise ProtocolError("agent exited or returned no response")
    try:
        return json.loads(line)
    except json.JSONDecodeError as exc:
        raise ProtocolError(f"agent emitted non-JSON: {line[:200]!r}") from exc


def build_blind_suite() -> list:
    return [make_task(f, s, "train") for f, s in TRAIN_TASKS] + [
        make_task(f, s, "heldout") for f, s in HELD_OUT_TASKS
    ]


def run_suite(agent_cmd: list[str]) -> list[EpisodeScore]:
    p = _launch_agent(agent_cmd)
    scores: list[EpisodeScore] = []
    try:
        for spec in build_blind_suite():
            env = HiddenRuleEnvironment(spec)
            invalid = 0
            try:
                msg = _send(p, env.public_init())
                for _ in range(spec.max_steps):
                    action = msg.get("action") if isinstance(msg, dict) else None
                    if not isinstance(action, str):
                        invalid += 1
                        action = "__invalid__"
                    result = env.step(action)
                    if not result["ok"]:
                        invalid += 1
                    if result["terminal"]:
                        scores.append(EpisodeScore(
                            spec.task_id, spec.family, spec.seed, spec.split,
                            result["reward"] == 1, result["steps"], invalid, False
                        ))
                        break
                    msg = _send(p, result)
                else:
                    scores.append(EpisodeScore(
                        spec.task_id, spec.family, spec.seed, spec.split,
                        False, env.steps, invalid, True
                    ))
            except (ProtocolError, BrokenPipeError, OSError):
                scores.append(EpisodeScore(
                    spec.task_id, spec.family, spec.seed, spec.split,
                    False, env.steps, invalid, False
                ))
                break
        return scores
    finally:
        try:
            p.kill()
            p.wait(timeout=1)
        except Exception:
            pass


def leakage_audit(tasks: Iterable) -> list[str]:
    problems = []
    for spec in tasks:
        public = HiddenRuleEnvironment(spec).public_init()
        required = {"type", "episode_id", "observation", "goal", "legal_actions", "step_limit"}
        if set(public) != required:
            problems.append(f"unexpected public keys: {spec.task_id}")
        public_json = json.dumps(public, sort_keys=True)
        if spec.family in public_json:
            problems.append(f"family leaked: {spec.task_id}")
        if spec.split in public_json:
            problems.append(f"split leaked: {spec.task_id}")
        for key in spec.hidden_parameters:
            if f'"{key}"' in public_json:
                problems.append(f"hidden key leaked: {spec.task_id}: {key}")
    return problems


def summarize(scores: list[EpisodeScore]) -> dict:
    total = len(scores)
    solved = sum(s.solved for s in scores)
    held = [s for s in scores if s.split == "heldout"]
    held_solved = sum(s.solved for s in held)
    invalid = sum(s.invalid_actions for s in scores)
    return {
        "episodes": total,
        "solved": solved,
        "solve_rate": solved / total if total else 0.0,
        "held_out_episodes": len(held),
        "held_out_solved": held_solved,
        "held_out_solve_rate": held_solved / len(held) if held else 0.0,
        "invalid_actions": invalid,
        "all_invalid_free": invalid == 0,
        "passed": total == 21 and solved == 21 and held_solved == 12 and invalid == 0,
    }
