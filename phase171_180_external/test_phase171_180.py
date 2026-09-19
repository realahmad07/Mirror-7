from __future__ import annotations
import json, subprocess, sys
import pytest
from phase171_180_external.protocol import (
    ExternalEpisode, build_pack, make_sealed_view,
    sanitize_for_agent, task_pack_fingerprint,
)

def make_pack(seed: int):
    return build_pack(
        f"external-{seed}", "1",
        [ExternalEpisode(
            "episode-0",
            ("obs-0","obs-1"),
            ("action-a","action-b"),
            ("out-0","out-1"),
            "goal-x",
            ("action-a","action-b"),
            {"domain":"unseen"},
        )],
        public_metadata={"source":"external-provider","seed":seed},
        evaluator_id="evaluator-only-id",
    )

@pytest.mark.parametrize("seed", [0,1,2])
def test_171_external_pack_is_well_formed(seed):
    pack = make_pack(seed)
    assert len(pack.episodes) == 1
    assert pack.public_view()["episodes"][0]["episode_id"] == "episode-0"
    assert len(task_pack_fingerprint(pack)) == 64

def test_172_agent_view_has_no_evaluator_secret():
    pack = make_pack(7)
    public = pack.public_view()
    agent_payload = sanitize_for_agent(public)
    text = json.dumps(agent_payload)
    assert "evaluator_id" not in text
    assert "task_pack_fingerprint" not in text
    assert make_sealed_view(pack.episodes[0]).goal == "goal-x"

@pytest.mark.parametrize("bad", [
    {"answer": 1},
    {"hidden": {"solution":"x"}},
    {"evaluator_id":"secret"},
])
def test_173_contamination_fields_are_rejected(bad):
    with pytest.raises(ValueError):
        sanitize_for_agent(bad)

def test_174_fingerprint_is_stable_and_changes_on_public_change():
    a, b, c = make_pack(1), make_pack(1), make_pack(2)
    assert task_pack_fingerprint(a) == task_pack_fingerprint(b)
    assert task_pack_fingerprint(a) != task_pack_fingerprint(c)

def run_eval(payload):
    return subprocess.run(
        [sys.executable, "phase171_180_external/independent_evaluator.py"],
        input=json.dumps(payload), text=True, capture_output=True, check=False
    )

def test_175_evaluator_is_independent_process():
    proc = run_eval({
        "total_episodes":4, "completed_episodes":4, "successes":4,
        "contamination":False, "pack_fingerprint":"a"*64,
    })
    assert proc.returncode == 0, proc.stderr
    assert json.loads(proc.stdout)["passed"] is True

def test_176_evaluator_rejects_partial_or_contaminated_result():
    proc = run_eval({
        "total_episodes":4, "completed_episodes":3, "successes":4,
        "contamination":True, "pack_fingerprint":"b"*64,
    })
    assert proc.returncode != 0
    assert json.loads(proc.stdout)["passed"] is False
