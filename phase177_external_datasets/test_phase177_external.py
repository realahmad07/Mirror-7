from __future__ import annotations
import json
import subprocess
import sys

import pytest

from phase171_180_external.protocol import make_sealed_view, task_pack_fingerprint
from phase177_external_datasets.uci_iris_pack import (
    EXPECTED_CLASSES, EXPECTED_ROWS, fetch_iris_records, make_external_pack,
)

@pytest.fixture(scope="module")
def records():
    return fetch_iris_records()

def test_177_real_external_dataset_has_expected_shape(records):
    assert len(records) == EXPECTED_ROWS
    assert {r.label for r in records} == set(EXPECTED_CLASSES)
    assert all(len(r.features) == 4 for r in records)

def test_177_external_pack_uses_real_rows_without_target_leak(records):
    pack = make_external_pack(records[:12])
    assert len(pack.episodes) == 12
    for episode in pack.episodes:
        sealed = make_sealed_view(episode).to_dict()
        assert "Iris-" not in json.dumps(sealed)
        assert "label" not in json.dumps(sealed).lower()
        assert episode.legal_actions == EXPECTED_CLASSES
    assert len(task_pack_fingerprint(pack)) == 64

def test_177_external_pack_fingerprint_is_stable(records):
    a = make_external_pack(records[:12])
    b = make_external_pack(records[:12])
    assert task_pack_fingerprint(a) == task_pack_fingerprint(b)

def test_177_external_pack_crosses_unchanged_independent_evaluator(records):
    pack = make_external_pack(records[:12])
    payload = {
        "total_episodes": len(pack.episodes),
        "completed_episodes": len(pack.episodes),
        # Plumbing gate only: the test supplies the externally known labels to
        # establish the evaluator boundary. This is NOT a Mirror 7 performance claim.
        "successes": len(pack.episodes),
        "contamination": False,
        "pack_fingerprint": task_pack_fingerprint(pack),
    }
    proc = subprocess.run(
        [sys.executable, "phase171_180_external/independent_evaluator.py"],
        input=json.dumps(payload), text=True, capture_output=True, check=False,
    )
    assert proc.returncode == 0, proc.stderr
    result = json.loads(proc.stdout)
    assert result["passed"] is True
