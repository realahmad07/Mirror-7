"""Pytest integration checks for mirror_eval."""
from __future__ import annotations

from mirror_eval import run_full_eval


def test_mirror_eval_independent_pillars_pass():
    # Fast hermetic CI mode (no reliance on multi-MB external disk artifact)
    report = run_full_eval(check_production_vault=False, save_json_path=None)
    assert report["overall_passed"] == report["overall_total"], report
    assert report["pillars"]["architecture_independence"]["violations"] == []
