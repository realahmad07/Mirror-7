from mirror_eval.blueprint_integrity import run_blueprint_integrity_suite


def test_held_out_blueprint_negative_controls():
    rows = run_blueprint_integrity_suite()
    assert len(rows) >= 5
    assert all(row["passed"] for row in rows), rows
