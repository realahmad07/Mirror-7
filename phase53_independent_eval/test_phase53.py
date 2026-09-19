from .evaluator import EpisodeScore, build_blind_suite, leakage_audit, summarize


def test_suite_shape():
    tasks = build_blind_suite()
    assert len(tasks) == 21
    assert len({t.task_id for t in tasks}) == 21
    assert all(t.task_id.startswith("ep-") for t in tasks)
    assert sum(t.split == "train" for t in tasks) == 9
    assert sum(t.split == "heldout" for t in tasks) == 12


def test_public_protocol_is_blind():
    assert leakage_audit(build_blind_suite()) == []


def test_strict_gate():
    good = (
        [EpisodeScore(f"ep-t{i}", "linear", i, "train", True, 3, 0, False) for i in range(9)]
        + [EpisodeScore(f"ep-h{i}", "linear", i, "heldout", True, 3, 0, False) for i in range(12)]
    )
    assert summarize(good)["passed"] is True

    bad = good[:-1] + [EpisodeScore("ep-bad", "composition", 11, "heldout", False, 40, 1, True)]
    assert summarize(bad)["passed"] is False
