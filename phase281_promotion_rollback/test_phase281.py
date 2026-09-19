from .mirror7_phase281 import PromotionRegistry

def test_first_promotion_is_recorded():
    r = PromotionRegistry({"window": 1})
    rec = r.promote({"window": 2}, train_gain=.2, held_out_score=.8, regression_ok=True)
    assert rec.accepted and r.current == {"window": 2} and r.history[-1].version == 1

def test_failed_promotion_does_not_change_state():
    r = PromotionRegistry({"window": 1})
    rec = r.promote({"window": 3}, train_gain=.2, held_out_score=0.0, regression_ok=True)
    assert not rec.accepted and r.current == {"window": 1} and not r.history

def test_regression_failure_blocks_promotion():
    r = PromotionRegistry({"window": 1})
    rec = r.promote({"window": 3}, train_gain=.2, held_out_score=.9, regression_ok=False)
    assert not rec.accepted and r.current == {"window": 1}

def test_three_seed_promotion_determinism():
    for seed in (2, 5, 8):
        r = PromotionRegistry({"seed": seed, "window": 1})
        rec = r.promote({"seed": seed, "window": 2}, train_gain=.3, held_out_score=.8, regression_ok=True)
        assert rec.accepted and rec.version == 1 and r.current["window"] == 2

def test_rollback_restores_previous_variant():
    r = PromotionRegistry({"window": 1})
    r.promote({"window": 2}, train_gain=.2, held_out_score=.8, regression_ok=True)
    r.promote({"window": 3}, train_gain=.1, held_out_score=.7, regression_ok=True)
    rec = r.rollback()
    assert rec.accepted and r.current == {"window": 2}

def test_unknown_rollback_is_safe():
    r = PromotionRegistry({"window": 1})
    r.promote({"window": 2}, train_gain=.2, held_out_score=.8, regression_ok=True)
    rec = r.rollback(999)
    assert not rec.accepted and r.current == {"window": 2}

def test_empty_history_rollback_fails_closed():
    r = PromotionRegistry({"window": 1})
    rec = r.rollback()
    assert not rec.accepted and r.current == {"window": 1}

def test_history_is_immutable_view():
    r = PromotionRegistry({"window": 1})
    r.promote({"window": 2}, train_gain=.2, held_out_score=.8, regression_ok=True)
    h = r.history
    assert isinstance(h, tuple)
    assert h[0].after == {"window": 2}

def test_candidate_mapping_is_copied():
    r = PromotionRegistry({"window": 1})
    candidate = {"window": 2}
    r.promote(candidate, train_gain=.2, held_out_score=.8, regression_ok=True)
    candidate["window"] = 99
    assert r.current["window"] == 2
