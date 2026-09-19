from phase294_source_evaluator import SourceEvaluation
from .mirror7_phase296 import SourcePromotionRegistry

BASE="def solve(values):\n return values[-1]\n"
GOOD="def solve(values):\n return values[-1] + 1\n"

def s(t,h,r=True): return SourceEvaluation(t,h,r,1)

def test_promotion_changes_source():
    r=SourcePromotionRegistry(BASE); rec=r.promote(GOOD,s(.2,.2),s(.8,.8))
    assert rec.accepted and r.source==GOOD and rec.version==1

def test_bad_promotion_does_not_change_source():
    r=SourcePromotionRegistry(BASE); rec=r.promote(GOOD,s(.8,.8),s(.7,.9))
    assert not rec.accepted and r.source==BASE

def test_heldout_required():
    r=SourcePromotionRegistry(BASE); rec=r.promote(GOOD,s(.2,.2),s(.8,0))
    assert not rec.accepted

def test_regression_required():
    r=SourcePromotionRegistry(BASE); rec=r.promote(GOOD,s(.2,.2),s(.8,.8,False))
    assert not rec.accepted

def test_fingerprint_stable():
    r=SourcePromotionRegistry(BASE); assert r.fingerprint(BASE)==r.fingerprint(BASE)

def test_three_seed_promotions_are_deterministic():
    for _ in (2,5,8):
        r=SourcePromotionRegistry(BASE); assert r.promote(GOOD,s(.2,.2),s(.8,.8)).accepted

def test_history_is_immutable_view():
    r=SourcePromotionRegistry(BASE); r.promote(GOOD,s(.2,.2),s(.8,.8)); assert isinstance(r.history,tuple)

def test_unknown_rollback_safe():
    r=SourcePromotionRegistry(BASE); assert not r.rollback(99).accepted and r.source==BASE

def test_empty_rollback_safe():
    r=SourcePromotionRegistry(BASE); assert not r.rollback().accepted

def test_reject_does_not_append_history():
    r=SourcePromotionRegistry(BASE); r.promote(GOOD,s(.8,.8),s(.7,.9)); assert len(r.history)==0


def test_rollback_restores_exact_previous_source():
    r=SourcePromotionRegistry(BASE)
    r.promote(GOOD,s(.2,.2),s(.8,.8))
    rec=r.rollback()
    assert rec.accepted and r.source==BASE

def test_second_rollback_does_not_target_rollback_record():
    r=SourcePromotionRegistry(BASE)
    r.promote(GOOD,s(.2,.2),s(.8,.8))
    r.rollback()
    rec=r.rollback()
    assert rec.accepted and r.source==BASE
