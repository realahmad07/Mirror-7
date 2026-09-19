from .mirror7_phase99 import IndependentEvaluator

def test_frozen_hash_is_deterministic():
    c=[{"id":"a","input":1,"expected":2},{"id":"h","input":2,"expected":4,"held_out":True}]
    assert IndependentEvaluator(c).suite_hash==IndependentEvaluator(c).suite_hash

def test_black_box_report():
    e=IndependentEvaluator([
        {"id":"a","input":1,"expected":2},
        {"id":"h","input":2,"expected":4,"held_out":True},
        {"id":"bad","input":3,"expected":6},
    ])
    r=e.report(e.run(lambda x:x*2 if x<3 else None))
    assert r.total==3 and r.passed==2 and r.invalid==1 and r.held_out_passed==1

def test_exception_is_invalid():
    e=IndependentEvaluator([{"id":"a","input":1,"expected":2}])
    r=e.report(e.run(lambda x:1/0))
    assert r.invalid==1 and r.passed==0

def test_malformed_suite_rejected():
    try: IndependentEvaluator([{"id":"a"}]); assert False
    except ValueError: pass
