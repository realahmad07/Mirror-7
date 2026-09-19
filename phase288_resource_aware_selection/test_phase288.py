from .mirror7_phase288 import ResourceAwareSelector

def test_higher_gain_ranks_higher():
    s=ResourceAwareSelector(); a=s.make("a",.4,.8,1,2); b=s.make("b",.1,.8,1,2); assert s.rank([b,a])[0].name=="a"

def test_complexity_is_penalized():
    s=ResourceAwareSelector(complexity_weight=.2); a=s.make("a",.4,.8,1,2); b=s.make("b",.4,.8,8,2); assert s.rank([b,a])[0].name=="a"

def test_operation_count_is_penalized():
    s=ResourceAwareSelector(operation_weight=.1); a=s.make("a",.4,.8,1,1); b=s.make("b",.4,.8,1,8); assert s.rank([b,a])[0].name=="a"

def test_three_seed_order_is_deterministic():
    s=ResourceAwareSelector()
    for _ in (2,5,8):
        xs=[s.make("a",.2,.5,1,1),s.make("b",.2,.5,2,2)]
        assert s.rank(xs)[0].name=="a"

def test_negative_metrics_fail_closed():
    try: ResourceAwareSelector().make("x",-.1,.5,1,1)
    except ValueError: pass
    else: assert False

def test_held_out_bounds_checked():
    try: ResourceAwareSelector().make("x",.1,1.1,1,1)
    except ValueError: pass
    else: assert False

def test_empty_selection_safe():
    assert ResourceAwareSelector().rank([])==[]

def test_exact_ties_are_deterministic():
    s=ResourceAwareSelector(); xs=[s.make("b",.2,.5,1,1),s.make("a",.2,.5,1,1)]
    assert s.rank(xs)[0].name=="b"

def test_zero_weights_keep_evidence_priority():
    s=ResourceAwareSelector(0,0); xs=[s.make("a",.3,.4,100,100),s.make("b",.2,.5,1,1)]
    assert s.rank(xs)[0].name=="b"
