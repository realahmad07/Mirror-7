from .mirror7_phase303 import RegressionGuard

def test_primary_gain_accepted():
    r=RegressionGuard().evaluate({"a":.5,"b":.8},{"a":.7,"b":.8},"a"); assert r.accepted

def test_protected_regression_blocks():
    r=RegressionGuard().evaluate({"a":.5,"b":.8},{"a":.7,"b":.6},"a"); assert not r.accepted and r.protected_regressions==("b",)

def test_tolerance_allows_small_regression():
    r=RegressionGuard(tolerance=.1).evaluate({"a":.5,"b":.8},{"a":.7,"b":.75},"a"); assert r.accepted

def test_min_gain_blocks_tiny_change():
    r=RegressionGuard(min_primary_gain=.05).evaluate({"a":.5},{"a":.53},"a"); assert not r.accepted

def test_three_seed_like_evaluations_stable():
    g=RegressionGuard()
    for _ in (2,5,8): assert g.evaluate({"a":.5},{"a":.7},"a").accepted

def test_missing_primary_rejected():
    try: RegressionGuard().evaluate({"a":.5},{"b":.7},"a")
    except KeyError: pass
    else: assert False

def test_invalid_parameters_rejected():
    try: RegressionGuard(-1)
    except ValueError: pass
    else: assert False

def test_empty_protected_set_safe():
    r=RegressionGuard().evaluate({"a":.5},{"a":.7},"a"); assert r.accepted and r.protected_regressions==()
