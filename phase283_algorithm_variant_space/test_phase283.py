from .mirror7_phase283 import AlgorithmVariant, AlgorithmVariantSpace

def test_valid_variant_executes():
    s = AlgorithmVariantSpace()
    v = AlgorithmVariant("linear", ("last", "add_delta"))
    assert s.execute(v, [1, 3, 5]) == 7

def test_three_seed_execution_is_stable():
    s = AlgorithmVariantSpace()
    v = AlgorithmVariant("linear", ("last", "add_delta"))
    for seed in (2, 5, 8):
        values = [seed, seed + 4, seed + 8]
        assert s.execute(v, values) == seed + 12

def test_variants_are_declarative_not_executable_code():
    v = AlgorithmVariant("x", ("last", "add_delta"))
    assert isinstance(v.operations, tuple)
    assert not hasattr(v, "code")

def test_unknown_operation_is_rejected():
    s = AlgorithmVariantSpace()
    try:
        s.execute(AlgorithmVariant("bad", ("eval",)), [1, 2])
    except ValueError:
        pass
    else:
        assert False

def test_empty_input_fails_closed():
    s = AlgorithmVariantSpace()
    try:
        s.execute(AlgorithmVariant("x", ("last",)), [])
    except ValueError:
        pass
    else:
        assert False

def test_enumeration_is_bounded_and_deterministic():
    s = AlgorithmVariantSpace(("last", "add_delta"))
    a = s.enumerate(3)
    b = s.enumerate(3)
    assert a == b
    assert len(a) == 14

def test_enumeration_rejects_invalid_depth():
    try:
        AlgorithmVariantSpace().enumerate(0)
    except ValueError:
        pass
    else:
        assert False

def test_last_variant():
    s = AlgorithmVariantSpace()
    assert s.execute(AlgorithmVariant("x", ("last",)), [2, 9]) == 9

def test_constant_negative_control():
    s = AlgorithmVariantSpace()
    assert s.execute(AlgorithmVariant("x", ("constant",)), [2, 9]) == 0
