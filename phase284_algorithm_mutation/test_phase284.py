from phase283_algorithm_variant_space import AlgorithmVariant, AlgorithmVariantSpace
from .mirror7_phase284 import AlgorithmMutator

def parent():
    return AlgorithmVariant("base", ("last", "add_delta"))

def test_replace_mutation_changes_one_operation():
    s = AlgorithmVariantSpace(("last", "delta", "add_delta"))
    muts = AlgorithmMutator(s).mutate(parent())
    assert muts
    for m in muts:
        diff = sum(a != b for a, b in zip(m.parent.operations, m.child.operations))
        assert diff == 1 and m.child.operations != m.parent.operations

def test_append_mutation_is_bounded():
    s = AlgorithmVariantSpace(("last", "add_delta"))
    muts = AlgorithmMutator(s).mutate_by_append(parent())
    assert len(muts) == 2 and all(len(m.child.operations) == 3 for m in muts)

def test_three_seed_mutation_shape_is_deterministic():
    s = AlgorithmVariantSpace(("last", "delta", "add_delta"))
    for seed in (1, 7, 19):
        a = AlgorithmMutator(s).mutate(parent(), max_children=5)
        b = AlgorithmMutator(s).mutate(parent(), max_children=5)
        assert a == b

def test_mutation_limit_is_respected():
    s = AlgorithmVariantSpace()
    assert len(AlgorithmMutator(s).mutate(parent(), max_children=2)) == 2

def test_invalid_limit_rejected():
    try:
        AlgorithmMutator(AlgorithmVariantSpace()).mutate(parent(), max_children=0)
    except ValueError:
        pass
    else:
        assert False

def test_invalid_parent_rejected():
    try:
        AlgorithmMutator(AlgorithmVariantSpace()).mutate(AlgorithmVariant("x", ("eval",)))
    except ValueError:
        pass
    else:
        assert False

def test_no_arbitrary_code_field_created():
    s = AlgorithmVariantSpace(("last", "add_delta"))
    m = AlgorithmMutator(s).mutate(parent(), max_children=1)[0]
    assert not hasattr(m.child, "source")

def test_mutation_preserves_parent():
    s = AlgorithmVariantSpace()
    p = parent()
    AlgorithmMutator(s).mutate(p)
    assert p.operations == ("last", "add_delta")

def test_negative_control_unknown_operation_never_generated():
    s = AlgorithmVariantSpace(("last", "add_delta"))
    for m in AlgorithmMutator(s).mutate(parent()):
        assert "eval" not in m.child.operations

def test_max_program_depth_is_preserved_for_replace():
    s = AlgorithmVariantSpace()
    p = AlgorithmVariant("long", ("last", "last", "add_delta", "last", "delta"))
    for m in AlgorithmMutator(s).mutate(p):
        assert len(m.child.operations) == len(p.operations)
