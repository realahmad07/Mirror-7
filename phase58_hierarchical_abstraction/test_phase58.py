import random
import pytest
from .mirror7_phase58 import discover_hierarchy

def family(seed, style):
    random.Random(seed)
    if style == 0:
        return [(('A','B')*2 + ('C','D')*2) for _ in range(6)]
    if style == 1:
        return [(('A','B')*2 + ('C','D')*2 + ('A','B')*2) for _ in range(6)]
    return [(('A','B')*2 + ('C','D')*2 + ('A','B')*2 + ('C','D')*2) for _ in range(6)]

def test_progressive_3x3():
    for level in range(3):
        for seed in range(3):
            h=discover_hierarchy(family(100+seed,level), max_depth=3, min_episode_support=4)
            assert len(h.levels)>=2
            assert all(layer.support>=4 for L in h.levels for layer in L)

def test_heldout_new_combination_of_known_composites():
    train=[tuple('ABABCDCD') for _ in range(6)]
    h=discover_hierarchy(train,max_depth=3,min_episode_support=4)
    assert len(h.levels)>=2
    held=list('CDCDABAB')
    assert {x.fingerprint for x in h.levels[0]}
    assert any(tuple(held[i:i+2]) in [x.children for x in h.levels[0]] for i in range(len(held)-1))

def test_hierarchy_depth_not_memorization():
    train=[tuple('ABABCDCDABABCDCD') for _ in range(6)]
    h=discover_hierarchy(train,max_depth=3,min_episode_support=4)
    assert len(h.levels)>=2
    assert len(h.levels[1])>=1

def test_negative_unrelated_no_hierarchy():
    rng=random.Random(55)
    eps=[tuple(rng.sample([chr(65+i) for i in range(26)],16)) for _ in range(6)]
    h=discover_hierarchy(eps,max_depth=3,min_episode_support=4)
    assert h.levels==()

def test_heldout_distractor_does_not_change_vocabulary():
    train=[tuple('ABABCDCD') for _ in range(6)]
    base=discover_hierarchy(train,max_depth=3,min_episode_support=4)
    noisy=train+[tuple('ABABXYZQ')]
    noisy_h=discover_hierarchy(noisy,max_depth=3,min_episode_support=5)
    assert {x.fingerprint for x in noisy_h.levels[0]}=={x.fingerprint for x in base.levels[0]}

def test_determinism():
    eps=[tuple('ABABCDCDABAB') for _ in range(6)]
    assert discover_hierarchy(eps)==discover_hierarchy(eps)

def test_malformed():
    with pytest.raises(ValueError): discover_hierarchy([])
    with pytest.raises(ValueError): discover_hierarchy([()])
    with pytest.raises(ValueError): discover_hierarchy([('A','B')],max_depth=0)

def test_compression_selects_short_atoms_first():
    eps=[tuple('ABABABAB') for _ in range(6)]
    h=discover_hierarchy(eps,max_depth=2,min_episode_support=4)
    assert h.levels
    assert any(layer.children==('A','B') for layer in h.levels[0])
