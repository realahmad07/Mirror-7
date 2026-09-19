import pytest
from .mirror7_phase266 import LifelongMemory

def test_interference_protection():
    mem = LifelongMemory(capacity=10)
    mem.observe((0,), "move", (1,), "Env_A")
    mem.observe((0,), "move", (1,), "Env_A")
    mem.observe((0,), "move", (2,), "Env_B")
    assert mem.predict((0,), "move", "Env_B") == (2,)
    assert mem.predict((0,), "move", "Env_A") == (1,)

def test_capacity_eviction():
    mem = LifelongMemory(capacity=2)
    mem.observe((1,), "move", (2,), "Env_A")
    mem.observe((1,), "move", (2,), "Env_A")
    mem.observe((2,), "move", (3,), "Env_A")
    mem.observe((3,), "move", (4,), "Env_A")
    assert mem.predict((1,), "move") == (2,)
    assert mem.predict((2,), "move") is None
    assert mem.predict((3,), "move") == (4,)

def test_capacity_validation():
    import pytest
    with pytest.raises(ValueError): LifelongMemory(0)
    with pytest.raises(ValueError): LifelongMemory(-1)

def test_three_environment_rotation_and_held_out():
    mem=LifelongMemory(capacity=8)
    for env,nxt in (("A",(1,)),("B",(2,)),("C",(3,))):
        mem.observe((0,),"move",nxt,env); mem.observe((0,),"move",nxt,env)
    assert mem.predict((0,),"move","A")== (1,)
    assert mem.predict((0,),"move","B")== (2,)
    assert mem.predict((0,),"move","C")== (3,)
    assert mem.predict((9,),"move","A") is None
