
from .mirror7_phase274 import SafeEmbodiedController, StochasticGrid

def test_three_seeds_terminate():
    for seed in (1,2,3):
        e=StochasticGrid(seed=seed,slip=0.1,delay=1)
        assert SafeEmbodiedController(e).run(100)

def test_held_out_noise_level():
    e=StochasticGrid(seed=17,slip=0.25,delay=2)
    assert SafeEmbodiedController(e).run(150)

def test_partial_observation_has_bounded_view():
    e=StochasticGrid(size=8,seed=0)
    o=e.reset()
    assert len(o.visible)<=5
    assert (7,7) not in o.visible

def test_delay_is_explicit():
    e=StochasticGrid(delay=2,seed=0)
    e.reset()
    _,reward=e.step("WAIT")
    assert reward is None
    _,reward=e.step("WAIT")
    assert reward is None
    _,reward=e.step("WAIT")
    assert reward in (0,1)

def test_invalid_action_rejected():
    e=StochasticGrid()
    e.reset()
    try:
        e.step("JUMP")
    except ValueError:
        pass
    else:
        raise AssertionError("invalid action must fail closed")

def test_deterministic_same_seed():
    a=StochasticGrid(seed=42,slip=0.2); b=StochasticGrid(seed=42,slip=0.2)
    assert a.reset()==b.reset()
    for action in ("R","U","R","WAIT"):
        assert a.step(action)==b.step(action)
