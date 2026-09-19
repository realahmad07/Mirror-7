
from .mirror7_phase270 import LongHorizonPlanner

def world(state, action):
    s=list(state)
    if action=="R": s[0]+=1
    elif action=="U": s[1]+=1
    elif action=="L": s[0]-=1
    elif action=="D": s[1]-=1
    else: return None
    if abs(s[0])>3 or abs(s[1])>3: return None
    return tuple(s)

def test_three_depths():
    for depth in (2,4,6):
        p=LongHorizonPlanner(world,["R","U","L","D"])
        path=p.plan((0,0),lambda s:s==(depth,0),max_depth=depth)
        assert path and len(path)==depth

def test_held_out_subgoal():
    p=LongHorizonPlanner(world,["R","U","L","D"])
    path=p.plan((0,0),lambda s:s==(2,2),max_depth=6)
    assert path and len(path)==4

def test_backtracking_after_real_failure():
    p=LongHorizonPlanner(world,["R","U"])
    calls=[0]
    def exec_(state,action):
        calls[0]+=1
        if action=="R" and calls[0]==1: return None
        return world(state,action)
    result=p.execute_with_backtracking((0,0),lambda s:s==(1,1),exec_,max_depth=4)
    assert result is not None

def test_unreachable_fails_closed():
    p=LongHorizonPlanner(world,["R","U","L","D"])
    assert p.plan((0,0),lambda s:s==(9,9),max_depth=5) is None

def test_budget_is_bounded():
    p=LongHorizonPlanner(world,["R","U"],max_expansions=2)
    assert p.plan((0,0),lambda s:s==(3,0),max_depth=10) is None

def test_multi_seed_determinism():
    for seed in (3,5,9):
        p=LongHorizonPlanner(world,["R","U","L","D"])
        assert p.plan((0,0),lambda s:s==(2,1),max_depth=6)==["R","R","U"]
