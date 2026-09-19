from .mirror7_phase100 import Mirror7FinalBoundary

def test_final_boundary_integration():
    b=Mirror7FinalBoundary([
        {"id":"a","input":1,"expected":2},
        {"id":"h","input":2,"expected":4,"held_out":True},
    ])
    b.learn_experience((1,2),(0,),"inc",(1,),.9)
    b.consolidate(("inc",),(1,)); b.ground([0,1,2])
    transitions={((0,),"a"):((1,),.1),((1,),"b"):((2,),.1)}
    p=b.plan(transitions,["a","b"],(0,),(2,))
    r=b.evaluate(lambda x:x*2)
    assert p.predicted==(2.0,)
    assert r.integrated and r.passed==2 and r.invalid==0 and r.held_out_passed==1

def test_final_boundary_rejects_bad_response():
    b=Mirror7FinalBoundary([{"id":"a","input":1,"expected":2}])
    r=b.evaluate(lambda x:None)
    assert r.invalid==1 and r.passed==0

def test_final_boundary_report_is_frozen():
    b=Mirror7FinalBoundary([{"id":"a","input":1,"expected":2}])
    r=b.evaluate(lambda x:x+1)
    assert r.total==1 and r.passed==1 and r.integrated
