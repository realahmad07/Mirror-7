from .mirror7_phase87 import CurriculumMemory

def test_uncertainty_and_novelty_drive_priority():
    c=CurriculumMemory()
    low=c.propose((0,),"a",.1,.1)
    high=c.propose((1,),"b",.9,.9)
    assert c.next()==high and high.priority>low.priority

def test_repeated_item_is_downweighted():
    c=CurriculumMemory()
    first=c.propose((0,),"a",1,1)
    second=c.propose((0,),"a",1,1)
    assert second.priority<first.priority

def test_bounds_and_fail_closed():
    c=CurriculumMemory()
    assert c.propose((0,),"a",float("nan"),.5) is None
    assert c.propose((), "a", .5,.5) is None

def test_consume_removes_selected_item():
    c=CurriculumMemory()
    c.propose((0,),"a",.2,.8)
    x=c.consume()
    assert x is not None and c.next() is None
