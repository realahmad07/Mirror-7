from phase76_autonomous_curriculum import AutonomousCurriculum

def test_selects_highest_score():
    c=AutonomousCurriculum(); assert c.select([1,2,3],lambda x:x)==3

def test_budget_is_hard():
    c=AutonomousCurriculum(budget=2); c.run([1,2,3,4],lambda x:x); assert c.used==2

def test_bad_scores_fail_closed():
    c=AutonomousCurriculum(); assert c.select([1,2],lambda x:1/0) is None

def test_stop_condition():
    c=AutonomousCurriculum(); r=c.run([1,5,2],lambda x:x,lambda x:x==5); assert r.status=="target_reached" and r.selected==5

def test_empty_candidates():
    c=AutonomousCurriculum(); assert c.run([],lambda x:x).status=="no_candidate"
