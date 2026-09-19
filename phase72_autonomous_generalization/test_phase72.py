from phase72_autonomous_generalization import GeneralizationLoop

class Learner:
    def __init__(self): self.data=[]
    def __call__(self,a,b): self.data.append((a,b))
    def predict(self,a,b): return a+b if self.data else None

def test_autonomous_generalization():
    g=GeneralizationLoop(Learner(),budget=3)
    r=g.run(((1,2),(2,3)),(4,5))
    assert r.status=="accepted" and r.prediction==9 and r.experiments_used==2

def test_hard_budget():
    g=GeneralizationLoop(Learner(),budget=1)
    r=g.run(((1,2),(2,3),(3,4)),(4,5))
    assert r.experiments_used==1

def test_rejects_no_evidence():
    g=GeneralizationLoop(Learner(),budget=2)
    r=g.run((),(4,5))
    assert r.status=="rejected" and r.confidence==0

def test_candidate_selection():
    g=GeneralizationLoop(Learner(),budget=2)
    assert g.choose(["a","bbb","cc"],len)=="bbb"

def test_fail_closed():
    g=GeneralizationLoop(Learner(),budget=0)
    assert g.choose(["x"],len) is None
    assert g.fail_closed()["budget"]==0
