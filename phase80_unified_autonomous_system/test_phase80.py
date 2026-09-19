from phase80_unified_autonomous_system import UnifiedAutonomousSystem

def test_unknown_then_predict():
    m=UnifiedAutonomousSystem(); r1=m.step((0,),["inc"],lambda s,a:(s[0]+1,)); assert r1.predicted is None
    r2=m.step((0,),["inc"],lambda s,a:(1,)); assert r2.predicted==(1.0,) and r2.discrepancy==0

def test_discrepancy_revision():
    m=UnifiedAutonomousSystem(max_revisions=1); m.step((0,),["a"],lambda s,a:(1,)); r=m.step((0,),["a"],lambda s,a:(2,)); assert r.revised

def test_fail_closed_revision_budget():
    m=UnifiedAutonomousSystem(max_revisions=0); m.step((0,),["a"],lambda s,a:(1,))
    try: m.step((0,),["a"],lambda s,a:(2,)); assert False
    except RuntimeError as e: assert "budget" in str(e)

def test_invalid_inputs():
    m=UnifiedAutonomousSystem()
    try: m.step((0,),[],lambda s,a:(1,)); assert False
    except ValueError: pass
    try: m.step((0,),["a"],lambda s,a:(1,),lambda s,a,p:float("nan")); assert False
    except ValueError: pass
