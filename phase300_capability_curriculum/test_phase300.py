from phase287_self_generated_tasks import SelfGeneratedTaskSuite
from .mirror7_phase300 import CapabilityCurriculum

def gen(seed):
    return SelfGeneratedTaskSuite(seed).linear_sequence(8)

def test_curriculum_builds_tasks():
    c=CapabilityCurriculum(gen)
    xs=c.build("sequence",7,6)
    assert len(xs)==6 and xs[0].capability=="sequence"

def test_public_task_hides_target():
    x=CapabilityCurriculum(gen).build("sequence",7,6)[0]
    assert "target" not in x.public

def test_three_seed_curricula_differ():
    c=CapabilityCurriculum(gen)
    assert c.build("sequence",1,6)!=c.build("sequence",999,6)

def test_split_is_deterministic():
    c=CapabilityCurriculum(gen); xs=c.build("sequence",4,6)
    a=c.split(xs); b=c.split(xs); assert a==b

def test_split_has_all_roles():
    a,b,c=CapabilityCurriculum(gen).split(CapabilityCurriculum(gen).build("sequence",4,6))
    assert len(a)==4 and len(b)==1 and len(c)==1

def test_invalid_count_rejected():
    try: CapabilityCurriculum(gen).build("x",1,0)
    except ValueError: pass
    else: assert False

def test_short_generator_rejected():
    try: CapabilityCurriculum(lambda seed: []).build("x",1,3)
    except ValueError: pass
    else: assert False

def test_invalid_split_rejected():
    try: CapabilityCurriculum(gen).split([])
    except ValueError: pass
    else: assert False
