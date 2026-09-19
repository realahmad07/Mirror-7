
from .mirror7_phase276 import AblationHarness, ReproducibilityManifest, ScalingStudy

def test_monotonic_scaling():
    s=ScalingStudy().run([1,2,4,8],lambda n:n*n)
    assert [p.operations for p in s]==[1,4,16,64]

def test_progressive_and_held_out_sizes():
    s=ScalingStudy().run([3,6,12],lambda n:3*n+1)
    assert all(p.operations>0 for p in s)
    assert s[-1].operations==37

def test_ablation_is_explicit():
    a=AblationHarness().compare("task",lambda _:True,lambda _:False)
    assert a=={"full":True,"ablated":False}

def test_reproducible_manifest():
    m=ReproducibilityManifest()
    assert m.digest({"a":[1,2],"b":3})==m.digest({"b":3,"a":[1,2]})

def test_manifest_changes_when_output_changes():
    m=ReproducibilityManifest()
    assert m.digest({"x":1})!=m.digest({"x":2})

def test_seeded_workload_shape():
    for seed in (1,2,3):
        n=seed*10
        p=ScalingStudy().run([n],lambda x:x+seed)[0]
        assert p.operations==n+seed
