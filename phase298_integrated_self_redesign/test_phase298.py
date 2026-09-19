from .mirror7_phase298 import IntegratedSelfRedesign

def test_integrated_run_promotes_source_redesign():
    e,rs=IntegratedSelfRedesign().run(7)
    assert rs[0].promoted and "values[-2]" in e.source

def test_integrated_run_stops_after_gap_closes():
    e,rs=IntegratedSelfRedesign().run(11)
    assert len(rs)==2 and not rs[1].promoted

def test_three_seed_integrations_are_stable():
    for seed in (2,5,8):
        e,rs=IntegratedSelfRedesign().run(seed)
        assert rs[0].promoted and "values[-2]" in e.source

def test_source_stays_bounded():
    e,_=IntegratedSelfRedesign().run(19,max_rounds=10)
    assert len(e.source)<500

def test_generated_public_inputs_hide_targets():
    suite=__import__("phase287_self_generated_tasks",fromlist=["SelfGeneratedTaskSuite"]).SelfGeneratedTaskSuite(3)
    assert all("target" not in t.public for t in suite.linear_sequence(4))

def test_no_arbitrary_import_is_generated():
    e,_=IntegratedSelfRedesign().run(3)
    assert "import " not in e.source

def test_redesign_keeps_solve_interface():
    e,_=IntegratedSelfRedesign().run(4)
    assert "def solve" in e.source

def test_repeatability_same_seed():
    a,ra=IntegratedSelfRedesign().run(21); b,rb=IntegratedSelfRedesign().run(21)
    assert a.source==b.source and ra==rb

def test_unseen_seed():
    e,rs=IntegratedSelfRedesign().run(997)
    assert rs[0].promoted

def test_bounded_rounds():
    e,rs=IntegratedSelfRedesign().run(31,max_rounds=1)
    assert len(rs)==1
