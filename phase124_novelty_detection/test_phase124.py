from .mirror7_phase124 import NoveltyDetector

def test_first_observation_is_novel():
    assert NoveltyDetector().observe((0,)).novel

def test_nearby_point_is_familiar():
    n=NoveltyDetector(.5); n.observe((0,0)); assert not n.observe((.2,.1)).novel

def test_far_point_is_novel():
    n=NoveltyDetector(.5); n.observe((0,0)); assert n.observe((2,0)).novel

def test_dimension_mismatch_is_novel():
    n=NoveltyDetector(); n.observe((0,0)); assert n.observe((0,)).novel