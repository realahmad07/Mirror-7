from .mirror7_phase125 import AdaptiveCurriculum,CurriculumItem

def test_high_uncertainty_is_prioritized():
 c=AdaptiveCurriculum(); a=CurriculumItem("a",.9,.1,.1,2); b=CurriculumItem("b",.1,.9,.1,2); assert c.select([a,b]).name=="a"

def test_novelty_matters_when_uncertainty_is_equal():
 c=AdaptiveCurriculum(); a=CurriculumItem("a",.5,.1,.1,2); b=CurriculumItem("b",.5,.9,.1,2); assert c.select([a,b]).name=="b"

def test_budget_filters_expensive_tasks():
 c=AdaptiveCurriculum(3); assert c.select([CurriculumItem("expensive",1,1,1,4)]) is None

def test_relevance_breaks_close_tie():
 c=AdaptiveCurriculum(); a=CurriculumItem("a",.5,.5,.1,2); b=CurriculumItem("b",.5,.5,.9,2); assert c.select([a,b]).name=="b"