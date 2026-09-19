from .mirror7_phase128 import GeneralizationAgent
from phase122_skill_library import Skill
from phase125_adaptive_curriculum import CurriculumItem

def test_integrated_generalization_path():
 a=GeneralizationAgent(); s=Skill("inc",lambda x:x>=0,lambda x:x+1,lambda x:x>0,.9)
 r=a.run(0,s,True,{"a":("b",),"b":("a",)},{"x":("y",),"y":("x",)},(0,),[CurriculumItem("learn",.9,.4,.8,1)],[lambda x:x+1],[.9,.85,.8])
 assert r.skill_promoted and r.transferred and r.novel and r.curriculum=="learn" and r.completed and r.generalized

def test_bad_skill_is_not_promoted():
 a=GeneralizationAgent(); s=Skill("bad",lambda x:True,lambda x:x,lambda x:True,.2)
 r=a.run(0,s,False,{"a":("b",)},{"x":("y",)},(0,),[],[],[.9,.9,.9])
 assert not r.skill_promoted

def test_weak_transfer_and_generalization_are_visible():
 a=GeneralizationAgent(); s=Skill("inc",lambda x:True,lambda x:x+1,lambda x:True,.9)
 r=a.run(0,s,True,{"a":("b",),"c":("d",)},{"x":("y",)},(9,),[],[],[.95,.2])
 assert not r.transferred and r.novel and not r.generalized

def test_curriculum_can_be_empty_and_execution_can_finish():
 a=GeneralizationAgent(); s=Skill("id",lambda x:True,lambda x:x,lambda x:True,.8)
 r=a.run(1,s,True,{"a":("b",)},{"x":("y",)},(0,),[],[],[.8,.8,.8])
 assert r.completed and r.curriculum is None

def test_checkpoint_integration_handles_multi_step_path():
 a=GeneralizationAgent(); s=Skill("inc",lambda x:True,lambda x:x+1,lambda x:True,.9)
 r=a.run(0,s,True,{"a":("b",)},{"x":("y",)},(0,),[],[lambda x:x+1,lambda x:x+1],[.9,.9,.9])
 assert r.completed

def test_gate_rejects_unseen_weakness_even_with_good_average():
 a=GeneralizationAgent(); s=Skill("inc",lambda x:True,lambda x:x+1,lambda x:True,.9)
 r=a.run(0,s,True,{"a":("b",)},{"x":("y",)},(0,),[],[],[1.0,.95,.1])
 assert not r.generalized