from .mirror7_phase122 import SkillLibrary,Skill

def skill():
    return Skill("inc",lambda s:s>=0,lambda s:s+1,lambda s:s>0,.9)

def test_only_verified_skill_is_promoted():
    l=SkillLibrary(); assert l.promote(skill(),True); assert "inc" in l.skills

def test_unverified_skill_is_rejected():
    l=SkillLibrary(); assert not l.promote(skill(),False); assert not l.skills

def test_applicable_skills_are_ranked():
    l=SkillLibrary(); l.promote(skill(),True); l.promote(Skill("low",lambda s:True,lambda s:s,lambda s:True,.2),True)
    assert l.applicable(1)[0].name=="inc"

def test_execution_requires_postcondition():
    l=SkillLibrary(); l.promote(Skill("bad",lambda s:True,lambda s:0,lambda x:x==1,.9),True)
    ok,_,reason=l.execute("bad",1); assert not ok and "postcondition" in reason