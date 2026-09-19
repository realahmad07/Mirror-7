from .mirror7_phase265 import GroundedLanguageAgent, AffordanceDiscoverer

def test_language_action_grounding():
    agent=GroundedLanguageAgent(); agent.learn_action_mapping("please step forward","action_1"); agent.learn_action_mapping("turn to the right","action_2")
    assert agent.translate_instruction("forward")=="action_1"
    assert agent.translate_instruction("turn")=="action_2"
    assert agent.translate_instruction("jump") is None

def test_language_state_composition_and_ambiguity():
    agent=GroundedLanguageAgent(); agent.learn_state_mapping("red",(1.0,0.0)); agent.learn_state_mapping("heavy",(0.0,1.0))
    assert agent.interpret_state("a red heavy object")== (1.0,1.0)
    agent.learn_action_mapping("open door","open"); agent.learn_action_mapping("open box","lift")
    assert agent.translate_instruction("open") is None

def test_affordance_discovery():
    disc=AffordanceDiscoverer(); state=(0.0,0.0); disc.observe_affordance(state,"push",True); disc.observe_affordance(state,"pull",False)
    assert disc.get_valid_actions(state)==["push"]


def test_language_held_out_composition_and_affordance_conflict():
    agent=GroundedLanguageAgent(); agent.learn_action_mapping("move forward","move"); agent.learn_action_mapping("turn right","turn")
    assert agent.translate_instruction("forward")=="move"
    assert agent.translate_instruction("forward right") is None
    disc=AffordanceDiscoverer(); state=(1.0,); disc.observe_affordance(state,"push",True); disc.observe_affordance(state,"push",True); disc.observe_affordance(state,"push",False)
    assert disc.get_valid_actions(state)==["push"]
