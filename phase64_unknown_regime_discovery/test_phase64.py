import random
import pytest
from .mirror7_phase64 import UnknownRegimeAgent, UnknownRegimeBank, _effect


class RegimeEnv:
    def __init__(self, seed, mode):
        acts=['drive','probe','toggle']
        random.Random(seed).shuffle(acts)
        self.actions=tuple(acts)
        self.mode=mode
        self.state=(0,0)

    def reset(self, s=(0,0)):
        self.state=tuple(s)
        return self.state

    def legal_actions(self, state):
        return self.actions

    def step(self, action):
        x,y=self.state
        if action=='drive': x += (1 if self.mode==0 else 2 if self.mode==1 else 3)
        elif action=='probe': y += (0 if self.mode==0 else 5 if self.mode==1 else 9)
        elif action=='toggle': y ^= 1
        self.state=(x,y)
        return self.state


def test_progressive_unknown_regime_discovery_3x3():
    for family in (0,1,2):
        for seed in range(3):
            env=RegimeEnv(100+seed,family)
            agent=UnknownRegimeAgent(max_experiment_depth=3,experiment_budget=10)
            result=agent.run(env,start=(0,0),goal=(6,family and (5 if family==1 else 9) or 0),max_steps=16)
            assert result['solved']
            assert result['invalid_actions']==0
            assert result['experiments']>=2
            assert result['discovered_regimes']>=1


def test_heldout_regime_is_invented_not_selected_from_labels():
    agent=UnknownRegimeAgent(max_experiment_depth=3,experiment_budget=20)
    for mode in (0,1):
        env=RegimeEnv(500+mode,mode)
        agent.reset_context((4,0))
        agent.run(env,start=(0,0),goal=(4,0),max_steps=12)
    before=len(agent.bank.hypotheses)
    env=RegimeEnv(777,2)
    result=agent.run(env,start=(0,0),goal=(6,0),max_steps=16)
    assert result['solved']
    assert result['discovered_regimes'] > before
    assert result['invalid_actions']==0


def test_experiment_sequence_maximizes_known_disagreement():
    bank=UnknownRegimeBank(min_support=1,merge_threshold=0.9)
    r0=[((0,0),'drive',(1,0)),((1,0),'probe',(1,0))]
    r1=[((0,0),'drive',(1,0)),((1,0),'probe',(1,5))]
    bank.add_experiment(r0); bank.add_experiment(r1)
    agent=UnknownRegimeAgent(max_experiment_depth=3,experiment_budget=8)
    agent.bank=bank
    exp=agent.design_experiment((0,0),(4,5),('drive','probe','toggle'))
    assert exp is not None
    assert 'probe' in exp.sequence
    assert exp.predicted_partitions >= 2


def test_repeated_same_regime_does_not_create_spurious_clusters():
    agent=UnknownRegimeAgent(max_experiment_depth=3,experiment_budget=12)
    env=RegimeEnv(12,1)
    for _ in range(3):
        result=agent.run(env,start=(0,0),goal=(6,0),max_steps=16)
        assert result['solved']
    assert len(agent.bank.hypotheses) <= 2


def test_irrelevant_state_noise_does_not_split_regime():
    bank=UnknownRegimeBank(min_support=1,merge_threshold=1.0)
    t1=[((0,7),'drive',(1,7)),((1,7),'probe',(1,9))]
    t2=[((0,123),'drive',(1,123)),((1,123),'probe',(1,125))]
    assert _effect((0,7),(1,7))==_effect((0,123),(1,123))
    assert _effect((1,7),(1,9))==_effect((1,123),(1,125))
    bank.add_experiment(t1)
    bank.add_experiment(t2)
    assert len(bank.hypotheses)==1


def test_active_regime_reidentifies_after_controlled_switch():
    agent=UnknownRegimeAgent(max_experiment_depth=3,experiment_budget=16)
    env0=RegimeEnv(1,0)
    r0=agent.run(env0,start=(0,0),goal=(3,0),max_steps=12)
    assert r0['solved']
    first=agent.active_regime
    env1=RegimeEnv(2,1)
    r1=agent.run(env1,start=(0,0),goal=(4,5),max_steps=16)
    assert r1['solved']
    assert agent.active_regime is not None
    assert len(agent.bank.hypotheses)>=2
    assert first != agent.active_regime


def test_experiment_budget_is_hard_bounded():
    agent=UnknownRegimeAgent(max_experiment_depth=3,experiment_budget=2)
    env=RegimeEnv(3,2)
    r=agent.run(env,start=(0,0),goal=(10,0),max_steps=20)
    assert r['experiments']<=2


def test_phase63_handoff_contract():
    agent=UnknownRegimeAgent()
    agent.reset_context((5,))
    assert agent._goal==(5,)
    exp=agent.design_experiment((0,), (5,), ('drive', 'probe'))
    assert exp is not None


def test_fail_closed_and_malformed():
    with pytest.raises(ValueError): UnknownRegimeAgent(max_experiment_depth=0)
    with pytest.raises(ValueError): UnknownRegimeBank(min_support=0)
    with pytest.raises(ValueError): UnknownRegimeAgent().reset_context([])
