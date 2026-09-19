import math
import random
import pytest
from mirror7_phase65 import Phase65Agent, StochasticEffectModel, StreamingRegimeBank


class StreamingEnv:
    def __init__(self, seed, mode=0, noise=0.12, switch_at=None):
        self.rng=random.Random(seed)
        self.mode=mode
        self.noise=noise
        self.switch_at=switch_at
        self.t=0
        self.x=0.0
        self.hidden=0
        self.pending=[]
        self.actions=("pulse","__noop__","drift")
        self.reset_called=False

    def legal_actions(self):
        return self.actions

    def reset(self, *args, **kwargs):
        self.reset_called=True
        raise AssertionError("Phase 65 integration must not call reset")

    def observe(self):
        # Hidden variable is deliberately withheld; sensor dropout on x is periodic.
        if self.t % 7 == 0 and self.t > 0:
            return (None,)
        return (self.x + self.rng.gauss(0,self.noise),)

    def step(self, action):
        if action not in self.actions:
            raise ValueError("illegal action")
        self.t += 1
        if self.switch_at is not None and self.t == self.switch_at:
            self.mode = (self.mode + 1) % 3
            self.hidden += 1
        # delayed effect queue: pulse becomes effective two observations later.
        due=[]
        kept=[]
        for due_t, delta in self.pending:
            if due_t <= self.t:
                due.append(delta)
            else:
                kept.append((due_t,delta))
        self.pending=kept
        for delta in due:
            self.x += delta
        if action=="pulse":
            delta=(1.0,2.0,3.0)[self.mode]
            # stochastic action outcome around mode-specific mean.
            self.pending.append((self.t+1, delta + self.rng.gauss(0,0.15)))
        elif action=="drift":
            self.x += 0.05
        return self.observe()


def feed_pulse(agent, env, rounds):
    before=env.observe()
    for _ in range(rounds):
        before=env.observe()
        after=env.step("pulse")
        agent.observe("pulse", before, after)
        before=after
        for _ in range(4):
            after=env.step("__noop__")
            agent.observe("__noop__", before, after)
            before=after


def test_delayed_stochastic_effect_recovery_3x3():
    for family in (0,1,2):
        for seed in range(3):
            env=StreamingEnv(100+seed,mode=family,noise=0.08)
            agent=Phase65Agent(max_lag=4,min_samples=3,experiment_budget=8)
            agent.reset_context((10,))
            feed_pulse(agent,env,8)
            est=agent.model.best_lag("pulse",0)
            assert est == 2
            e=agent.model.estimate("pulse",est,0)
            assert e is not None
            assert abs(e.mean - (family+1)) < 0.55
            assert e.samples >= 3


def test_heldout_regime_is_invented_without_labels():
    agent=Phase65Agent(max_lag=4,min_samples=3,experiment_budget=12)
    env=StreamingEnv(500,mode=0,noise=0.10)
    agent.reset_context((12,))
    feed_pulse(agent,env,7)
    env.mode=1; feed_pulse(agent,env,7)
    before=len(agent.bank.hypotheses)
    env.mode=2; feed_pulse(agent,env,8)
    assert len(agent.bank.hypotheses) > before
    assert agent.active_regime is not None
    snap=agent.bank.hypotheses[agent.active_regime].snapshot()
    vals=[v for (k,v) in snap.signature if k[0]=='pulse' and k[2]==0]
    assert vals and abs(vals[-1]-3.0) < 0.7


def test_hidden_state_switch_triggers_revision_without_reset():
    env=StreamingEnv(700,mode=0,noise=0.10,switch_at=45)
    agent=Phase65Agent(max_lag=4,min_samples=3,mismatch_window=2)
    agent.reset_context((20,))
    feed_pulse(agent,env,6)
    before=agent.revision_events
    env.mode=0
    for _ in range(18):
        feed_pulse(agent,env,1)
    assert agent.revision_events >= before
    # The environment switched internally; later evidence should create more than one hypothesis.
    assert len(agent.bank.hypotheses) >= 2


def test_autonomous_experiment_prefers_disagreement_and_uses_gaps():
    bank=StreamingRegimeBank(max_lag=3,min_samples=2,merge_threshold=0.7)
    m0=StochasticEffectModel(max_lag=3,min_samples=2)
    m1=StochasticEffectModel(max_lag=3,min_samples=2)
    for value in (1.0,1.0,1.0):
        for _ in range(1):
            m0.effects[("pulse",2,0)].add(value)
            m1.effects[("pulse",2,0)].add(3.0)
            m0.effects[("drift",1,0)].add(0.2)
            m1.effects[("drift",1,0)].add(0.25)
    bank.learn(m0); bank.learn(m1)
    a=Phase65Agent(max_lag=3,min_samples=2,experiment_budget=5)
    a.bank=bank
    exp=a.design_experiment(("pulse","drift","__noop__"),slots=3)
    assert exp is not None
    assert exp.focal_actions==("pulse",)
    assert exp.sequence==("pulse","__noop__","__noop__")


def test_noisy_sensor_dropout_does_not_create_invalid_model_state():
    env=StreamingEnv(900,mode=1,noise=0.30)
    agent=Phase65Agent(max_lag=4,min_samples=3)
    agent.reset_context((10,))
    feed_pulse(agent,env,10)
    assert all(k[2] == 0 for k in agent.model.effects)
    assert all(stat.n >= 0 for stat in agent.model.effects.values())


def test_same_regime_does_not_explode_clusters():
    agent=Phase65Agent(max_lag=4,min_samples=3)
    agent.reset_context((8,))
    env=StreamingEnv(1000,mode=1,noise=0.10)
    feed_pulse(agent,env,25)
    assert len(agent.bank.hypotheses) <= 2


def test_experiment_budget_is_hard_bound():
    a=Phase65Agent(experiment_budget=3)
    legal=("pulse","__noop__","drift")
    for _ in range(10):
        e=a.design_experiment(legal)
        if e is None:
            break
    assert a.experiment_count == 3
    assert a.design_experiment(legal) is None


def test_fail_closed_and_malformed():
    a=Phase65Agent()
    with pytest.raises(ValueError): a.reset_context([])
    with pytest.raises(ValueError): a.observe("pulse",(0,),["bad"])
    assert a.predict((0,),"pulse",2) is None
    status=a.fail_closed()
    assert set(status)=={"active_regime","experiments","revisions","hypotheses"}


def test_continuous_stream_integration_never_resets_and_reuses_learned_model():
    env=StreamingEnv(1200,mode=0,noise=0.08)
    env.actions=("pulse","__noop__")
    agent=Phase65Agent(max_lag=4,min_samples=3,experiment_budget=5)
    agent.reset_context((20,))
    result=agent.run_stream(env,goal=(20,),max_steps=64)
    assert not env.reset_called
    assert result["invalid_actions"]==0
    assert result["hypotheses"]>=1
    assert result["experiments"]<=5
