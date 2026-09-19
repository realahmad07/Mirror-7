import pytest
from .mirror7_phase59 import PredictiveConceptModel

def train_eps():
    return [tuple('ABABCDCDABAB') for _ in range(6)]

def test_progressive_prediction_3x3():
    families=[tuple('ABABABAB'),tuple('ABABCDCDABAB'),tuple('ABCABCABCABC')]
    for level in range(3):
        for seed in range(3):
            eps=[families[level] for _ in range(6)]
            m=PredictiveConceptModel(max_order=3,min_support=3).fit(eps)
            candidates=[]
            for ctx in m.known_contexts():
                p=m.predict(ctx)
                if p.value is not None:
                    candidates.append(p)
            assert candidates and max(p.context_length for p in candidates)>=1
            assert m.known_contexts()

def test_heldout_recombination():
    m=PredictiveConceptModel(max_order=3,min_support=2).fit(train_eps())
    p=m.predict(tuple('ABAB'))
    assert p.value=='C' and p.confidence==1.0
    p2=m.predict(tuple('CDCD'))
    assert p2.value=='A'

def test_longer_context_beats_shorter_context():
    eps=[tuple('XABY') for _ in range(5)]+[tuple('ZABW') for _ in range(5)]
    m=PredictiveConceptModel(max_order=2,min_support=2).fit(eps)
    p=m.predict(('Z','A'))
    assert p.value=='B' and p.context_length==2
    assert m.predict(('B',)).value is None

def test_abstention_on_ambiguity():
    eps=[tuple('ABX') for _ in range(4)]+[tuple('ABY') for _ in range(4)]
    m=PredictiveConceptModel(max_order=2,min_support=2).fit(eps)
    p=m.predict(('A','B'))
    assert p.value is None and p.alternatives==2

def test_insufficient_support_abstains():
    m=PredictiveConceptModel(max_order=3,min_support=3).fit([tuple('ABX')])
    assert m.predict(('A','B')).value is None

def test_unseen_context_backoff():
    m=PredictiveConceptModel(max_order=3,min_support=2).fit([tuple('ABCABC') for _ in range(4)])
    p=m.predict(('Q','B','C'))
    assert p.value=='A' and p.context_length==2

def test_determinism_and_signature():
    a=PredictiveConceptModel(max_order=3,min_support=2).fit(train_eps())
    b=PredictiveConceptModel(max_order=3,min_support=2).fit(train_eps())
    assert a.signature==b.signature and a.predict(('A','B'))==b.predict(('A','B'))

def test_malformed():
    with pytest.raises(ValueError): PredictiveConceptModel(max_order=0)
    with pytest.raises(ValueError): PredictiveConceptModel().fit([])
    with pytest.raises(RuntimeError): PredictiveConceptModel().predict(('A',))
