from .mirror7_phase221_230 import *

def test_221_tokenization():
    g=LanguageGrounder(); g.learn(Utterance("move red",("move",)))
    assert g.ground("move red")

def test_222_grounding():
    g=LanguageGrounder(); g.learn(Utterance("move red",("move",)))
    assert g.ground("please move red").intent=="move"

def test_223_case_invariance():
    g=LanguageGrounder(); g.learn(Utterance("MOVE RED",("move",)))
    assert g.ground("move red").intent=="move"

def test_224_unsupported_abstains():
    g=LanguageGrounder(); g.learn(Utterance("move",("move",)))
    assert g.ground("zzzz") is None

def test_225_ambiguity_abstains():
    g=LanguageGrounder()
    g.learn(Utterance("go",("move",)))
    g.learn(Utterance("go",("stop",)))
    assert g.ground("go") is None

def test_226_allowed_intents():
    g=LanguageGrounder(); g.learn(Utterance("move red",("move",))); g.learn(Utterance("stop red",("stop",)))
    assert g.ground("move red",allowed=("move",)).intent=="move"

def test_227_evidence():
    g=LanguageGrounder(); g.learn(Utterance("move red",("move",)))
    c=g.ground("move red"); assert "move" in c.evidence

def test_228_explanation():
    g=LanguageGrounder(); g.learn(Utterance("move red",("move",)))
    c=g.ground("move red"); assert "move" in g.explain(c)

def test_229_merge():
    a=LanguageGrounder(); b=LanguageGrounder(); b.learn(Utterance("stop",("stop",))); a.merge(b)
    assert a.ground("stop").intent=="stop"

def test_230_fingerprint():
    g=LanguageGrounder(); g.learn(Utterance("move",("move",)))
    assert g.fingerprint("move")==g.fingerprint("MOVE")
