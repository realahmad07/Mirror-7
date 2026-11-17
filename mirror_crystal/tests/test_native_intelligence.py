import pytest
from fractions import Fraction
from mirror_crystal.native_intelligence import (
    PersonalFactMemory,
    ConversationOrgan,
    ArithmeticOrgan,
    IndependentArithmeticEvaluator,
    LogicOrgan,
    WordProblemSolver,
    CuratedKnowledgeOrgan,
    RetrievalGuard,
    values_agree,
)


def test_arithmetic_organ_exact_fractions_and_large_numbers():
    organ = ArithmeticOrgan()
    
    r1 = organ.respond("What is 347 * 29?")
    assert r1 is not None
    assert "10063" in r1.answer
    assert r1.verified is False  # becomes verified when cross-checked by orchestrator
    assert r1.qualification == "computed_result"

    r2 = organ.respond("What is 123456789 * 987654321?")
    assert r2 is not None
    assert "121932631112635269" in r2.answer

    r3 = organ.respond("Compute (17 + 5) * 3 - 8 / 4")
    assert r3 is not None
    assert "64" in r3.answer

    r4 = organ.respond("What is 15% of 80?")
    assert r4 is not None
    assert "12" in r4.answer

    # Division by zero
    r_zero = organ.respond("What is 10 / 0?")
    assert r_zero is not None
    assert "undefined" in r_zero.answer.lower()


def test_values_agree_cross_verification():
    assert values_agree("10063", "$$347 * 29 = 10063$$")
    assert values_agree("64", "\\boxed{64}")
    assert values_agree("121932631112635269", "Result: 121932631112635269")
    assert not values_agree("100", "42")


def test_personal_fact_memory():
    mem = PersonalFactMemory()
    sess = "test_user_sess"

    # 1. Store name
    s1 = mem.try_store(sess, "My name is Ahmad.")
    assert s1 is not None
    assert "Nice to meet you, Ahmad" in s1.answer
    assert s1.qualification == "memory_stored"

    # 2. Recall name
    r1 = mem.try_recall(sess, "What is my name?")
    assert r1 is not None
    assert "Your name is Ahmad." in r1.answer
    assert r1.qualification == "session_memory"

    # 3. Recall unknown age
    r_age = mem.try_recall(sess, "How old am i?")
    assert r_age is not None
    assert "haven't told me your age" in r_age.answer


def test_conversation_organ():
    chat = ConversationOrgan()
    
    # Greeting
    g = chat.respond("Hello, how are you?")
    assert g is not None
    assert "I'm running well" in g.answer or "Hello" in g.answer
    assert g.qualification == "conversational"

    # Identity
    ident = chat.respond("What is your name?")
    assert ident is not None
    assert "Mirror 7" in ident.answer
    assert "Ahmad Hassan" in ident.answer

    # Joke
    j = chat.respond("Tell me a joke")
    assert j is not None
    assert len(j.answer) > 10

    # Capabilities
    c = chat.respond("What can you do?")
    assert c is not None
    assert "CPU-only reasoning engine" in c.answer


def test_logic_organ():
    logic = LogicOrgan()

    # Categorical syllogism
    syl = logic.respond("If all cats are animals and Tom is a cat, is Tom an animal?")
    assert syl is not None
    assert "Yes" in syl.answer
    assert syl.verified is True
    assert syl.qualification == "deductive_proof"

    # Transitive comparison
    comp = logic.respond("Alice is taller than Bob. Bob is taller than Carl. Who is shortest?")
    assert comp is not None
    assert "Carl is the shortest" in comp.answer
    assert comp.verified is True

    # Comparative query
    comp_q = logic.respond("Alice is taller than Bob. Bob is taller than Carl. Is Alice taller than Carl?")
    assert comp_q is not None
    assert "Yes" in comp_q.answer

    # Modus ponens
    prop = logic.respond("If it rains, the ground is wet. It rains. Is the ground wet?")
    assert prop is not None
    assert "Yes" in prop.answer


def test_word_problem_solver():
    solver = WordProblemSolver()

    # Speed, Distance, Time
    p1 = solver.solve("A train travels 180 km in 2.5 hours. What is its average speed?")
    assert p1 is not None
    assert "72" in p1.answer
    assert p1.verified is True

    # Shopping total with relative cost
    p2 = solver.solve("Sofia buys 2 shirts at $7 each and shoes that cost $3 more than a shirt. How much does she spend?")
    assert p2 is not None
    # 2 * 7 = 14; shoes = 7 + 3 = 10; total = 24
    assert "$24" in p2.answer
    assert p2.verified is True

    # Discount & Tax
    p3 = solver.solve("A coat costs $100. It has a 20% discount and 10% tax. What is the final price?")
    assert p3 is not None
    # 100 - 20 = 80; 80 + 8 = 88
    assert "$88" in p3.answer

    # Sum & Difference
    p4 = solver.solve("The sum of two numbers is 50 and their difference is 14. What are the numbers?")
    assert p4 is not None
    assert "32" in p4.answer
    assert "18" in p4.answer


def test_curated_knowledge_organ():
    kb = CuratedKnowledgeOrgan()

    # Geography
    k1 = kb.lookup("What is the capital of France?")
    assert k1 is not None
    assert "Paris" in k1.answer
    assert k1.verified is True

    k1_para = kb.lookup("which city is france's capital")
    assert k1_para is not None
    assert "Paris" in k1_para.answer

    # Literature
    k2 = kb.lookup("Who wrote Romeo and Juliet?")
    assert k2 is not None
    assert "William Shakespeare" in k2.answer

    # Science
    k3 = kb.lookup("Explain why the sky is blue in simple words")
    assert k3 is not None
    assert "Rayleigh scattering" in k3.answer

    k4 = kb.lookup("What is photosynthesis?")
    assert k4 is not None
    assert "photosynthesis" in k4.answer.lower()
    assert "glucose" in k4.answer.lower()

    # Creative
    k5 = kb.lookup("Write a short poem about the ocean")
    assert k5 is not None
    assert "waves" in k5.answer.lower() or "ocean" in k5.answer.lower()


def test_retrieval_guard_filters_spurious_rules():
    # True query: capital of France
    query = "What is the capital of France?"

    # Bad rule from vault: Morse code mentioning Paris as an example sequence
    bad_prompt = "How does Morse code work?"
    bad_answer = "Morse code uses dot and dash patterns to represent letters. The sequence listed earlier represents P a r i s."

    ok, cov, reason = RetrievalGuard.is_relevant(query, bad_prompt, bad_answer)
    assert not ok, f"Expected guard to reject, got ok={ok}, cov={cov}, reason={reason}"
    assert "insufficient_overlap" in reason or "missing_critical" in reason

    # Good rule
    good_prompt = "What is France's capital city?"
    good_answer = "The capital of France is Paris."
    ok_good, cov_good, _ = RetrievalGuard.is_relevant(query, good_prompt, good_answer)
    assert ok_good
    assert cov_good >= 0.40


def test_knowledge_graph_engine(tmp_path):
    from mirror_crystal.native_intelligence import KnowledgeGraphEngine
    kg = KnowledgeGraphEngine(db_path=tmp_path / "test_kg.sqlite")

    # Seed check
    assert kg.total_triples() > 50

    # Query
    caps = kg.query("france", "capital")
    assert caps[0][0] == "Paris"

    currs = kg.query("japan", "currency")
    assert "Yen" in currs[0][0]

    chem = kg.query("gold", "chemical_symbol")
    assert chem[0][0] == "Au"

    # Natural language lookup
    ans_curr = kg.lookup_from_prompt("What is the currency of Japan?")
    assert ans_curr is not None
    assert "Yen" in ans_curr.answer
    assert ans_curr.verified is True

    ans_inv = kg.lookup_from_prompt("Who invented the telephone?")
    assert ans_inv is not None
    assert "Alexander Graham Bell" in ans_inv.answer

    # Dynamic insertion
    added = kg.insert_triple("quantum computer", "inventor", "David Deutsch", category="physics")
    assert added is True
    assert kg.query("quantum computer", "inventor")[0][0] == "David Deutsch"


def test_micro_embedding_engine():
    from mirror_crystal.native_intelligence import MicroEmbeddingEngine
    emb = MicroEmbeddingEngine()

    v1 = emb.embed("What is the capital of France?")
    v2 = emb.embed("Tell me the capital city of France")
    v3 = emb.embed("Write a Python sorting algorithm")

    sim_12 = emb.cosine_similarity(v1, v2)
    sim_13 = emb.cosine_similarity(v1, v3)

    assert sim_12 > 0.65, f"Expected high similarity between paraphrases, got {sim_12}"
    assert sim_12 > sim_13, f"Paraphrase similarity ({sim_12}) should exceed cross-topic ({sim_13})"

    # Ranking test
    candidates = [
        ("Write code to quicksort a list", "code_algo"),
        ("The capital of France is Paris", "geo_fact"),
        ("Solve the quadratic equation x^2 - 4 = 0", "math_algebra"),
    ]
    ranked = emb.rank_candidates("capital city of france", candidates)
    assert ranked[0][2] == "geo_fact"


def test_multi_hop_chaining_engine(tmp_path):
    from mirror_crystal.native_intelligence import KnowledgeGraphEngine, MultiHopChainingEngine
    kg = KnowledgeGraphEngine(db_path=tmp_path / "test_hop_kg.sqlite")
    chainer = MultiHopChainingEngine(kg=kg)

    # 1. "What is the currency of the country whose capital is Paris?"
    # Paris -> France -> Euro
    ans1 = chainer.resolve_chain("What is the currency of the country whose capital is Paris?")
    assert ans1 is not None
    assert "Euro" in ans1.answer
    assert "France" in ans1.answer
    assert ans1.verified is True
    assert len(ans1.steps) == 2

    # 2. "What is the continent of the country whose capital is Tokyo?"
    # Tokyo -> Japan -> Asia
    ans2 = chainer.resolve_chain("What is the continent of the country whose capital is Tokyo?")
    assert ans2 is not None
    assert "Asia" in ans2.answer
    assert "Japan" in ans2.answer


def test_document_auto_crystallizer_engine(tmp_path):
    from mirror_crystal.native_intelligence import KnowledgeGraphEngine, DocumentAutoCrystallizerEngine
    kg = KnowledgeGraphEngine(db_path=tmp_path / "test_doc_kg.sqlite")
    crystallizer = DocumentAutoCrystallizerEngine(kg=kg)

    sample_doc = """# Quantum Computing Overview
The capital of Canada is Ottawa.
Jane Austen wrote Pride and Prejudice.

```python
def add_five(x):
    return x + 5
```
"""
    report = crystallizer.crystallize_text(sample_doc, title="TestDoc")
    assert report["ok"] is True
    assert report["triples_added"] >= 2
    assert report["code_snippets_verified"] == 1

    # Check that Canadian capital triple was extracted and queryable
    res = kg.query("canada", "capital")
    assert len(res) > 0
    assert "Ottawa" in res[0][0]

