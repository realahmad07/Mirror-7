"""Tests for Parameter-Free Unstructured Prose Comprehension, Submodular-NCD Summarization,
OpenIE Concept-Graph Ingestion, and Comparative RST Discourse Synthesis."""
from __future__ import annotations

from mirror7_backend import UnifiedCognitiveOrchestrator
from mirror_crystal.nl_uol_router import NaturalLanguageUOLRouter
from mirror_crystal.unstructured_prose_engine import (
    SubmodularNCDSummarizer,
    UnstructuredProseEngine,
    extract_openie_propositions,
)

SAMPLE_LEGAL_CONTRACT = (
    "This Master Software Services Agreement is entered into between Party A (Client) and Party B (Contractor). "
    "The Contractor shall deliver the production payment gateway module within 45 days of the effective date. "
    "The Client agrees to pay $120000 USD in three equal milestone installments upon verified code delivery. "
    "All source code and background background boilerplate notes remain recorded in the project repository archive. "
    "The Contractor must maintain strict confidentiality of all customer financial records for 5 years. "
    "If the Contractor fails to meet the 45-day deadline, a late penalty of 2% per week shall apply up to 10%. "
    "Either party may terminate this Agreement upon 30 days written notice in the event of a material breach. "
    "Disputes arising under this contract shall be resolved through binding arbitration in Lahore."
)


def test_submodular_ncd_legal_contract_summary_english_and_urdu():
    summarizer = SubmodularNCDSummarizer()

    rep_en = summarizer.summarize(SAMPLE_LEGAL_CONTRACT, max_sentences=3, language="en")
    assert rep_en.ok is True
    assert rep_en.uses_neural_parameters is False
    assert rep_en.source_sentence_count == 8
    assert rep_en.selected_sentence_count == 3
    assert rep_en.compression_ratio < 0.65
    assert len(rep_en.clause_breakdown.obligations) >= 2
    assert len(rep_en.clause_breakdown.conditions_and_deadlines) >= 1
    assert len(rep_en.clause_breakdown.financial_and_numeric_terms) >= 2
    assert len(rep_en.clause_breakdown.risks_and_termination) >= 2
    assert "Executive Summary (0-Parameter Submodular-NCD" in rep_en.rendered_summary
    assert "Key Obligations:" in rep_en.rendered_summary

    rep_ur = summarizer.summarize(SAMPLE_LEGAL_CONTRACT, max_sentences=3, language="roman_urdu")
    assert rep_ur.ok is True
    assert "Khulasa (0-Parameter Submodular-NCD Summary" in rep_ur.rendered_summary
    assert "Ahem Zimmedariyan (Obligations):" in rep_ur.rendered_summary
    assert "Khatraat aur Mansookhi (Termination & Liability):" in rep_ur.rendered_summary


def test_comparative_rst_synthesis_mughal_vs_ottoman_and_tcp_vs_udp():
    engine = UnstructuredProseEngine()

    rep_ur = engine.compare_topics("Mughal", "Ottoman", language="roman_urdu")
    assert rep_ur.matched is True
    assert rep_ur.uses_neural_parameters is False
    assert rep_ur.propositions_used == 8
    assert "Mansabdari" in rep_ur.rendered_text
    assert "Devshirme" in rep_ur.rendered_text
    assert "Taqabuli Tajziya" in rep_ur.rendered_text

    rep_tcp = engine.compare_topics("TCP", "UDP", language="en")
    assert rep_tcp.matched is True
    assert "three-way SYN/SYN-ACK/ACK handshake" in rep_tcp.rendered_text
    assert "connectionless" in rep_tcp.rendered_text


def test_live_openie_unstructured_text_ingestion_and_comparison():
    router = NaturalLanguageUOLRouter()

    # Before ingestion: comparing unknown topics must honestly abstain (0% hallucination)
    abstain = router.route("Compare Indus Valley and Mesopotamia")
    assert abstain.matched is False

    # Live 1-shot unstructured text ingestion for Topic 1
    ing1 = router.route(
        "Learn text about Indus Valley: "
        "The Indus Valley Civilization was a Bronze Age urban society in South Asia known for standardized baked-brick cities. "
        "Indus governance operated through civic municipal planning, uniform weights and measures, and advanced drainage networks without monumental royal palaces. "
        "Indus cities achieved high public hygiene, grid-aligned street layouts, and maritime trade with West Asia. "
        "However, late Indus urban centers faced river-course shifts, aridification, and gradual de-urbanization after 1900 BCE."
    )
    assert ing1.matched is True
    assert "prose.openie_ingest" in ing1.rule_ids

    # Live 1-shot unstructured text ingestion for Topic 2
    ing2 = router.route(
        "Learn text about Mesopotamia: "
        "Mesopotamia was an ancient Tigris-Euphrates civilization organized around competing city-states and territorial kingdoms. "
        "Mesopotamian governance operated through centralized temple-palace bureaucracies, cuneiform legal codes like Hammurabi's Code, and canal irrigation taxation. "
        "Mesopotamia pioneered written law, astronomical mathematics, and monumental ziggurat architecture. "
        "However, Mesopotamian agriculture suffered from soil salinization and frequent warfare between rival city-states."
    )
    assert ing2.matched is True

    # Now compare the two newly ingested unstructured topics!
    cmp_res = router.route("Compare Indus Valley and Mesopotamia")
    assert cmp_res.matched is True
    assert "prose.comparative_rst_synthesis" in cmp_res.rule_ids
    assert "standardized baked-brick cities" in cmp_res.explanation
    assert "Hammurabi's Code" in cmp_res.explanation


def test_end_to_end_orchestrator_urdu_contract_summary_and_mughal_ottoman():
    orch = UnifiedCognitiveOrchestrator(
        vault_provider=lambda: None,
        memory_provider=lambda: None,
        reasoning_provider=lambda: None,
        code_provider=lambda: None,
    )

    r1 = orch.ask(f"10-page legal contract ka khulasa Urdu mein likho: {SAMPLE_LEGAL_CONTRACT}")
    assert r1["task_result_verified"] is True
    assert r1["epistemic_tier"] == "verified_task_result"
    assert r1["abstained"] is False
    assert "Khulasa (0-Parameter Submodular-NCD Summary" in r1["answer"]
    assert "$120000 USD" in r1["answer"]

    r2 = orch.ask("Mughal aur Ottoman governance ka muqabla karo")
    assert r2["task_result_verified"] is True
    assert r2["epistemic_tier"] == "verified_task_result"
    assert r2["abstained"] is False
    assert "Mansabdari" in r2["answer"]
    assert "Timar" in r2["answer"]


def test_grounded_contract_and_document_qa_plus_sqlite_persistence(tmp_path):
    vault_file = tmp_path / "openie_test.dcrystal"
    engine = UnstructuredProseEngine(vault_path=vault_file)
    engine.ingest_unstructured_text(SAMPLE_LEGAL_CONTRACT, topic="contract", provenance="master_services_agreement")

    # Reload into a brand-new engine instance from SQLite .dcrystal vault
    reloaded = UnstructuredProseEngine(vault_path=vault_file)
    q1 = reloaded.route_unstructured_prompt("In the contract, what is the late penalty?")
    assert q1 is not None and q1["matched"] is True
    assert "2% per week" in q1["explanation"]

    q2 = reloaded.route_unstructured_prompt("Contract mein arbitration kis shehar mein hogi?")
    assert q2 is not None and q2["matched"] is True
    assert "Lahore" in q2["explanation"]

    # Test local file ingestion
    doc_file = tmp_path / "quantum_computing.txt"
    doc_file.write_text(
        "Quantum computing is a computational paradigm that uses quantum bits (qubits) in superposition. "
        "Quantum processors operate through unitary logic gates and entanglement to accelerate factoring and simulation.",
        encoding="utf-8",
    )
    router = NaturalLanguageUOLRouter()
    r_file = router.route(f"Learn file about Quantum Computing: {doc_file}")
    assert r_file.matched is True
    r_qa = router.route("How do quantum computing processors operate?")
    assert r_qa.matched is True
    assert "unitary logic gates" in r_qa.explanation


def test_counterfactual_what_if_scaling_and_contract_simulation():
    router = NaturalLanguageUOLRouter()

    # Physics scaling in English & Roman Urdu
    c1 = router.route("If mass doubles and velocity triples, by what factor does kinetic energy change?")
    assert c1.matched is True
    assert c1.output == 18.0

    c2 = router.route("Agar mass double aur velocity triple ho jaye to kinetic energy kitne guna barh jaye gi?")
    assert c2.matched is True
    assert c2.output == 18.0

    c3 = router.route("What if force is tripled and area is halved — by what factor does axial stress change?")
    assert c3.matched is True
    assert c3.output == 6.0

    # Contract late penalty counterfactual simulation
    c4 = router.route(
        "What if a $120000 contract has a 2% per week late penalty (capped at 10%) and is delivered 3 weeks late? What is the net payout?"
    )
    assert c4.matched is True
    assert c4.output == 112800.0


def test_counterexample_guided_ast_code_repair():
    router = NaturalLanguageUOLRouter()

    r1 = router.route(
        "Fix this Python code:\ndef factorial(n):\n    return 0 if n <= 1 else n * factorial(n - 1)"
    )
    assert r1.matched is True
    assert "Constant(0 -> 1)" in r1.explanation
    assert "return 1 if n <= 1 else n * factorial(n - 1)" in r1.explanation

    r2 = router.route(
        "Fix this function:\ndef sum_to_n(n):\n    total = 0\n    for i in range(n):\n        total += i\n    return total"
    )
    assert r2.matched is True
    assert "range(n + 1)" in r2.explanation


def test_zero_parameter_tabular_data_analyst():
    router = NaturalLanguageUOLRouter()

    d1 = router.route("Predict next value for series: 100, 150, 200, 250")
    assert d1.matched is True
    assert d1.output == 300.0

    d2 = router.route("Find outliers in data: 120, 135, 150, 165, 490, 180")
    assert d2.matched is True
    assert d2.output == 490.0

    d3 = router.route("Analyze sales data in Urdu: Jan=100, Feb=120, Mar=140, Apr=160, May=180")
    assert d3.matched is True
    assert d3.output == 140.0
    assert "R^2 = 1.0000" in d3.explanation


def test_creative_poetry_story_empathy_and_live_dialogue_feeding(tmp_path):
    router = NaturalLanguageUOLRouter()

    # 1. Poetry in English & Roman Urdu Shayari
    p_en = router.route("Write a poem about autumn in Lahore.")
    assert p_en.matched is True
    assert "Lahore" in p_en.explanation
    assert "leaves" in p_en.explanation

    p_ur = router.route("Lahore ki khizan par shayari likho")
    assert p_ur.matched is True
    assert "Lahore" in p_ur.explanation
    assert "patton" in p_ur.explanation

    # 2. 5-Act Story in Roman Urdu & English
    st_ur = router.route("Dosti ke baare mein kahani sunao")
    assert st_ur.matched is True
    assert "5-Act Kahani" in st_ur.explanation

    # 3. Empathetic Roman-Urdu Conversation & Philosophy
    emp = router.route("Yaar aaj bohot thak gaya hoon, mera mood off hai")
    assert emp.matched is True
    assert "break" in emp.explanation

    phil = router.route("Explain the meaning of consciousness in philosophy.")
    assert phil.matched is True
    assert "Qualia" in phil.explanation

    # 4. Live 1-shot conversational feeding ("Feed: <input> => <reply>")
    f1 = router.route(
        "Feed: chai peeni hai yaar => Garma garam elaichi wali chai tayyar hai, saath mein coding karte hain!"
    )
    assert f1.matched is True
    ans1 = router.route("Yaar aaj chai peeni hai")
    assert ans1.matched is True
    assert "elaichi wali chai" in ans1.explanation

    # 5. Bulk JSONL corpus file feeding
    corpus_file = tmp_path / "custom_dialogue.jsonl"
    corpus_file.write_text(
        '{"prompt": "pakistan ka sab se khoobsurat pahari ilaqa konsa hai", "response": "Hunza aur Skardu apni kudrati khoobsurti aur paharon ke liye lajawab hain!"}\n',
        encoding="utf-8",
    )
    fc = router.route(f"Feed file: {corpus_file}")
    assert fc.matched is True
    ans2 = router.route("Pakistan ka sab se khoobsurat pahari ilaqa konsa hai?")
    assert ans2.matched is True
    assert "Hunza aur Skardu" in ans2.explanation


def test_conversational_blueprint_distillation_eviction_and_unbounded_multiturn():
    router = NaturalLanguageUOLRouter()

    # 1. Feed two analogous examples -> induces 1 generalized <slot_0> blueprint and EVICTS raw data!
    e1 = router.route("Feed: mujhe python seekhna hai => python seekhne ke liye rozana 1 chota project banao!")
    assert e1.matched is True

    e2 = router.route("Feed: mujhe rust seekhna hai => rust seekhne ke liye rozana 1 chota project banao!")
    assert e2.matched is True
    assert "Distilled Generalized Blueprint Rule" in e2.explanation
    assert "EVICTED" in e2.explanation
    assert len(router.creative_engine.blueprint_engine._pending_examples) == 0

    # Now test on a completely UNSEEN topic that was never fed or stored!
    unseen = router.route("mujhe quantum robotics seekhna hai")
    assert unseen.matched is True
    assert "quantum robotics seekhne ke liye rozana 1 chota project banao!" in unseen.explanation.casefold()

    # 2. Unbounded open-topic conversation + 128-byte multi-turn pronoun coreference ('us mein')
    t1 = router.route("Yaar aaj main bakery ki dukan chala kar bohot thak gaya hoon", session_id="sess_bp")
    assert t1.matched is True
    assert "bakery ki dukan" in t1.explanation.casefold()

    # Turn 2 uses pronoun 'Us' without repeating 'bakery ki dukan'
    t2 = router.route("Us mein kamiyabi kaise mile gi?", session_id="sess_bp")
    assert t2.matched is True
    assert "bakery ki dukan" in t2.explanation.casefold()

    # 3. Unbounded English advisory blueprint on arbitrary unseen topic
    t3 = router.route("How can I master underwater cinematography?", session_id="sess_bp2")
    assert t3.matched is True
    assert "underwater cinematography" in t3.explanation.casefold()


def test_four_new_organs_fusion_morphology_distiller_and_goal_planner(tmp_path):
    router = NaturalLanguageUOLRouter()

    # 1. Multi-Intent Heterogeneous Discourse Fusion (Empathy Blueprint + Quantitative Word Problem in 1 prompt!)
    fused = router.route(
        "Yaar aaj main bakery ki dukan chala kar bohot thak gaya hoon; Ali has 800 dollars and gives 500 dollars to his vendor. How many dollars does Ali have left?"
    )
    assert fused.matched is True
    assert fused.output == 300.0
    assert "Multi-Intent Fused Response" in fused.explanation
    assert "bakery ki dukan" in fused.explanation.casefold()

    # 2. Roman-Urdu Morpho-Syntactic Gender Agreement ('parhai' is Feminine -> 'nichor leti hai', 'karobar' is Masculine -> 'nichor leta hai')
    fem = router.route("Yaar aaj main parhai kar ke bohot thak gaya hoon")
    assert fem.matched is True
    assert "nichor leti hai" in fem.explanation

    masc = router.route("Yaar aaj main karobar chala kar bohot thak gaya hoon")
    assert masc.matched is True
    assert "nichor leta hai" in masc.explanation

    # 3. Zero-Bloat Folder & Corpus Blueprint Distiller (60%+ storage savings!)
    folder = tmp_path / "sample_books"
    folder.mkdir()
    (folder / "dialogues.txt").write_text(
        "hamza ko flutter seekhna hai => flutter seekhne ke liye rozana 2 ghantay code likho aur widgets banao!\n"
        "hamza ko golang seekhna hai => golang seekhne ke liye rozana 2 ghantay code likho aur widgets banao!\n"
        "hamza ko docker seekhna hai => docker seekhne ke liye rozana 2 ghantay code likho aur widgets banao!\n"
        "hamza ko kubernetes seekhna hai => kubernetes seekhne ke liye rozana 2 ghantay code likho aur widgets banao!\n",
        encoding="utf-8",
    )
    dist_res = router.route(f"Distill folder: {folder}")
    assert dist_res.matched is True
    assert dist_res.output is not None and dist_res.output >= 50.0
    assert "storage saved!" in dist_res.explanation

    # Verify the distilled blueprint generalizes to an unseen topic from the distilled folder!
    unseen_bp = router.route("hamza ko elixir seekhna hai")
    assert unseen_bp.matched is True
    assert "elixir seekhne ke liye rozana 2 ghantay code likho" in unseen_bp.explanation.casefold()

    # 4. Personal Goal & Habit DAG Planner VM (English & Roman Urdu + multi-turn progress tracking)
    g1 = router.route("Mujhe 40 din mein Python seekhna hai, plan banao", session_id="goal_sess")
    assert g1.matched is True
    assert g1.output == 40.0
    assert "Din 1..10" in g1.explanation

    g2 = router.route("Goal progress: completed phase 3", session_id="goal_sess")
    assert g2.matched is True
    assert g2.output == 75.0
    assert "10 days remaining" in g2.explanation


def test_four_market_intelligence_organs_law_decision_pipeline_and_auditor():
    router = NaturalLanguageUOLRouter()

    # 1. Self-Evolving Symbolic Law Discovery (y = x^2 + 3*x + 1 -> at x=10, y=131)
    law = router.route("Discover law for x=[1, 2, 3, 4], y=[5, 11, 19, 29] and evaluate at x=10")
    assert law.matched is True
    assert law.output == 131.0
    assert "y = x^2 + 3*x + 1" in law.explanation

    # 2. Multi-Criteria Decision & Expected-Utility Pareto Engine
    dec = router.route(
        "Compare Option A (cost $10000, return $50000, success 80%) vs Option B (cost $4000, return $30000, success 90%)"
    )
    assert dec.matched is True
    assert dec.output == 30000.0  # EV(A) = 0.8*50000 - 10000 = $30,000 vs EV(B) = 0.9*30000 - 4000 = $23,000
    assert "Winner: Option A" in dec.explanation

    # 3. Autonomous Multi-Organ Workflow Pipeline Executor (Outlier Clean -> OLS Forecast -> +15% tax)
    pipe = router.route(
        "Clean outliers from [100, 120, 950, 140, 160], predict the next value, and add 15% tax"
    )
    assert pipe.matched is True
    assert pipe.output == 207.0  # clean=[100,120,140,160] -> forecast=180 -> +15% = 207.0

    # 4. Zero-Parameter AST Security, Big-O Complexity & Dead-Code Auditor
    aud = router.route(
        "Audit security and complexity of:\n"
        "def run_query(items, cmd):\n"
        "    eval(cmd)\n"
        "    for x in items:\n"
        "        for y in items:\n"
        "            pass\n"
        "    return 1\n"
        "    z = 99\n"
    )
    assert aud.matched is True
    assert "O(N^2)" in aud.explanation
    assert "eval()" in aud.explanation
    assert "Unreachable / Dead Statements: 1" in aud.explanation





