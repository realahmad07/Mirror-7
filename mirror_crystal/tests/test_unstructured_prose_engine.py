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


def test_four_frontier_organs_knapsack_testsynth_contradiction_and_faulttree():
    router = NaturalLanguageUOLRouter()

    # 1. Exact 0/1 Knapsack Budget & Resource Allocation Optimizer (B + C beats greedy A!)
    knap = router.route(
        "Optimize budget $50000 across Project A (cost $35000, profit $40000), "
        "Project B (cost $25000, profit $28000), Project C (cost $25000, profit $28000)"
    )
    assert knap.matched is True
    assert knap.output == 56000.0
    assert "Project B, Project C" in knap.explanation

    # 2. White-Box AST Boundary & Edge-Case Unit-Test Synthesizer (catches ZeroDivisionError!)
    ts = router.route(
        "Generate edge-case unit tests for:\n"
        "def safe_divide(a, b):\n"
        "    return a / b\n"
    )
    assert ts.matched is True
    assert "ZeroDivisionError" in ts.explanation
    assert "assert safe_divide(10, 2) == 5.0" in ts.explanation

    # 3. Multi-Document Contradiction & Fact-Check Detector
    c1 = router.route(
        "Check contradiction: Doc A: The Contractor shall deliver the payment module within 45 days. "
        "vs Doc B: The Contractor shall deliver the payment module within 30 days."
    )
    assert c1.matched is True
    assert c1.output is True
    assert "CONTRADICTION DETECTED (Numeric Value Conflict)" in c1.explanation

    c2 = router.route(
        "Check contradiction: Doc A: Remote work is permitted for all engineers. "
        "vs Doc B: Remote work is not permitted for all engineers."
    )
    assert c2.matched is True
    assert c2.output is True
    assert "Polarity / Negation Conflict" in c2.explanation

    # 4. Causal Fault-Tree Root-Cause Diagnostician
    ft = router.route(
        "Diagnose root cause: Server CPU is at 95%, RAM is normal, and DB queries are taking 500ms latency."
    )
    assert ft.matched is True
    assert "Missing Database Index" in ft.explanation
    assert "EXPLAIN ANALYZE" in ft.explanation


def test_antigravity_science_plugins_chemistry_calculus_arxiv_and_uniprot():
    router = NaturalLanguageUOLRouter()

    # 1. PubChem-Plugin Distilled Drug Blueprint & Lipinski Rule-of-5
    pc = router.route("Check PubChem properties and Lipinski drug-likeness for Aspirin")
    assert pc.matched is True
    assert "CID 2244" in pc.explanation
    assert "C9H8O4" in pc.explanation
    assert "Lipinski Rule-of-5: PASS" in pc.explanation

    # 2. Universal IUPAC Molar Mass with Nested Parentheses + Acid-Base pH
    mm = router.route("Calculate molar mass of Ca(OH)2")
    assert mm.matched is True
    assert mm.output == 74.092

    ph = router.route("Calculate pH for H+ concentration 0.001 M")
    assert ph.matched is True
    assert ph.output == 3.0

    # 3. Symbolic Calculus (Derivative & Definite Integral)
    diff = router.route("Differentiate 3x^3 - 5x^2 + 4x - 7 at x=2")
    assert diff.matched is True
    assert diff.output == 20.0
    assert "9*x^2 - 10*x + 4" in diff.explanation

    integ = router.route("Integrate 3x^2 + 2x from 0 to 4")
    assert integ.matched is True
    assert integ.output == 80.0

    # 4. Relativistic Lorentz Factor & arXiv Quantum Entanglement
    lor = router.route("Calculate Lorentz factor for velocity 0.8c")
    assert lor.matched is True
    assert round(float(lor.output), 4) == 1.6667

    arx = router.route("Explain quantum entanglement according to arXiv literature")
    assert arx.matched is True
    assert "quant-ph/0702225v2" in arx.explanation

    # 5. UniProtKB Distilled Protein Blueprint (P04637 / TP53) & Genomics Codon Translation
    up = router.route("Lookup UniProt P04637 TP53 protein function")
    assert up.matched is True
    assert up.output == 393.0
    assert "99.994% storage saved" in up.explanation

    rna = router.route("Translate RNA of AUGGCCUAA")
    assert rna.matched is True
    assert "Met-Ala-Stop" in rna.explanation


def test_12_categorized_conversation_domains_equation_balancer_fda_and_clinvar():
    router = NaturalLanguageUOLRouter()

    # 1. Universal Chemical Equation Matrix Balancer + Stoichiometry
    b1 = router.route("Balance chemical equation: Fe + O2 -> Fe2O3")
    assert b1.matched is True
    assert "4Fe + 3O2 -> 2Fe2O3" in b1.explanation

    b2 = router.route("Balance equation: C3H8 + O2 -> CO2 + H2O")
    assert b2.matched is True
    assert "C3H8 + 5O2 -> 3CO2 + 4H2O" in b2.explanation

    st = router.route("How many grams in 4 moles of H2O?")
    assert st.matched is True
    assert st.output == 72.06

    # 2. openFDA Drug Safety & ClinVar Pathogenic Variants
    fda = router.route("Check openFDA side effects and safety warning for Metformin")
    assert fda.matched is True
    assert "lactic acidosis" in fda.explanation

    cv = router.route("How many pathogenic variants does BRCA1 have in ClinVar?")
    assert cv.matched is True
    assert cv.output == 14336.0

    # 3. 12-Domain Categorized General Conversation + Cross-Domain Blending (Loop Iteration 2)
    d_blend = router.route(
        "Yaar mujhe freelancing mein clients nahi mil rahe aur is wajah se bohot anxiety aur overthinking ho rahi hai"
    )
    assert d_blend.matched is True
    assert "Cross-Domain Synthesized Blueprint" in d_blend.explanation
    assert "Proof-of-Work" in d_blend.explanation
    assert "Cognitive Reset" in d_blend.explanation

    d_phil = router.route("Allama Iqbal ki Khudi aur Stoicism ka falsafa kya sikhata hai?")
    assert d_phil.matched is True
    assert "Khudi" in d_phil.explanation

    d_hist = router.route("Ibn Khaldun ka Asabiyyah aur Islamic Golden Age ke scientists ka kya kirdar tha?")
    assert d_hist.matched is True
    assert "Al-Khwarizmi" in d_hist.explanation


def test_socratic_coach_document_composer_remaining_plugins_and_service_status():
    from mirror7_backend.service import BackendService

    router = NaturalLanguageUOLRouter()

    # 1. Multi-Turn Socratic Personal Coach (Name + Project + Sub-Problem Continuity)
    t1 = router.route("Mera naam Ahmad hai aur main AI startup bana raha hoon", session_id="socratic_1")
    assert t1.matched is True
    assert "Ahmad bhai" in t1.explanation
    assert "AI startup" in t1.explanation

    t2 = router.route("Lakin marketing aur sales samajh nahi aa rahi", session_id="socratic_1")
    assert t2.matched is True
    assert "Ahmad bhai" in t2.explanation
    assert "AI startup" in t2.explanation
    assert "marketing aur sales" in t2.explanation

    # 2. Professional Document, Email, Upwork Proposal & JSON Schema Composer
    prop = router.route("Write an Upwork proposal for a Python FastAPI backend project")
    assert prop.matched is True
    assert "Python FastAPI backend project" in prop.explanation
    assert "48 hours" in prop.explanation

    email = router.route("Boss ko sick leave ki email likho")
    assert email.matched is True
    assert "Subject: Leave Notification" in email.explanation

    jschema = router.route("Generate JSON schema for an e-commerce order API")
    assert jschema.matched is True
    assert '"$schema": "https://json-schema.org/draft/2020-12/schema"' in jschema.explanation

    # 3. Remaining 4 Antigravity Plugins (ChEMBL, Reactome, ClinicalTrials.gov, Context7)
    chembl = router.route("Lookup ChEMBL CHEMBL203 EGFR target bioactivity and IC50")
    assert chembl.matched is True
    assert "Osimertinib" in chembl.explanation

    reactome = router.route("Explain Reactome R-HSA-69278 mitotic cell cycle pathway")
    assert reactome.matched is True
    assert "Cyclin-CDK" in reactome.explanation

    ct = router.route("Explain ClinicalTrials.gov clinical trial phases I II III")
    assert ct.matched is True
    assert "Phase III" in ct.explanation

    c7 = router.route("Show Context7 FastAPI blueprint and SQLite WAL best practices")
    assert c7.matched is True
    assert "PRAGMA journal_mode=WAL" in c7.explanation

    # 4. BackendService crystal_coverage_map telemetry verification
    svc = BackendService()
    st = svc.crystal_coverage_map()
    assert st["uses_neural_parameters"] is False
    assert st["uses_third_party_llm"] is False
    assert st["conversation_domains_count"] == 12
    assert "chembl-database" in st["antigravity_science_plugins_distilled"]


def test_open_ended_concept_lattice_sublinear_trigram_and_cross_synthesis(tmp_path):
    from mirror_crystal.open_ended_concept_lattice import OpenEndedConceptLatticeEngine

    router = NaturalLanguageUOLRouter()
    router.concept_lattice.close()
    router.concept_lattice = OpenEndedConceptLatticeEngine(vault_path=tmp_path / "compile_iso.dcrystal")

    try:
        # 1. Messy Roman-Urdu with typo ('blokchain', 'kese krta h', 'nuqsan kia h') -> resolved via O(1) trigrams + fuzzy intents!
        r1 = router.route("yaar blokchain asal mein kaam kese krta h or iska nuqsan kia h?")
        assert r1.matched is True
        assert "Merkle trees" in r1.explanation
        assert "Throughput and latency overhead" in r1.explanation

        # 2. Cross-Concept Compositional Synthesis (Blockchain x Cryptography)
        r2 = router.route("How do blockchain and cryptography work together?")
        assert r2.matched is True
        assert "Cross-Concept Lattice Synthesis: Blockchain & Distributed Ledger x Modern Cryptography" in r2.explanation

        # 3. Live Unsupervised Concept Compilation + Immediate Open-Ended Query (isolated from production vault)
        c_comp = router.route(
            "Compile concept WebAssembly: "
            "WebAssembly (Wasm) is a portable binary instruction format for stack-based virtual machines. "
            "Wasm works by executing pre-compiled near-native bytecode inside a memory-safe sandboxed linear memory. "
            "It enables high-performance browser and edge-cloud applications across C++, Rust, and Go. "
            "However, Wasm has limited direct DOM access and requires host-binding marshaling overhead."
        )
        assert c_comp.matched is True
        assert "Compiled 5-Facet ConceptNode 'webassembly'" in c_comp.explanation

        r3 = router.route("WebAssembly kaam kaise karta hai aur iska fayda aur nuqsan kya hai?")
        assert r3.matched is True
        assert "sandboxed linear memory" in r3.explanation
        assert "DOM access" in r3.explanation
    finally:
        router.concept_lattice.close()


def test_bulk_harvested_3200plus_multisource_vault_and_offline_harvester(tmp_path):
    import time
    from mirror7_backend.service import BackendService
    from mirror_crystal.bulk_concept_harvester import run_bulk_concept_harvest

    router = NaturalLanguageUOLRouter()

    # 1. Production .dcrystal vault holds >= 10,000 (10,494) multi-domain ConceptNodes
    assert router.concept_lattice.concept_count() >= 10000
    assert len(router.concept_lattice.domain_distribution()) >= 20

    # 1b. Zero-Overwrite & 3-Layer Canonical Deduplication Guarantee
    from mirror_crystal.open_ended_concept_lattice import ConceptNode, OpenEndedConceptLatticeEngine
    iso_lat = OpenEndedConceptLatticeEngine(vault_path=tmp_path / "no_overwrite.dcrystal")
    orig_def = iso_lat._nodes["blockchain"].definition
    dedup_stats = iso_lat.register_nodes_no_overwrite(
        [
            ConceptNode(
                concept_id="blockchain",
                title="Weak Overwrite",
                category="general_knowledge",
                aliases=("blockchain",),
                definition="Weak auto stub trying to overwrite seed blockchain.",
                mechanism="None",
                applications="None",
                tradeoffs="None",
                provenance="wikipedia:mediawiki_api:en",
            ),
            ConceptNode(
                concept_id="blockchains",
                title="Blockchains Plural Duplicate",
                category="computer_science",
                aliases=("blockchains",),
                definition="Plural duplicate that canonical_concept_key must block.",
                mechanism="Duplicate",
                applications="Duplicate",
                tradeoffs="Duplicate",
                provenance="wikipedia:mediawiki_api:en",
            ),
        ],
        persist=True,
    )
    assert dedup_stats["overwrites_prevented"] == 1
    assert dedup_stats["skipped_canonical_duplicate"] == 1
    assert dedup_stats["inserted_new"] == 0
    assert iso_lat._nodes["blockchain"].definition == orig_def

    # 2. Vault telemetry & domain listing commands
    st_res = router.route("Concept lattice status")
    assert st_res.matched is True
    assert "Total 5-Facet ConceptNodes:" in st_res.explanation

    dom_res = router.route("List concepts in physics")
    assert dom_res.matched is True
    assert "Concept Lattice Domain 'physics'" in dom_res.explanation

    # 3. Cross-source open-ended queries (MDN Web Docs, Official Python Docs, Wikipedia Vital, Curated, Cross-Comparison)
    cors = router.route("Explain Cross-Origin Resource Sharing and its tradeoffs")
    assert cors.matched is True
    assert "cross-site" in cors.explanation.casefold()

    py_async = router.route("Explain Python asyncio")
    assert py_async.matched is True
    assert "async/await" in py_async.explanation.casefold()

    cmp_dk = router.route("Compare Docker and Kubernetes")
    assert cmp_dk.matched is True
    assert "Cross-Concept Comparison" in cmp_dk.explanation

    # 4. Verify sub-linear O(1) lookup latency (< 15 ms per query even at 3,200+ concepts)
    t0 = time.perf_counter()
    for q in ("What is Raft consensus?", "Explain CRISPR gene editing", "How does Fourier transform work?"):
        res = router.concept_lattice.route_open_ended(q)
        assert res.matched is True
    avg_ms = ((time.perf_counter() - t0) * 1000.0) / 3.0
    assert avg_ms < 15.0

    # 5. Backend telemetry exposes live concept lattice scale
    svc = BackendService()
    cov = svc.crystal_coverage_map()
    assert cov["open_ended_concept_lattice"]["concept_nodes"] >= 1000
    assert cov["open_ended_concept_lattice"]["combinatorial_cross_pairs"] >= 499_500

    # 6. Offline deterministic harvester smoke test on isolated tmp vault
    tmp_vault = tmp_path / "test_harvest.dcrystal"
    rep = run_bulk_concept_harvest(
        vault_path=tmp_vault,
        include_live_wikipedia=False,
        include_live_arxiv=False,
        include_python_docs=True,
    )
    assert rep.total_unique_concepts >= 750
    assert rep.total_domains >= 20


def test_multi_shard_discovery_zero_dup_quality_gate_and_slas(tmp_path):
    import hashlib
    import sqlite3
    import statistics
    import time
    import tracemalloc
    import pytest
    from mirror_crystal.open_ended_concept_lattice import (
        ConceptNode,
        OpenEndedConceptLatticeEngine,
        _VAULT_SNAPSHOT_CACHE,
        canonical_concept_key,
        content_fingerprint,
        provenance_tier_rank,
    )

    # 1. Verify Phase-3 provenance tier ordering
    assert provenance_tier_rank("mirror:concept_lattice:v1") == 5.0
    assert provenance_tier_rank("mirror:auto_compiled_concept") == 5.0
    assert provenance_tier_rank("mirror:domain_agent_harvester:v2") == 4.0
    assert provenance_tier_rank("browser:mdn:cors") == 3.0
    for p28 in ("mitre:attack:T1059", "owasp:cheatsheet:ssrf", "cncf:glossary:mesh", "mdn:glossary:cors", "nist:csrc:zero-trust", "ietf:rfc:9114"):
        assert provenance_tier_rank(p28) == 2.8
    for p25 in ("ols:go:0001", "chembl:CHEMBL25", "pubchem:2244", "uniprot:P04637", "reactome:R-HSA-1", "interpro:IPR001", "openfda:drug", "clinicaltrials:NCT1", "clinvar:BRCA1", "openalex:T101", "schema.org:Dataset", "mesh:D001", "hf:sciq:1"):
        assert provenance_tier_rank(p25) == 2.5
    assert provenance_tier_rank("docs:python:asyncio") == 2.0
    assert provenance_tier_rank("wikipedia:mediawiki_api:en") == 2.0
    assert provenance_tier_rank("arxiv:2401.00001") == 1.0

    # 2. Isolated primary vault without concept_shards/ must load 0 stray shards
    primary_path = tmp_path / "primary_vault.dcrystal"
    eng = OpenEndedConceptLatticeEngine(vault_path=primary_path)
    assert eng.shard_telemetry()["shards_count"] == 0
    eng.register_nodes_bulk(
        [
            ConceptNode(
                concept_id="raft consensus",
                title="Raft Consensus Algorithm",
                category="computer_science",
                aliases=("raft consensus", "raft protocol"),
                definition="Raft is an understandable distributed consensus algorithm that manages a replicated log across fault-tolerant server clusters.",
                mechanism="Elects a single leader per term via randomized election timeouts and replicates state-machine log entries through AppendEntries RPC quorums.",
                applications="Used in etcd, Consul, CockroachDB, and TiKV for strongly consistent metadata coordination and leader election.",
                tradeoffs="Requires a strict majority quorum (2f+1 nodes) and suffers write latency proportional to wide-area network round-trip times.",
                provenance="mirror:domain_agent_harvester:v1",
            )
        ],
        persist=True,
    )
    eng.close()

    # Re-open primary and record byte-exact SHA-256 hash to prove 0 primary modifications
    primary_sha_before = hashlib.sha256(primary_path.read_bytes()).hexdigest()
    eng = OpenEndedConceptLatticeEngine(vault_path=primary_path)
    assert eng.concept_count() == 11  # 10 seeds + raft consensus
    assert eng.shard_telemetry()["shards_count"] == 0

    # Primary vault cannot be targeted by admit_to_shard
    with pytest.raises(ValueError):
        eng.admit_to_shard(primary_path, [])

    shard_dir = tmp_path / "concept_shards"
    shard1_path = shard_dir / "shard_github_engineering.dcrystal"
    shard2_path = shard_dir / "shard_public_datasets.dcrystal"

    # 3. Admit Shard 1 with valid nodes + duplicates against primary + low-quality junk
    shard1_candidates = [
        # Valid high-quality MITRE / CNCF / OWASP concepts
        ConceptNode(
            concept_id="ebpf telemetry",
            title="eBPF Kernel Telemetry",
            category="cybersecurity_networking",
            aliases=("ebpf telemetry", "extended berkeley packet filter"),
            definition="eBPF is an in-kernel virtual machine technology that runs sandboxed bytecode programs attached to Linux kernel hooks safely.",
            mechanism="A static in-kernel verifier proves memory safety and bounded execution before JIT-compiling bytecode attached to tracepoints, kprobes, or XDP hooks.",
            applications="High-performance observability, Cilium container networking, runtime intrusion detection, and zero-overhead syscall profiling.",
            tradeoffs="Strict verifier complexity limits loop depth and stack size, while older kernel versions lack CO-RE BTF portability features.",
            provenance="cncf:glossary:ebpf",
        ),
        ConceptNode(
            concept_id="ssrf defense",
            title="Server-Side Request Forgery Defense",
            category="cybersecurity_networking",
            aliases=("ssrf defense", "ssrf prevention"),
            definition="Server-Side Request Forgery defense prevents attackers from forcing a backend server to issue unauthorized internal network requests.",
            mechanism="Combines strict egress allowlists, DNS resolution verification blocking RFC1918 and link-local metadata IPs, and IMDSv2 session tokens.",
            applications="Hardening webhook fetchers, PDF renderers, cloud microservices, and API gateways against cloud metadata exfiltration.",
            tradeoffs="Strict egress proxies add deployment friction and must guard against DNS rebinding and HTTP redirect bypasses.",
            provenance="owasp:cheatsheet:ssrf-prevention",
        ),
        # Duplicate 1: Exact primary concept_id
        ConceptNode(
            concept_id="raft consensus",
            title="Raft Consensus Duplicate",
            category="computer_science",
            aliases=("raft consensus",),
            definition="A completely separate description trying to overwrite or duplicate an existing primary concept identifier cleanly.",
            mechanism="Uses quorum heartbeats and leader terms across distributed cluster nodes to synchronize state machines.",
            applications="Distributed key-value coordination, lock services, and strongly consistent configuration storage.",
            tradeoffs="Requires odd node counts and degrades under network partitions that split majority quorums.",
            provenance="cncf:glossary:raft",
        ),
        # Duplicate 2: Canonical plural duplicate of primary seed 'blockchain'
        ConceptNode(
            concept_id="blockchains",
            title="Blockchains",
            category="computer_science",
            aliases=("blockchains",),
            definition="Plural canonical variant of blockchain that must be blocked globally across primary and all concept shards.",
            mechanism="Chains cryptographic block digests together across distributed peer networks using consensus proofs.",
            applications="Financial ledgers, smart-contract execution, and decentralized asset registries.",
            tradeoffs="Higher latency and storage overhead than centralized transactional database systems.",
            provenance="cncf:glossary:blockchains",
        ),
        # Duplicate 3: Alias duplicate of primary 'raft protocol'
        ConceptNode(
            concept_id="raft protocol",
            title="Raft Protocol",
            category="computer_science",
            aliases=("raft protocol",),
            definition="Alias collision attempting to re-register an already claimed surface alias belonging to primary Raft Consensus.",
            mechanism="Leader election and log replication through serialized RPC heartbeats across replicated nodes.",
            applications="Cluster metadata coordination and fault-tolerant consensus replication.",
            tradeoffs="Majority quorum requirement limits availability during severe asymmetric network splits.",
            provenance="cncf:glossary:raft-protocol",
        ),
        # Duplicate 4: 64-bit content_fp collision with primary 'blockchain'
        ConceptNode(
            concept_id="unique_ledger_clone",
            title="Unique Ledger Clone",
            category="computer_science",
            aliases=("unique_ledger_clone",),
            definition=eng._nodes["blockchain"].definition,
            mechanism="Uses cryptographic block chains and distributed peer verification across participating validator nodes.",
            applications="Tamper-evident financial settlement and verifiable audit logging across organizations.",
            tradeoffs="Throughput limits and immutable state growth across full archival network participants.",
            provenance="cncf:glossary:clone",
        ),
        # Low-quality rejects:
        ConceptNode(
            concept_id="short_stub_item",
            title="Short Stub",
            category="computer_science",
            aliases=("short_stub_item",),
            definition="Too short stub.",
            mechanism="Minimal stub mechanism text.",
            applications="Minimal stub applications text.",
            tradeoffs="Minimal stub tradeoffs text.",
            provenance="mdn:glossary:stub",
        ),
        ConceptNode(
            concept_id="mcq_junk_item",
            title="MCQ Junk Concept",
            category="computer_science",
            aliases=("mcq_junk_item",),
            definition="Which of the following protocols guarantees reliable delivery? All of the above are correct answers.",
            mechanism="Select option A or option B based on textbook exam review questions and trivia rules.",
            applications="Multiple-choice assessment banks for introductory networking students.",
            tradeoffs="Contains uncleaned quiz phrasing rather than factual encyclopedic knowledge.",
            provenance="hf:sciq:bad1",
        ),
        ConceptNode(
            concept_id="disambig_junk_item",
            title="Mercury (disambiguation)",
            category="astronomy_space",
            aliases=("disambig_junk_item",),
            definition="Mercury may refer to: the innermost planet of the Solar System, the chemical element Hg, or the Roman messenger deity.",
            mechanism="Disambiguation listing separating homonymous encyclopedia article titles across unrelated domains.",
            applications="Navigation index for ambiguous single-word queries in general encyclopedias.",
            tradeoffs="Provides zero actionable technical mechanism or domain-specific synthesis value.",
            provenance="wikipedia:mediawiki_api:en",
        ),
        ConceptNode(
            concept_id="markdown_garbage_item",
            title="Markdown Garbage",
            category="software_engineering",
            aliases=("markdown_garbage_item",),
            definition="See [MDN HTTP Guide](https://developer.mozilla.org/en-US/docs/Web/HTTP) and {{Glossary('CORS')}} for uncleaned syntax.",
            mechanism="Contains raw markdown links, macros, and unstripped template syntax inside concept prose.",
            applications="Demonstrates strict quality gate rejection of uncleaned documentation scrapes.",
            tradeoffs="Raw markup pollutes synthesized answers and degrades readability for downstream users.",
            provenance="mdn:glossary:bad",
        ),
        ConceptNode(
            concept_id="low_explicit_score",
            title="Low Explicit Quality Score",
            category="software_engineering",
            aliases=("low_explicit_score",),
            definition="A grammatically valid definition exceeding fifty characters whose explicit quality score falls below the shard threshold.",
            mechanism="Evaluates minimum quality threshold enforcement when upstream harvesters flag borderline records.",
            applications="Ensuring strict quality-over-quantity curation across all admitted multi-shard vaults.",
            tradeoffs="Trades raw concept volume for higher precision and cleaner synthesis outputs.",
            provenance="openalex:T999",
            quality_score=0.55,
        ),
        # QA adversarial edge cases: relative markdown link, unlisted HTML5 tags, subject-led deprecation
        ConceptNode(
            concept_id="rel_md_link_junk",
            title="Relative Markdown Link",
            category="software_engineering",
            aliases=("rel_md_link_junk",),
            definition="See the [Service Mesh](service-mesh) glossary entry for detailed sidecar proxy routing behavior in clusters.",
            mechanism="Demonstrates unstripped relative markdown anchor syntax inside prose.",
            applications="Quality gate test for relative markdown links without leading slash or extension.",
            tradeoffs="Raw markdown syntax degrades clean prose synthesis.",
            provenance="cncf:glossary:bad-rel",
        ),
        ConceptNode(
            concept_id="html5_kbd_junk",
            title="HTML5 Tag Leakage",
            category="software_engineering",
            aliases=("html5_kbd_junk",),
            definition="Press <kbd>Ctrl</kbd> with <var>x</var> inside <details><summary>shortcuts</summary></details> to interrupt execution.",
            mechanism="Demonstrates unstripped HTML5 inline/semantic tags from MDN or OWASP pages.",
            applications="Quality gate test for extended HTML5 markup filtering.",
            tradeoffs="Unescaped HTML tags corrupt plain-text and markdown output surfaces.",
            provenance="mdn:glossary:bad-html",
        ),
        ConceptNode(
            concept_id="subject_deprecated_junk",
            title="SSLv2 Transport",
            category="cybersecurity_networking",
            aliases=("subject_deprecated_junk",),
            definition="SSLv2 is deprecated due to severe cryptographic weaknesses including DROWN attacks and truncated MAC protection.",
            mechanism="Legacy transport encryption handshake replaced by modern TLS 1.2 and TLS 1.3 protocols.",
            applications="Historical protocol study only; prohibited in modern production systems.",
            tradeoffs="Vulnerable to downgrade and padding oracle exploits.",
            provenance="ietf:rfc:6176",
        ),
    ]

    st1 = eng.admit_to_shard(shard1_path, shard1_candidates, min_quality=0.78, vacuum=True)
    assert st1["inserted_new"] == 2
    assert st1["skipped_existing_primary_or_shard"] == 1
    assert st1["skipped_canonical_duplicate"] == 1
    assert st1["skipped_alias_duplicate"] == 1
    assert st1["skipped_content_duplicate"] == 1
    assert st1["rejected_low_quality"] == 8
    assert st1["shard_node_count"] == 2
    assert st1["shard_disk_mb"] >= 0.0
    # Primary vault must be byte-for-byte untouched!
    assert hashlib.sha256(primary_path.read_bytes()).hexdigest() == primary_sha_before

    # Verify shard1 is in self-contained journal_mode='delete' AND idempotent 0-insert re-run preserves it!
    shard1_sha_before_rerun = hashlib.sha256(shard1_path.read_bytes()).hexdigest()
    c_chk = sqlite3.connect(str(shard1_path))
    assert str(c_chk.execute("PRAGMA journal_mode").fetchone()[0]).casefold() == "delete"
    c_chk.close()

    st1_rerun = eng.admit_to_shard(shard1_path, shard1_candidates, min_quality=0.78, vacuum=True)
    assert st1_rerun["inserted_new"] == 0
    assert hashlib.sha256(shard1_path.read_bytes()).hexdigest() == shard1_sha_before_rerun
    c_chk2 = sqlite3.connect(str(shard1_path))
    assert str(c_chk2.execute("PRAGMA journal_mode").fetchone()[0]).casefold() == "delete"
    c_chk2.close()

    # 4. Admit Shard 2 and verify global cross-shard zero-duplication against Primary + Shard 1
    shard2_candidates = [
        ConceptNode(
            concept_id="crispr prime editing",
            title="CRISPR Prime Editing",
            category="biology_genomics",
            aliases=("crispr prime editing", "prime editing", "pegRNA editing"),
            definition="Prime editing is a search-and-replace genome editing technology that writes targeted DNA edits without double-strand breaks.",
            mechanism="Fuses a Cas9 nickase to an engineered reverse transcriptase guided by a pegRNA that encodes both target site and edit template.",
            applications="Correcting point mutations, small insertions, and deletions in genetic disease research and therapeutic development.",
            tradeoffs="Larger ribonucleoprotein payload complicates viral AAV delivery and editing efficiency varies across cell types.",
            provenance="openalex:T10442",
        ),
        ConceptNode(
            concept_id="federated thermodynamics",
            title="Non-Equilibrium Statistical Mechanics",
            category="physics",
            aliases=("federated thermodynamics", "fluctuation theorem", "jarzynski equality"),
            definition="Non-equilibrium statistical mechanics extends thermodynamic laws to systems driven far from thermal equilibrium via fluctuation theorems.",
            mechanism="Relates microscopic trajectory entropy production ratios to free-energy differences using Jarzynski and Crooks fluctuation identities.",
            applications="Single-molecule biophysics pulling experiments, nanoscale heat engines, and active matter transport modeling.",
            tradeoffs="Exponential averages in free-energy estimators suffer severe finite-sample variance in rare dissipative trajectories.",
            provenance="openalex:T20891",
        ),
        # Cross-shard exact ID duplicate against Shard 1
        ConceptNode(
            concept_id="ebpf telemetry",
            title="eBPF Duplicate in Shard 2",
            category="cybersecurity_networking",
            aliases=("ebpf telemetry",),
            definition="Cross-shard duplicate attempting to re-admit eBPF telemetry into the second public datasets shard.",
            mechanism="Sandboxed kernel hooks executing verified bytecode on Linux networking and tracing events.",
            applications="Runtime security telemetry and cloud-native observability pipelines.",
            tradeoffs="Kernel verifier restrictions on unbounded loops and arbitrary pointer arithmetic.",
            provenance="openalex:T90001",
        ),
        # Cross-shard alias duplicate against Shard 1 ('ssrf prevention')
        ConceptNode(
            concept_id="ssrf prevention",
            title="SSRF Prevention Duplicate",
            category="cybersecurity_networking",
            aliases=("ssrf prevention",),
            definition="Cross-shard alias duplicate attempting to claim an alias already owned by Shard 1 SSRF defense.",
            mechanism="Validates outbound URLs and denies private IP ranges on server-side HTTP client requests.",
            applications="Preventing cloud metadata SSRF attacks in multi-tenant web applications.",
            tradeoffs="Requires continuous maintenance of egress rules and DNS rebinding protections.",
            provenance="openalex:T90002",
        ),
    ]

    st2 = eng.admit_to_shard(shard2_path, shard2_candidates, min_quality=0.78, vacuum=True)
    assert st2["inserted_new"] == 2
    assert st2["skipped_existing_primary_or_shard"] == 1
    assert st2["skipped_alias_duplicate"] == 1
    assert st2["rejected_low_quality"] == 0
    assert hashlib.sha256(primary_path.read_bytes()).hexdigest() == primary_sha_before
    eng.close()

    # 5. Cold-start SLA (< 50 ms) & Peak RAM SLA (< 25 MB) across Primary + 2 auto-discovered shards
    _VAULT_SNAPSHOT_CACHE.clear()
    tracemalloc.start()
    t0 = time.perf_counter()
    fresh = OpenEndedConceptLatticeEngine(vault_path=primary_path)
    cold_ms = (time.perf_counter() - t0) * 1000.0
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_mb = peak_bytes / (1024 * 1024)

    assert cold_ms < 50.0, f"Multi-shard cold start exceeded 50ms: {cold_ms:.2f} ms"
    assert peak_mb < 25.0, f"Multi-shard peak RAM exceeded 25MB: {peak_mb:.2f} MB"

    # 6. Verify unified counts, domains, proxy mapping, FTS5, and telemetry across Primary + Shard 1 + Shard 2
    assert fresh.concept_count() == 15  # 11 primary + 2 shard1 + 2 shard2
    assert len(fresh._nodes) == 15
    assert "raft consensus" in fresh._nodes
    assert "ebpf telemetry" in fresh._nodes
    assert "crispr prime editing" in fresh._nodes
    assert fresh._nodes["ebpf telemetry"].provenance == "cncf:glossary:ebpf"
    assert fresh._nodes["crispr prime editing"].provenance == "openalex:T10442"
    all_items = dict(fresh._nodes.items())
    assert len(all_items) == 15

    # Check global uniqueness across all 3 SQLite files directly
    global_ids: set[str] = set()
    global_cks: set[str] = set()
    global_fps: set[str] = set()
    for db_p in (primary_path, shard1_path, shard2_path):
        c = sqlite3.connect(str(db_p))
        assert c.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        for cid, ck, fp in c.execute("SELECT concept_id, canonical_key, content_fp FROM mirror_concept_lattice").fetchall():
            assert cid not in global_ids
            assert ck not in global_cks
            assert fp not in global_fps
            global_ids.add(cid)
            global_cks.add(ck)
            global_fps.add(fp)
        c.close()

    stel = fresh.shard_telemetry()
    assert stel["shards_count"] == 2
    assert stel["primary_node_count"] == 11
    assert stel["shards_node_count"] == 4
    assert stel["total_node_count"] == 15
    assert len(stel["shards_breakdown"]) == 2

    dist = fresh.domain_distribution()
    assert sum(dist.values()) == 15
    assert dist["cybersecurity_networking"] == 3  # cybersecurity + ebpf telemetry + ssrf defense
    cyber_nodes = fresh.search_by_category("cybersecurity_networking", limit=10)
    assert {nd.concept_id for nd in cyber_nodes} == {"cybersecurity", "ebpf telemetry", "ssrf defense"}
    fts_hits = fresh.search_fts("Jarzynski fluctuation", limit=5)
    assert any(nd.concept_id == "federated thermodynamics" for nd in fts_hits)

    # 7. Cross-shard retrieval, uncached lookup SLA (< 2.0 ms), and OOD epistemic guards
    probes = [
        ("What is Raft consensus?", "replicated log"),
        ("How does Raft consensus work?", "AppendEntries"),
        ("Explain eBPF telemetry and its tradeoffs", "sandboxed bytecode"),
        ("How does CRISPR prime editing work?", "pegRNA"),
        ("Compare Raft consensus and eBPF telemetry", "Cross-Concept Comparison"),
    ]
    latencies_ms: list[float] = []
    for q, expected in probes:
        fresh._lru.clear()
        tq0 = time.perf_counter()
        res = fresh.route_open_ended(q)
        latencies_ms.append((time.perf_counter() - tq0) * 1000.0)
        assert res.matched is True, f"Failed multi-shard query: {q}"
        assert expected in res.response

    assert statistics.median(latencies_ms) < 2.0, f"Uncached multi-shard median lookup >= 2.0ms: {latencies_ms}"

    # Epistemic OOD guards preserved across multi-shard lattice
    assert fresh.route_open_ended("What is my stock price tomorrow?").matched is False
    assert fresh.route_open_ended("Explain why Hamlet hesitated in Act III.").matched is False

    fresh.close()

    # 8. Multi-word shard concepts MUST beat shorter 1-word/2-word primary component concepts (zero shadowing!)
    p_shadow = tmp_path / "shadow_primary.dcrystal"
    s_shadow_dir = tmp_path / "shadow_shards"
    s_eng = OpenEndedConceptLatticeEngine(vault_path=p_shadow, shard_dir=s_shadow_dir)
    s_eng.register_nodes_bulk(
        [
            ConceptNode(
                concept_id="privacy",
                title="Privacy",
                category="law_governance",
                aliases=("privacy",),
                definition="Privacy is the right of individuals to control access to their personal information and communications.",
                mechanism="Enforced through legal consent frameworks, encryption, and data minimization practices.",
                applications="Personal data protection regulations, confidential messaging, and enterprise governance.",
                tradeoffs="Strong privacy controls can complicate telemetry, auditing, and fraud-detection analytics.",
                provenance="wikipedia:mediawiki_api:en",
            ),
            ConceptNode(
                concept_id="risk management",
                title="Risk Management",
                category="business_management",
                aliases=("risk management",),
                definition="Risk management is the structured identification, evaluation, and prioritization of operational uncertainties.",
                mechanism="Quantifies likelihood and impact across threat matrices to allocate mitigation controls efficiently.",
                applications="Financial portfolio hedging, enterprise compliance, and safety-critical engineering.",
                tradeoffs="Excessive risk aversion slows product iteration and increases compliance overhead.",
                provenance="wikipedia:mediawiki_api:en",
            ),
            ConceptNode(
                concept_id="radio",
                title="Radio",
                category="physics",
                aliases=("radio",),
                definition="Radio is the technology of signaling and communicating using modulated electromagnetic radio-frequency waves.",
                mechanism="Transmitters modulate carrier amplitude or frequency and receivers demodulate induced antenna currents.",
                applications="Wireless broadcasting, aviation navigation, radar sensing, and cellular telecommunications.",
                tradeoffs="Shared electromagnetic spectrum suffers interference, attenuation, and regulatory allocation limits.",
                provenance="wikipedia:mediawiki_api:en",
            ),
            ConceptNode(
                concept_id="broadcast",
                title="Broadcast",
                category="computer_science",
                aliases=("broadcast",),
                definition="Broadcast is a one-to-many communication model that transmits a single signal simultaneously to all receivers.",
                mechanism="Delivers packets or RF transmissions across a shared medium without per-recipient point-to-point sessions.",
                applications="Television distribution, ARP discovery on LAN segments, and emergency alert systems.",
                tradeoffs="Broadcast storms consume network bandwidth and lack native per-receiver confidentiality.",
                provenance="wikipedia:mediawiki_api:en",
            ),
        ],
        persist=True,
    )
    adm_sh = s_eng.admit_to_shard(
        s_shadow_dir / "shard_multiword.dcrystal",
        [
            ConceptNode(
                concept_id="privacy risk management",
                title="Privacy Risk Management",
                category="cybersecurity_networking",
                aliases=("privacy risk management",),
                definition="A cross-organizational set of processes for identifying, assessing, and responding to personal privacy risks.",
                mechanism="Applies NIST Privacy Framework risk models to evaluate problematic data actions across system lifecycles.",
                applications="Enterprise privacy engineering, DPIA governance audits, and regulatory compliance programs.",
                tradeoffs="Requires cross-functional coordination across legal, engineering, and security operations teams.",
                provenance="nist:csrc:privacy_risk_management",
            ),
            ConceptNode(
                concept_id="schema_radio_service",
                title="Radio Broadcast Service",
                category="engineering",
                aliases=("schema_radio_service", "radio broadcast service"),
                definition="A structured broadcasting service delivering audio programming over terrestrial or digital radio channels.",
                mechanism="Models broadcast frequency, station callsign, and parent broadcast affiliate metadata using Schema.org types.",
                applications="Media directory indexing, structured search metadata, and broadcast schedule syndication.",
                tradeoffs="Requires accurate regional frequency and timezone synchronization across affiliate networks.",
                provenance="schema.org:class:RadioBroadcastService",
            ),
            # Valid CRL concept mentioning revoked certificates must PASS
            ConceptNode(
                concept_id="certificate revocation list crl",
                title="Certificate Revocation List (CRL)",
                category="cybersecurity_networking",
                aliases=("certificate revocation list crl", "crl"),
                definition="A Certificate Revocation List is a signed list of digital certificates that have been revoked by the issuing CA before expiration.",
                mechanism="Certificate authorities periodically publish ASN.1 DER signed revocation lists fetched by TLS clients during validation.",
                applications="Public key infrastructure revocation checks, enterprise VPN gateways, and code-signing verification.",
                tradeoffs="Large CRL files increase validation latency and introduce freshness windows between publication intervals.",
                provenance="ietf:rfc4949:crl",
            ),
            # 3 deprecated phrasings that MUST be rejected natively
            ConceptNode(
                concept_id="flexvolume_bad",
                title="FlexVolume",
                category="software_engineering",
                aliases=("flexvolume_bad",),
                definition="FlexVolume is a deprecated interface for creating out-of-tree container volume plugins in Kubernetes.",
                mechanism="Invoked host-installed binaries on cluster nodes before CSI replaced it.",
                applications="Legacy Kubernetes storage driver extensions prior to Container Storage Interface adoption.",
                tradeoffs="Requires root filesystem access on nodes and lacks dynamic provisioning support.",
                provenance="cncf:k8s_glossary:flexvolume",
            ),
            ConceptNode(
                concept_id="general_header_bad",
                title="General Header",
                category="cybersecurity_networking",
                aliases=("general_header_bad",),
                definition="General header is an outdated term used to refer to an HTTP header valid in both request and response messages.",
                mechanism="Replaced by representation and request/response context classifications in modern HTTP RFCs.",
                applications="Historical HTTP/1.1 specification terminology no longer used in RFC 9110.",
                tradeoffs="Ambiguous classification compared to modern HTTP semantics.",
                provenance="mdn:glossary:general_header",
            ),
            ConceptNode(
                concept_id="hpkp_bad",
                title="HTTP Public Key Pinning",
                category="cybersecurity_networking",
                aliases=("hpkp_bad",),
                definition="HTTP Public Key Pinning (HPKP) is an obsolete security feature that told web clients to associate specific keys.",
                mechanism="Pinned SPKI hashes via HTTP response headers before browsers removed support.",
                applications="Deprecated TLS anti-impersonation experiment replaced by Certificate Transparency.",
                tradeoffs="High risk of permanent domain bricking if pinned keys are lost or compromised.",
                provenance="mdn:glossary:hpkp",
            ),
        ],
        min_quality=0.78,
        vacuum=True,
    )
    assert adm_sh["inserted_new"] == 3
    assert adm_sh["rejected_low_quality"] == 3

    r_prm = s_eng.route_open_ended("What is privacy risk management?")
    assert r_prm.matched is True
    assert r_prm.concepts_used == ("privacy risk management",)
    assert "cross-organizational" in r_prm.response

    r_rbs = s_eng.route_open_ended("What is Radio Broadcast Service?")
    assert r_rbs.matched is True
    assert r_rbs.concepts_used == ("schema_radio_service",)
    assert "audio programming" in r_rbs.response

    s_eng.close()











