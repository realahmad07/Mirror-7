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
