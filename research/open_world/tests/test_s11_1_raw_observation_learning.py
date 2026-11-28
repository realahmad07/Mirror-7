"""Independent Adversarial Tests for Stage S11.1 Raw-Observation Open-World Learning.

Verifies all 9 architectural stages, edge cases, controlled ablations (A1..A6),
isolation/freeze guarantees, and honest gate reporting without weakening any check:

1. Whole-tree AST isolation across `research/open_world/learner/` (0 evaluator/oracle/loader
   imports, 0 GPU/LLM/neural imports, 0 Python 3.12+ stdlib APIs).
2. Stage 2 `RawEvidenceLedger`: exact multi-byte UTF-8 byte spans, SHA-256 hash chain,
   SQLite BEFORE UPDATE / BEFORE DELETE immutability triggers, empty-input handling, and
   `.freeze()` enforcement (`SolverFrozenError`).
3. Stage 3 schema-free extraction: empty inputs, Unicode spans, footnote stripping,
   unsupervised noun/entity/quantity/date spans, coreference clustering with explicit
   uncertainty (`coref_confidence < 1.0`), and polarity/modality classification.
4. Stage 4 `BiTemporalGraph`: non-destructive `SUPERSEDES` revision preserving BOTH
   current (`latest`) and historical (`earliest`) values, concurrent contradiction marked
   `CONTESTED` (never deleted), and `A2_no_bitemporal_supersession` ablation.
5. Stage 5 `RawCausalMiner`: asymmetric `CausalLift(A -> B)` directionality, Wilson 95% CI,
   multi-cluster verification promotion, `TwinEGraphDoCalculus` counterfactual check,
   `.freeze()` lock, and `A3_no_asymmetric_causal_lift` ablation.
6. Stages 6 & 9 `OpenWorldSHARDIndex` + `CapabilityVersionStore`: 10,240-bit BSC LSH vs
   `A5` linear scan (`candidates_scanned` strictly smaller under LSH), ITL anti-unification
   vs `A4`, cross-cluster promotion gate, MDL bits, and Stage 9 poison-update quarantine
   & non-destructive rollback.
7. Stage 8 independent verifier: hallucinated span rejection, antonym contradiction,
   polarity inconsistency, strict grammatical answer-type checks (including subordinate
   `when` inside `"How many ... when ..."`), and executable calendar date & numeric math.
8. Full offline smoke S11.1 end-to-end run in `tmp_path` (full + all 6 ablations + 5/5
   unseen-combination & rollback probes + freeze locks + 0 post-freeze writes + vault intact).
9. Committed real-dataset S11.1 reports integrity, honesty (`gates_weakened == False`,
   `s11_1_complete == all_4_gates_passed`), and zero raw external dataset text in git.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sqlite3
import pytest

from mirror_crystal.quotient_egraph_resonator import hv10240_of
from research.open_world.contracts import (
    ConversationSession,
    ConversationTurn,
    EvidenceSentence,
    OpenWorldObservation,
    SolverFrozenError,
)
from research.open_world.evaluation.run_s11_0_foundation import (
    REPO_ROOT,
    S11_PACKAGE_ROOT,
    S11_REPORTS_DIR,
    audit_s11_isolation_and_constraints,
)
from research.open_world.evaluation.run_s11_1_experiment import (
    REQUIRED_ABLATIONS,
    audit_zero_manual_domain_schemas,
    run_s11_1_experiment,
)
from research.open_world.evaluator import build_offline_smoke_bundles
from research.open_world.learner import (
    AblationConfig,
    BiTemporalGraph,
    CapabilityVersionStore,
    OpenWorldCapabilityCard,
    OpenWorldSHARDIndex,
    RawCausalMiner,
    RawEvidenceLedger,
    RawEvidenceUnit,
    RawObservationOpenWorldLearner,
    cluster_coreferent_entities,
    execute_temporal_or_numeric_verifier,
    extract_candidate_events,
    extract_candidate_spans,
    extract_evidence_units,
    verify_evidence_grounding,
)
from research.open_world.learner.extraction import detect_polarity_and_modality
from research.open_world.learner.verifier import detect_required_answer_type


def test_s11_1_whole_tree_ast_isolation_zero_llm_gpu_and_py311_compat() -> None:
    """AST-verify every Stage S11.1 learner module obeys strict isolation and Py3.11 stdlib rules."""
    audit = audit_s11_isolation_and_constraints(S11_PACKAGE_ROOT)
    assert audit["passed"] is True, f"S11.1 AST isolation failed: {audit}"
    assert audit["solver_import_violations"] == []
    assert audit["non_eval_oracle_violations"] == []
    assert audit["llm_gpu_violations"] == []
    assert audit["py312_violations"] == []

    learner_files = sorted((S11_PACKAGE_ROOT / "learner").glob("*.py"))
    assert len(learner_files) >= 7, f"Expected >=7 learner modules, found {learner_files}"

    assert RawObservationOpenWorldLearner.evaluator_only is False
    assert RawObservationOpenWorldLearner.uses_gpu is False
    assert RawObservationOpenWorldLearner.uses_pretrained_llm is False
    assert RawObservationOpenWorldLearner.uses_external_api is False
    assert RawObservationOpenWorldLearner.uses_neural_training is False


def test_s11_1_stage2_raw_evidence_ledger_immutability_unicode_and_freeze() -> None:
    """Verify exact UTF-8 byte spans, hash chain, SQLite immutability triggers, empty input, and freeze."""
    empty_obs = OpenWorldObservation(
        task_id="empty_0",
        benchmark_id="test_bench",
        split="train",
        task_kind="extractive_qa",
        prompt_text="   \n\t  ",
        question_text="What happened?",
    )
    assert extract_evidence_units(empty_obs) == ()

    # Multi-byte UTF-8 accented characters and degree symbols
    raw_text = "Les Cévennes lie near 28.5°E. The prétendus réformés lived there in 1685."
    uni_obs = OpenWorldObservation(
        task_id="uni_1",
        benchmark_id="test_bench",
        split="train",
        task_kind="extractive_qa",
        prompt_text=raw_text,
        question_text="Where did the prétendus réformés live?",
    )
    units = extract_evidence_units(uni_obs)
    assert len(units) == 2
    full_bytes = raw_text.encode("utf-8")
    for u in units:
        assert u.raw_utf8_bytes == u.text.encode("utf-8")
        assert (u.byte_end - u.byte_start) == len(u.raw_utf8_bytes)
        assert full_bytes[u.byte_start : u.byte_end] == u.raw_utf8_bytes
        assert u.sha256 == hashlib.sha256(u.raw_utf8_bytes).hexdigest()

    # Append-only SQLite ledger immutability & chain integrity
    ledger = RawEvidenceLedger(db_path=":memory:")
    inserted = ledger.append_batch(units)
    assert inserted == 2
    # Duplicate append is idempotent (ignored, never mutates)
    assert ledger.append_unit(units[0]) is None

    integrity = ledger.verify_integrity()
    assert integrity["ok"] is True
    assert integrity["checked_records"] == 2
    assert integrity["triggers_present"] is True

    # SQLite BEFORE UPDATE and BEFORE DELETE triggers must abort tampering!
    with pytest.raises(sqlite3.IntegrityError):
        ledger._conn.execute(
            "UPDATE raw_evidence_records SET speaker_role = 'tampered' WHERE seq_id = 1"
        )
    with pytest.raises(sqlite3.IntegrityError):
        ledger._conn.execute("DELETE FROM raw_evidence_records WHERE seq_id = 1")

    # Freeze lock rejects post-freeze appends without mutating SQLite rows
    ledger.freeze()
    assert ledger.is_frozen is True
    assert ledger.post_freeze_writes == 0
    with pytest.raises(SolverFrozenError):
        ledger.append_unit(
            RawEvidenceUnit(
                evidence_id="post_freeze_1",
                source_id="src",
                byte_start=0,
                byte_end=5,
                observed_timestamp="",
                observed_ordinal=0,
                speaker_role="document",
                raw_utf8_bytes=b"hello",
                text="hello",
                sha256=hashlib.sha256(b"hello").hexdigest(),
            )
        )
    assert ledger.post_freeze_writes == 1
    assert ledger.verify_integrity()["checked_records"] == 2
    ledger.close()


def test_s11_1_stage3_schema_free_extraction_coref_uncertainty_and_modality() -> None:
    """Verify schema-free span/event extraction, footnote stripping, coref uncertainty, and modality."""
    empty_unit = RawEvidenceUnit(
        evidence_id="u_empty", source_id="s0", byte_start=0, byte_end=0,
        observed_timestamp="", observed_ordinal=0, speaker_role="document",
        raw_utf8_bytes=b"", text="", sha256=hashlib.sha256(b"").hexdigest(),
    )
    assert extract_candidate_spans(empty_unit) == ()
    assert extract_candidate_events(empty_unit) == ()

    txt1 = 'Marie Curie discovered "radium" in 1898[citation needed] with 99.5% purity.'
    txt2 = "Curie might never have published if the lab closed early."
    u1 = RawEvidenceUnit(
        evidence_id="u1", source_id="doc1", byte_start=0, byte_end=len(txt1.encode("utf-8")),
        observed_timestamp="1898-12-26", observed_ordinal=693590, speaker_role="document",
        raw_utf8_bytes=txt1.encode("utf-8"), text=txt1,
        sha256=hashlib.sha256(txt1.encode("utf-8")).hexdigest(),
    )
    u2 = RawEvidenceUnit(
        evidence_id="u2", source_id="doc1", byte_start=0, byte_end=len(txt2.encode("utf-8")),
        observed_timestamp="1899-01-01", observed_ordinal=693596, speaker_role="document",
        raw_utf8_bytes=txt2.encode("utf-8"), text=txt2,
        sha256=hashlib.sha256(txt2.encode("utf-8")).hexdigest(),
    )
    spans1 = extract_candidate_spans(u1, compute_hv=True)
    spans2 = extract_candidate_spans(u2, compute_hv=True)
    surfs1 = {s.surface for s in spans1}
    assert "Marie Curie" in surfs1
    assert "radium" in surfs1
    assert "1898" in surfs1
    assert any("99.5%" in s for s in surfs1)
    assert all("citation needed" not in s.casefold() for s in surfs1)

    # Coreference clusters "Marie Curie" and "Curie" while preserving sub-span uncertainty (< 1.0)
    clusters, confs = cluster_coreferent_entities(list(spans1) + list(spans2))
    mc_id = next(s.entity_id for s in spans1 if s.surface == "Marie Curie")
    c_id = next(s.entity_id for s in spans2 if s.surface == "Curie")
    assert clusters[mc_id] == clusters[c_id]
    assert confs[c_id] < 1.0, "Non-exact sub-span coreference must preserve explicit uncertainty < 1.0"

    # Polarity & modality checks
    assert detect_polarity_and_modality(txt1) == ("AFFIRMED", "FACTUAL")
    pol2, mod2 = detect_polarity_and_modality(txt2)
    assert pol2 in ("NEGATED", "UNCERTAIN")
    assert mod2 == "HYPOTHETICAL"
    assert detect_polarity_and_modality("What books would you recommend for my trip?")[1] in ("PREFERENCE", "QUERY")

    events1 = extract_candidate_events(u1)
    assert len(events1) >= 1
    assert events1[0].polarity == "AFFIRMED"


def test_s11_1_stage4_bitemporal_supersession_contradiction_and_ablation_a2() -> None:
    """Verify non-destructive SUPERSEDES & CONTESTED belief revision and A2 timestamp ablation."""
    obs_update = OpenWorldObservation(
        task_id="bt_update",
        benchmark_id="test_mem",
        split="hidden_eval",
        task_kind="long_memory_qa",
        prompt_text="",
        question_text="Where is my passport?",
        question_date="2023/06/01",
        sessions=(
            ConversationSession(
                session_id="sess_old",
                session_date="2023/05/01 (Mon) 10:00",
                turns=(ConversationTurn("t0", "sess_old", 0, "user", "I initially stored my passport in the garage."),),
            ),
            ConversationSession(
                session_id="sess_new",
                session_date="2023/05/20 (Sat) 10:00",
                turns=(ConversationTurn("t1", "sess_new", 0, "user", "Actually I moved my passport to the attic."),),
            ),
        ),
    )
    units_up = extract_evidence_units(obs_update, flatten_timestamps=False)
    bt = BiTemporalGraph(enable_bitemporal_supersession=True)
    bt.ingest_units(units_up)

    assert bt.stats()["num_supersedes_edges"] >= 1
    assert len(bt.relations) >= 2, "Old belief must be preserved non-destructively!"
    all_rids = list(bt.relations.keys())
    latest = bt.resolve_query_state(all_rids, temporal_mode="latest")
    earliest = bt.resolve_query_state(all_rids, temporal_mode="earliest")
    assert latest["winner"] is not None and "attic" in latest["winner"].arg_right.casefold()
    assert earliest["winner"] is not None and "garage" in earliest["winner"].arg_right.casefold()

    # Concurrent contradiction on SAME date -> marked CONTESTED, neither deleted
    obs_conflict = OpenWorldObservation(
        task_id="bt_conflict",
        benchmark_id="test_mem",
        split="hidden_eval",
        task_kind="long_memory_qa",
        prompt_text="",
        question_text="Where is the summit?",
        question_date="2023/06/05",
        sessions=(
            ConversationSession(
                session_id="src_a",
                session_date="2023/06/01 (Thu) 09:00",
                turns=(ConversationTurn("c0", "src_a", 0, "user", "The summit is scheduled in Bern."),),
            ),
            ConversationSession(
                session_id="src_b",
                session_date="2023/06/01 (Thu) 11:00",
                turns=(ConversationTurn("c1", "src_b", 0, "user", "The summit is scheduled in Zurich."),),
            ),
        ),
    )
    bt_c = BiTemporalGraph(enable_bitemporal_supersession=True)
    bt_c.ingest_units(extract_evidence_units(obs_conflict, flatten_timestamps=False))
    assert bt_c.stats()["num_contested_edges"] >= 1
    assert len(bt_c.relations) >= 2

    # Ablation A2: flattening timestamps disables bi-temporal supersession
    units_flat = extract_evidence_units(obs_update, flatten_timestamps=True)
    bt_a2 = BiTemporalGraph(enable_bitemporal_supersession=False)
    bt_a2.ingest_units(units_flat)
    assert bt_a2.stats()["num_supersedes_edges"] == 0


def test_s11_1_stage5_causal_miner_asymmetric_lift_wilson_ci_and_ablation_a3() -> None:
    """Verify directional CausalLift(A->B) != CausalLift(B->A), Wilson CI, counterfactuals, freeze & A3."""
    miner = RawCausalMiner(enable_asymmetric_causal_lift=True)
    miner_a3 = RawCausalMiner(enable_asymmetric_causal_lift=False)

    docs = [
        "Heavy cyclone caused severe flooding across the valley.",
        "Another cyclone caused coastal flooding near the port.",
        "Because the cyclone intensified, rapid flooding damaged bridges.",
        "The fierce cyclone led to widespread flooding in lowlands.",
        "Rare meteor caused glowing sparks above the hill.",
        "Another meteor led to faint sparks over the lake.",
        "Warm sunshine brought calm meadows and bright gardens.",
    ]
    for idx, d in enumerate(docs):
        miner.observe_raw_document(d, source_id=f"cluster_{idx}")
        miner_a3.observe_raw_document(d, source_id=f"cluster_{idx}")

    lift_fwd, _ = miner.compute_asymmetric_causal_lift("cyclon", "flood")
    lift_rev, _ = miner.compute_asymmetric_causal_lift("flood", "cyclon")
    assert lift_fwd > lift_rev, f"Expected directional lift(cyclone->flood) > lift(flood->cyclone), got {lift_fwd} vs {lift_rev}"

    hyps = miner.materialize_hypotheses(k_min_clusters=3, max_hypotheses=50)
    assert len(hyps) >= 2
    verified = [h for h in hyps if h.epistemic_status == "VERIFIED_MULTI_CLUSTER"]
    unverified = [h for h in hyps if h.epistemic_status == "HYPOTHESIS_UNVERIFIED"]
    assert len(verified) >= 1, f"Expected verified hypothesis from 4 clusters, got {[h.to_compact_dict() for h in hyps[:5]]}"
    assert len(unverified) >= 1, "Low-support 2-cluster hypothesis must remain HYPOTHESIS_UNVERIFIED"
    for h in verified:
        assert 0.0 <= h.ci_low <= h.ci_high <= 1.0
        assert h.asymmetric_margin > 0.0
        assert len(h.support_evidence_ids) >= 3

    # Verify TwinEGraph counterfactual scoring runs cleanly inside score_causal_choices
    choice_res = miner.score_causal_choices(
        premise="A heavy cyclone hit the coast.",
        choices=("Severe flooding damaged the port.", "Calm meadows bloomed in sunshine."),
        relation_direction="effect",
    )
    assert choice_res["best_idx"] == 0

    # Full miner distinguishes cause vs effect direction asymmetrically, while A3 is strictly symmetric!
    choice_rev = miner.score_causal_choices(
        premise="A heavy cyclone hit the coast.",
        choices=("Severe flooding damaged the port.", "Calm meadows bloomed in sunshine."),
        relation_direction="cause",
    )
    assert choice_res["scores"][0] > choice_rev["scores"][0]
    assert choice_res["matched_hypotheses"] >= 1

    miner_a3.materialize_hypotheses(k_min_clusters=3, max_hypotheses=50)
    a3_eff = miner_a3.score_causal_choices(
        premise="A heavy cyclone hit the coast.",
        choices=("Severe flooding damaged the port.", "Calm meadows bloomed in sunshine."),
        relation_direction="effect",
    )
    a3_cau = miner_a3.score_causal_choices(
        premise="A heavy cyclone hit the coast.",
        choices=("Severe flooding damaged the port.", "Calm meadows bloomed in sunshine."),
        relation_direction="cause",
    )
    assert a3_eff["scores"] == a3_cau["scores"], "Ablation A3 must remove directional asymmetry!"
    assert a3_eff["matched_hypotheses"] == 0

    # Freeze blocks further writes
    miner.freeze()
    assert miner.is_frozen is True
    assert miner.post_freeze_writes == 0
    with pytest.raises(SolverFrozenError):
        miner.observe_raw_document("Storm caused hail.", source_id="late")
    assert miner.post_freeze_writes == 1


def test_s11_1_stages_6_and_9_shard_lsh_vs_a5_itl_vs_a4_and_poison_rollback() -> None:
    """Verify SHARD 10,240-bit LSH speedup vs A5, ITL anti-unification vs A4, and Stage 9 rollback."""
    idx_lsh = OpenWorldSHARDIndex(num_tables=12, bits_per_table=4, use_lsh=True)
    idx_lin = OpenWorldSHARDIndex(num_tables=12, bits_per_table=4, use_lsh=False)

    cards: list[OpenWorldCapabilityCard] = []
    for i in range(80):
        hv = hv10240_of(f"test_capability_frame_{i}")
        card = OpenWorldCapabilityCard(
            card_id=f"card_{i}",
            capability_kind="QA_ALIGNMENT",
            query_signature=(("task_kind", "extractive_qa"), ("wh_class", f"cls_{i % 7}")),
            expected_span_shape="PROPER_SPAN",
            anchor_pattern="after_anchor",
            left_pivots=("in",),
            right_pivots=(),
            preferred_len_min=1,
            preferred_len_max=4,
            support_clusters=(f"c{i}", "c_a", "c_b"),
            hit_count=5,
            total_count=6,
            mdl_gain_bits=32.0,
            status="verified",
            version=1,
            card_hv=hv,
        )
        cards.append(card)
        idx_lsh.add_card(card)
        idx_lin.add_card(card)

    target_hv = cards[42].card_hv
    res_lsh = idx_lsh.query(target_hv, top_k=3)
    res_lin = idx_lin.query(target_hv, top_k=3)
    assert res_lsh[0][0].card_id == "card_42"
    assert res_lin[0][0].card_id == "card_42"
    assert idx_lin.candidates_scanned == 80
    assert idx_lsh.candidates_scanned < idx_lin.candidates_scanned

    # Stage 6 ITL anti-unification vs A4 raw instance + Stage 9 poison quarantine & rollback
    smoke_qa = build_offline_smoke_bundles()["smoke_qa"]
    store_itl = CapabilityVersionStore(enable_anti_unification=True, use_shard_lsh=True, k_min_clusters=2)
    store_a4 = CapabilityVersionStore(enable_anti_unification=False, use_shard_lsh=True, k_min_clusters=2)
    info_itl = store_itl.induce_from_training(smoke_qa.train_examples)
    info_a4 = store_a4.induce_from_training(smoke_qa.train_examples)
    assert info_itl["total_mdl_bits"] > 0
    assert info_itl["verified_card_count"] >= 1
    assert info_a4["total_mdl_bits"] > 0

    # Inject a poisoned capability that degrades regression score -> must quarantine & roll back!
    good_card = cards[0]
    ok_init = store_itl.commit_with_regression_guard(
        [good_card],
        regression_evaluator=lambda active: 0.90 if good_card.card_id in active else 0.20,
        min_baseline_score=0.80,
    )
    assert ok_init is True
    poison_card = OpenWorldCapabilityCard(
        card_id="poison_bad_card",
        capability_kind="QA_ALIGNMENT",
        query_signature=(("task_kind", "extractive_qa"), ("wh_class", "agent_entity")),
        expected_span_shape="QUANTITY_SPAN",
        anchor_pattern="after_anchor",
        left_pivots=(),
        right_pivots=(),
        preferred_len_min=1,
        preferred_len_max=2,
        support_clusters=("poison_c",),
        hit_count=1,
        total_count=5,
        mdl_gain_bits=-5.0,
        status="verified",
        version=1,
        card_hv=hv10240_of("poison_card"),
    )
    ok_poison = store_itl.commit_with_regression_guard(
        [poison_card],
        regression_evaluator=lambda active: 0.15 if "poison_bad_card" in active else 0.90,
        min_baseline_score=0.80,
    )
    assert ok_poison is False
    assert "poison_bad_card" not in store_itl.verified_cards
    assert "poison_bad_card" in store_itl.quarantined_cards
    assert store_itl.rollback_count >= 1

    store_itl.freeze()
    assert store_itl.post_freeze_writes == 0
    with pytest.raises(SolverFrozenError):
        store_itl.commit_with_regression_guard([cards[1]])
    assert store_itl.post_freeze_writes == 1


def test_s11_1_stage8_independent_verifier_antonyms_types_and_executable_math() -> None:
    """Verify Stage 8 span grounding, antonym/polarity rejection, grammatical types, and date/numeric math."""
    # Grammatical answer-type disambiguation: subordinate 'when' must NOT override 'How many'
    assert detect_required_answer_type("How many engineers did I lead when I started my new role?") == "numeric"
    assert detect_required_answer_type("When did the conference start?") == "temporal"
    assert detect_required_answer_type("What percentage of budgets went to solar?") == "percentage"

    s_text = "Mount Kestrel is the largest mountain in the range, discovered in 1892."
    u = RawEvidenceUnit(
        evidence_id="v0", source_id="doc", byte_start=0, byte_end=len(s_text.encode("utf-8")),
        observed_timestamp="", observed_ordinal=0, speaker_role="document",
        raw_utf8_bytes=s_text.encode("utf-8"), text=s_text,
        sha256=hashlib.sha256(s_text.encode("utf-8")).hexdigest(),
    )

    # 1. Hallucinated span not grounded in raw text
    v_hall = verify_evidence_grounding(
        question="What mountain is in the range?",
        proposed_answer="Mount Everest",
        cited_units=(u,),
        all_units=(u,),
        idf_lookup={},
    )
    assert v_hall.span_grounded is False

    # 2. Scalar antonym contradiction ("smallest" vs "largest")
    v_ant = verify_evidence_grounding(
        question="What is the smallest mountain in the range?",
        proposed_answer="Mount Kestrel",
        cited_units=(u,),
        all_units=(u,),
        idf_lookup={},
    )
    assert v_ant.antonym_contradiction is True

    # 3. Polarity mismatch ("did not discover" vs affirmed "discovered")
    v_pol = verify_evidence_grounding(
        question="Which mountain was not discovered in 1892?",
        proposed_answer="Mount Kestrel",
        cited_units=(u,),
        all_units=(u,),
        idf_lookup={},
    )
    assert v_pol.polarity_consistent is False

    # 4. Required percentage type rejected when passage has only a year
    v_pct = verify_evidence_grounding(
        question="What percentage of Mount Kestrel is granite?",
        proposed_answer="1892",
        cited_units=(u,),
        all_units=(u,),
        idf_lookup={},
    )
    assert v_pct.answer_type_compatible is False

    # 5. Executable temporal date-difference with relative "yesterday" offset
    t_obs = OpenWorldObservation(
        task_id="date_math",
        benchmark_id="test_mem",
        split="hidden_eval",
        task_kind="long_memory_qa",
        prompt_text="",
        question_text="How many days passed between visiting Lisbon and visiting Porto?",
        question_date="2023/06/01",
        sessions=(
            ConversationSession(
                session_id="s_lisbon",
                session_date="2023/05/02 (Tue) 09:00",
                turns=(ConversationTurn("t0", "s_lisbon", 0, "user", "Yesterday I visited Lisbon."),),
            ),
            ConversationSession(
                session_id="s_porto",
                session_date="2023/05/21 (Sun) 09:00",
                turns=(ConversationTurn("t1", "s_porto", 0, "user", "Today I visited Porto."),),
            ),
        ),
    )
    t_units = extract_evidence_units(t_obs, flatten_timestamps=False)
    exec_date = execute_temporal_or_numeric_verifier(
        question=t_obs.question_text,
        question_date=t_obs.question_date,
        ranked_units=[(0.9, t_units[0]), (0.9, t_units[1])],
        idf_lookup={"lisbon": 4.0, "porto": 4.0},
        enable_bitemporal=True,
    )
    assert exec_date is not None
    assert exec_date["answer"] == "20 days"


def test_s11_1_offline_smoke_end_to_end_all_6_ablations_and_unseen_probes(tmp_path: Path) -> None:
    """Run full S11.1 experiment offline in tmp_path: 6 ablations, 5 unseen probes, freeze, vault intact."""
    summary = run_s11_1_experiment(
        reports_dir=tmp_path / "reports",
        repo_root=REPO_ROOT,
        offline_smoke=True,
        allow_download=False,
    )
    assert summary["stage"] == "S11.1"
    assert summary["mode"] == "offline_smoke"
    assert summary["isolation_audit"]["passed"] is True
    assert summary["vault_integrity"]["untouched"] is True
    assert summary["gate_evaluation"]["gates"]["S11_1_G4"]["passed"] is True
    assert summary["unseen_combination_probes"]["passed"] is True

    expected_ab_ids = {ab_id for ab_id, _ in REQUIRED_ABLATIONS}
    assert len(expected_ab_ids) == 6
    for bid in ("smoke_causal", "smoke_qa", "smoke_memory"):
        bdata = summary["benchmarks"][bid]
        assert bdata["full_s11_1"]["freeze_lock_verified"] is True
        assert bdata["full_s11_1"]["zero_eval_memory_writes"] is True
        assert set(bdata["ablations"].keys()) == expected_ab_ids
        for ab_res in bdata["ablations"].values():
            assert ab_res["freeze_lock_verified"] is True
            assert ab_res["zero_eval_memory_writes"] is True

    assert (tmp_path / "reports" / "s11_1_summary.json").is_file()
    assert (tmp_path / "reports" / "s11_1_ablation_matrix.csv").is_file()
    assert (tmp_path / "reports" / "s11_1_research_report.md").is_file()


def test_s11_1_committed_real_dataset_artifacts_integrity_and_honest_verdict() -> None:
    """Verify committed S11.1 real-dataset summary, ablations, honesty invariants, and zero raw datasets."""
    summary_path = S11_REPORTS_DIR / "s11_1_summary.json"
    assert summary_path.is_file(), "Committed s11_1_summary.json must exist"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    assert summary["stage"] == "S11.1"
    assert summary["mode"] == "real_external_benchmarks"
    assert summary["isolation_audit"]["passed"] is True
    assert summary["vault_integrity"]["untouched"] is True
    assert summary["unseen_combination_probes"]["passed"] is True

    # Honesty invariant: S11.1 may NEVER be claimed complete unless ALL 4 preregistered gates pass!
    gates = summary["gate_evaluation"]
    assert set(gates["gates"].keys()) == {"S11_1_G1", "S11_1_G2", "S11_1_G3", "S11_1_G4"}
    assert gates["gates"]["S11_1_G4"]["passed"] is True
    assert summary["honest_verdict"]["gates_weakened"] is False
    assert summary["honest_verdict"]["s11_1_complete"] is gates["all_4_gates_passed"]

    expected_ab_ids = {ab_id for ab_id, _ in REQUIRED_ABLATIONS}
    assert set(summary["benchmarks"].keys()) == {"balanced_copa", "squad_v2", "longmemeval"}
    for bid, bdata in summary["benchmarks"].items():
        assert bdata["full_s11_1"]["freeze_lock_verified"] is True
        assert bdata["full_s11_1"]["zero_eval_memory_writes"] is True
        assert set(bdata["ablations"].keys()) == expected_ab_ids
        assert (S11_REPORTS_DIR / "raw_runs" / f"s11_1_{bid}.json").is_file()

    assert (S11_REPORTS_DIR / "s11_1_ablation_matrix.csv").is_file()
    assert (S11_REPORTS_DIR / "s11_1_research_report.md").is_file()

    forbidden_raw_names = {
        "train.csv", "test.csv", "dev-v2.0.json", "longmemeval_oracle.json", "longmemeval_s_cleaned.json"
    }
    found_raw = [p.name for p in S11_PACKAGE_ROOT.rglob("*") if p.name in forbidden_raw_names]
    assert found_raw == [], f"Raw external benchmark files must never be committed in git: {found_raw}"


def test_s11_1_review_regressions_contested_abstention_jtms_large_bucket_multichoice_utf8_and_sqlite_commit(
    tmp_path: Path,
) -> None:
    """Adversarial regression suite covering all Step-2 review findings (zero test weakening)."""
    # 1. Programmatic zero-manual-schema audit passes cleanly
    schema_audit = audit_zero_manual_domain_schemas(S11_PACKAGE_ROOT / "learner")
    assert schema_audit["passed"] is True, f"Manual schema/hack violation: {schema_audit}"

    # 2. File-backed SQLite single append_unit() persists immediately across connections
    db_path = tmp_path / "evidence_ledger.sqlite"
    ledger_disk = RawEvidenceLedger(db_path=str(db_path))
    u_disk = RawEvidenceUnit(
        evidence_id="disk_u0", source_id="src_disk", byte_start=0, byte_end=11,
        observed_timestamp="2024/05/01", observed_ordinal=739007, speaker_role="user",
        raw_utf8_bytes=b"hello world", text="hello world",
        sha256=hashlib.sha256(b"hello world").hexdigest(),
    )
    assert ledger_disk.append_unit(u_disk) is not None
    ledger_disk.close()
    conn_check = sqlite3.connect(str(db_path))
    row_cnt = conn_check.execute("SELECT COUNT(*) FROM raw_evidence_records").fetchone()[0]
    conn_check.close()
    assert row_cnt == 1

    # 3. End-to-end CONTESTED contradictory observation MUST abstain in RawObservationOpenWorldLearner.predict()
    learner = RawObservationOpenWorldLearner(seed=101)
    learner.freeze()
    obs_conflict = OpenWorldObservation(
        task_id="reg_conflict",
        benchmark_id="longmemeval",
        split="hidden_eval",
        task_kind="long_memory_qa",
        prompt_text="Question date: 2024/04/05",
        question_text="What is my studio locker code?",
        question_date="2024/04/05",
        sessions=(
            ConversationSession("cs1", "2024/04/01", (ConversationTurn("cs1::t0", "cs1", 0, "user", "My studio locker code is 4821."),)),
            ConversationSession("cs2", "2024/04/01", (ConversationTurn("cs2::t0", "cs2", 0, "user", "My studio locker code is 9934."),)),
        ),
    )
    p_conf = learner.predict(obs_conflict)
    assert p_conf.abstained is True, f"CONTESTED observation must abstain, got {p_conf}"
    assert p_conf.metadata.get("verification", {}).get("bitemporal_contested") is True

    # 4. Large >25-relation bucket, TropicalJTMS defeat, Stage 3 BSC coreference, and same-day t_event progression
    sessions_many = tuple(
        ConversationSession(
            f"s_{i}", f"2024/01/{i + 1:02d}",
            (ConversationTurn(f"t_{i}", f"s_{i}", 0, "user", f"My observatory telescope is stored in room {i}."),),
        )
        for i in range(28)
    ) + (
        ConversationSession(
            "s_28", "2024/02/01",
            (ConversationTurn("t_28", "s_28", 0, "user", "Actually I moved my observatory telescope to the rooftop dome."),),
        ),
    )
    obs_many = OpenWorldObservation(
        task_id="reg_many", benchmark_id="longmemeval", split="hidden_eval", task_kind="long_memory_qa",
        prompt_text="", question_text="Where is my observatory telescope?", sessions=sessions_many, question_date="2024/02/10",
    )
    bt_many = BiTemporalGraph(enable_bitemporal_supersession=True)
    bt_many.ingest_units(extract_evidence_units(obs_many))
    st_many = bt_many.stats()
    assert st_many["num_supersedes_edges"] >= 1
    assert st_many["jtms_defeated_relations"] >= 1
    assert st_many["jtms_active_relations"] == 1
    assert st_many["num_entity_clusters"] >= 1

    # Same calendar date with relative event-date progression ("yesterday" -> "today")
    obs_sameday = OpenWorldObservation(
        task_id="reg_sameday", benchmark_id="longmemeval", split="hidden_eval", task_kind="long_memory_qa",
        prompt_text="", question_text="Where is my bicycle?", question_date="2024/05/12",
        sessions=(
            ConversationSession("sd1", "2024/05/10", (ConversationTurn("sd1::t0", "sd1", 0, "user", "Yesterday my bicycle was in the shed."),)),
            ConversationSession("sd2", "2024/05/10", (ConversationTurn("sd2::t0", "sd2", 0, "user", "Today I moved my bicycle to the porch."),)),
        ),
    )
    bt_sd = BiTemporalGraph(enable_bitemporal_supersession=True)
    bt_sd.ingest_units(extract_evidence_units(obs_sameday))
    assert bt_sd.stats()["num_supersedes_edges"] >= 1

    # 5. 3-choice causal reasoning selects choice index 2 (not hardcoded 0 vs 1!)
    obs_3c = OpenWorldObservation(
        task_id="reg_3c", benchmark_id="balanced_copa", split="hidden_eval", task_kind="causal_choice",
        prompt_text="A heavy cyclone hit the coastal town.",
        question_text="What happened as a result?",
        candidate_choices=("Nothing changed.", "A cat slept quietly.", "Severe cyclone flooding damaged coastal homes."),
        relation_direction="effect",
    )
    p_3c = learner.predict(obs_3c)
    assert p_3c.predicted_choice_index == 2

    # 6. Non-Latin UTF-8 (Japanese) tokenization, contraction entity bug fix, coordinated operand & Markdown bullet rejection
    from research.open_world.learner.extraction import unicode_tokenize
    from research.open_world.learner.verifier import _extract_question_proper_entities, answer_satisfies_type

    jp_toks = unicode_tokenize("東京は日本の首都です。")
    assert "東京" in jp_toks and "首都" in jp_toks

    # Contraction apostrophes ("I've ... I'm") must NEVER be parsed as quoted proper entities!
    ents_contr = _extract_question_proper_entities("I've been thinking about making tea, but I'm not sure which blend to pick?")
    assert all("been thinking" not in e for e in ents_contr)

    # Markdown numbered list bullet must be rejected as a numeric answer
    assert answer_satisfies_type("8. **Lemon Bars**: Tart and sweet", "numeric") is False
    assert answer_satisfies_type("45 minutes", "numeric") is True
    assert answer_satisfies_type("twice", "numeric") is True

    # Coordinated operand missing check ("tomatoes and chili peppers" vs "5 tomato plants")
    u_tom = RawEvidenceUnit(
        evidence_id="u_tom", source_id="s_tom", byte_start=0, byte_end=34,
        observed_timestamp="2024/04/01", observed_ordinal=738977, speaker_role="user",
        raw_utf8_bytes=b"I planted 5 tomato plants initially.", text="I planted 5 tomato plants initially.",
        sha256=hashlib.sha256(b"I planted 5 tomato plants initially.").hexdigest(),
    )
    ver_coord = verify_evidence_grounding(
        question="How many plants did I initially plant for tomatoes and chili peppers?",
        proposed_answer="5",
        cited_units=(u_tom,),
        all_units=(u_tom,),
        idf_lookup={},
    )
    assert ver_coord.coord_operand_ungrounded is True
    assert "ungrounded_coordinated_operand" in ver_coord.rejection_reasons

    # 7. Honest memory_footprint() includes retained persistent tables (idf_by_bench, causal npmi_neighbors)
    smoke = build_offline_smoke_bundles()
    l_mem = RawObservationOpenWorldLearner(seed=101)
    l_mem.fit("smoke_qa", smoke["smoke_qa"].train_examples)
    l_mem.freeze()
    fp = l_mem.memory_footprint()
    assert fp.serialized_bytes > 500
    assert fp.active_entry_count >= len(l_mem._idf_by_bench["smoke_qa"])
