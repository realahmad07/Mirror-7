"""Comprehensive verification suite for Mirror-7's 5 Unified CPU-Native Organs (Stages 1-13).

Tests every mathematical invariant and end-to-end pipeline across:
- Organ 1: mirror_crystal/cortical_active_inference.py (Stages 1, 5B, 10, 12-L2, 13-L3)
- Organ 2: mirror_crystal/quotient_egraph_resonator.py (Stages 2, 5A, 7, 11, 13-L4)
- Organ 3: mirror_crystal/hippocampal_sdm_memory.py (Stages 6, 8, 9, 11, 13-L2)
- Organ 4: mirror_crystal/multifile_cegis_architect.py (Stages 3, 5C, 12-L3/L4/L5)
- Organ 5: mirror_crystal/ternary_lut_realizer.py + training/cpu_autonomous_engine.py
           + scripts/transcompile_open_weights_to_lut.py (Stages 4, 12-L1, 13-L1)
- Router & Backend integration: nl_uol_router.py + mirror7_backend/service.py
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import random

import pytest

from mirror_crystal.cortical_active_inference import (
    BilingualCCGHDCParser,
    CharacterReferenceFrame,
    CorticalActiveInferenceController,
    CorticalCMPVoter,
    ModuleReferenceFrame,
    OCC_22_EMOTIONS,
    TropicalRSAPragmaticEngine,
)
from mirror_crystal.hippocampal_sdm_memory import (
    BitParallelPPRWorkingMemory,
    CompactRoaringBitmap,
    FCAGaloisConceptSplitter,
    FlyHash,
    KanervaSDMInt8,
    MengerProvenanceVerifier,
    SequiturSymbolMinter,
    SubmodularMDLRuleDistiller,
)
from mirror_crystal.multifile_cegis_architect import (
    FirstOrderASTAntiUnifier,
    MultiFileCEGISArchitect,
    StitchMDLLibraryLearner,
)
from mirror_crystal.open_ended_concept_lattice import OpenEndedConceptLatticeEngine
from mirror_crystal.quotient_egraph_resonator import (
    BitSlicedBundle10240,
    BitSlicedStochasticResonator,
    DEFAULT_ALGEBRA_REWRITE_RULES,
    D_UNIFIED,
    DefeasibleRule,
    LeeArcSSP,
    NUM_BLOCKS,
    QuotientEGraph,
    TropicalJTMS,
    TwinEGraphDoCalculus,
    WeisfeilerLemanAnalogicalMapper,
    bind10240,
    hamming_distance10240,
    hv10240_of,
    sim_q16_10240,
)
from mirror_crystal.ternary_lut_realizer import (
    I_LLM_IntegerOps,
    LTAGKneserNeyRealizer,
    PDCSTModifiedKneserNey,
    Q16,
    SAESteeringBank,
    TernaryLUTLinearKernel,
)


# ===========================================================================
# ORGAN 2: Quotient E-Graph Resonator, 0-Drift SSP, 1-WL, T-JTMS & Twin-do
# ===========================================================================


def test_organ2_10240bit_algebra_and_zero_drift_lee_arc_ssp():
    assert D_UNIFIED == 10_240
    assert NUM_BLOCKS == 160

    a, b, c = hv10240_of("alpha"), hv10240_of("beta"), hv10240_of("gamma")
    bd = BitSlicedBundle10240()
    bd.add(a, 3)
    bd.add(b, 1)
    bd.add(c, 1)
    superpos = bd.binary()
    assert sim_q16_10240(superpos, a) > 40_000
    assert sim_q16_10240(superpos, b) > 10_000

    # 0-Drift Fixed-Point Lee-Arc SSP: 1,000 sequential steps == 1 direct jump with d_H == 0!
    ssp = LeeArcSSP(axes=2)
    p0 = ssp.encode_point((3, 7))
    p_stepped = p0
    for _ in range(1000):
        p_stepped = ssp.translate(p_stepped, (2, -3))
    p_direct = ssp.encode_point((3 + 2000, 7 - 3000))
    assert p_stepped.hamming_distance(p_direct) == 0
    assert ssp.overlap_q16(p_stepped, p_direct) == Q16

    # Reversing 1,000 steps returns bit-identically to origin
    p_back = p_stepped
    for _ in range(1000):
        p_back = ssp.translate(p_back, (-2, 3))
    assert p_back.hamming_distance(p0) == 0

    # Ballistic trajectory collision check
    traj = ssp.simulate_ballistic_trajectory((0, 0), (3, 10), (0, -2), steps=5)
    collides_same, ov_same = ssp.check_collision(traj[-1][1], ssp.encode_point(traj[-1][0]))
    collides_far, _ = ssp.check_collision(traj[-1][1], ssp.encode_point((99, 99)))
    assert collides_same is True and ov_same == Q16
    assert collides_far is False


def test_organ2_1wl_analogical_transfer_solar_to_atom():
    mapper = WeisfeilerLemanAnalogicalMapper(hops=3)
    solar = [
        ("sun", "attracts", "planet"),
        ("planet", "orbits", "sun"),
        ("sun", "more_massive_than", "planet"),
        ("planet", "cooler_than", "sun"),
    ]
    atom = [
        ("nucleus", "attracts", "electron"),
        ("electron", "orbits", "nucleus"),
    ]
    res = mapper.align_and_transfer("solar_system", solar, "rutherford_atom", atom)
    assert res.entity_mapping["sun"] == "nucleus"
    assert res.entity_mapping["planet"] == "electron"
    assert ("nucleus", "more_massive_than", "electron") in res.transferred_triples
    assert ("electron", "cooler_than", "nucleus") in res.transferred_triples


def test_organ2_deferred_egraph_qehdc_zero_hamming_invariance():
    eg = QuotientEGraph()
    e1 = eg.add_expr(("add", "x", "y"))
    e2 = eg.add_expr(("add", "y", "x"))
    e3 = eg.add_expr(("mul", "z", "2"))
    e4 = eg.add_expr(("add", "z", "z"))
    e5 = eg.add_expr(("not", ("not", "p")))
    e6 = eg.add_expr("p")

    stats = eg.saturate(DEFAULT_ALGEBRA_REWRITE_RULES, max_iters=6)
    assert stats["applications"] >= 3
    assert eg.equivalent(e1, e2)
    assert eg.equivalent(e3, e4)
    assert eg.equivalent(e5, e6)

    # QE-HDC Invariant: equivalent expressions map to exact same 10,240-bit HV (d_H == 0)
    assert hamming_distance10240(eg.project_eclass_hv(e1), eg.project_eclass_hv(e2)) == 0
    assert hamming_distance10240(eg.project_eclass_hv(e3), eg.project_eclass_hv(e4)) == 0
    assert hamming_distance10240(eg.project_eclass_hv(e5), eg.project_eclass_hv(e6)) == 0
    assert hamming_distance10240(eg.project_eclass_hv(e1), eg.project_eclass_hv(e6)) > 4000


def test_organ2_stochastic_resonator_and_uint8_multiset_deflation():
    cb = {
        "color": {k: hv10240_of(f"c:{k}") for k in ("red", "blue", "green", "gold")},
        "shape": {k: hv10240_of(f"s:{k}") for k in ("circle", "square", "star", "hex")},
        "size": {k: hv10240_of(f"z:{k}") for k in ("tiny", "mid", "large", "giant")},
    }
    res = BitSlicedStochasticResonator(cb, seed=42)

    # Single 3-factor bound composite factorization
    target_tuple = ("gold", "star", "giant")
    comp = bind10240(bind10240(cb["color"]["gold"], cb["shape"]["star"]), cb["size"]["giant"])
    fact = res.factorize(comp)
    assert fact.converged is True
    assert fact.factors == target_tuple
    assert fact.similarity_q16 == Q16

    # Superposition peeling via exact uint8 multiset histogram deflation A_{t+1} = max(0, A_t - E*)
    triples = [
        ("red", "circle", "tiny"),
        ("blue", "square", "large"),
        ("gold", "star", "giant"),
    ]
    comps = [
        bind10240(bind10240(cb["color"][c], cb["shape"][s]), cb["size"][z])
        for c, s, z in triples
    ]
    peeled = res.peel_superposition(comps)
    recovered_set = {p.factors for p in peeled}
    assert recovered_set == set(triples)


def test_organ2_tropical_jtms_specificity_and_twin_egraph_do_calculus():
    # Poole specificity: penguin -> bird, bird -> flies, penguin -> not flies
    jtms = TropicalJTMS()
    jtms.add_isa("penguin", "bird")
    jtms.add_isa("sparrow", "bird")
    jtms.add_rule(DefeasibleRule("r_bird_flies", ("bird",), "flies", negated=False))
    jtms.add_rule(DefeasibleRule("r_penguin_no_fly", ("penguin",), "flies", negated=True))

    j_peng = jtms.clone()
    j_peng.assert_fact("penguin", True)
    res_peng = j_peng.query("flies")
    assert res_peng is not None and res_peng.truth_value is False
    assert "r_bird_flies" in res_peng.defeated_rules

    j_spar = jtms.clone()
    j_spar.assert_fact("sparrow", True)
    res_spar = j_spar.query("flies")
    assert res_spar is not None and res_spar.truth_value is True

    # Copy-on-Write Twin-E-Graph do-calculus + Dual-Mode World Gate (M_truth vs M_canon)
    twin = TwinEGraphDoCalculus()
    twin.assert_world_fact("rain", True, world="truth")
    twin.add_causal_rule("r_rain_wet", ("rain",), "wet_ground", world="truth")
    twin.add_causal_rule("r_wet_slip", ("wet_ground",), "slippery", world="truth")

    report = twin.do_intervene({"wet_ground": False}, world="truth")
    assert "r_rain_wet" in report.severed_rules
    assert report.baseline_beliefs["slippery"] is True
    assert report.counterfactual_beliefs["rain"] is True
    assert report.counterfactual_beliefs["wet_ground"] is False
    assert report.counterfactual_beliefs.get("slippery") is not True
    # Base world must remain untouched (Copy-on-Write)
    assert twin.truth_jtms.query("wet_ground").truth_value is True

    # M_truth blocks ungrounded metaphor; M_canon allows metaphor & blocks canon contradiction
    ok_truth_fig, _ = twin.verify_claim_gate("grief_is_rusted_anchor", True, mode="truth", is_figurative=True)
    ok_canon_fig, _ = twin.verify_claim_gate("grief_is_rusted_anchor", True, mode="canon", is_figurative=True)
    assert ok_truth_fig is False
    assert ok_canon_fig is True

    twin.assert_world_fact("captain_alive", False, world="canon")
    ok_dead_speaks, msg = twin.verify_claim_gate("captain_alive", True, mode="canon", is_figurative=False)
    assert ok_dead_speaks is False and "M_canon=0" in msg


# ===========================================================================
# ORGAN 3: Hippocampal SDM, FlyHash, Roaring PPR, Sequitur, Menger & FCA
# ===========================================================================


def test_organ3_roaring_flyhash_and_kanerva_sdm_50k_tokens():
    bm1 = CompactRoaringBitmap([1, 5, 70_000, 131_075])
    bm2 = CompactRoaringBitmap([5, 131_075, 200_000])
    assert len(bm1 | bm2) == 5
    assert (bm1 & bm2).to_list() == [5, 131_075]
    assert (bm1 - bm2).to_list() == [1, 70_000]
    assert not bm1.isdisjoint(bm2)

    fh = FlyHash(input_dim=64, hash_dim=256, top_k=16)
    sdr = fh.hash_text("amina hides forged sea chart inside brass astrolabe")
    assert sdr.bit_count() == 16

    # 1-shot KanervaSDMInt8 across 50,000+ simulated tokens in fixed counter RAM
    sdm = KanervaSDMInt8(num_hard_locations=256, sdr_dim=256, top_k_sdr=16, value_dim=64)
    ram_before = sdm.resident_counter_bytes()
    sdm.write(
        "turn_1_astrolabe",
        "amina hid forged sea chart inside brass astrolabe cabin",
        {"actor": "amina", "object": "forged_sea_chart", "container": "brass_astrolabe"},
        salience_q8=512,
        token_count=400,
    )
    for turn in range(2, 62):
        sdm.write(
            f"turn_{turn}_filler",
            f"ship deck voyage weather log chapter {turn} rigging lantern watch",
            {"turn": turn},
            token_count=900,
        )
    assert sdm.total_tokens_ingested >= 50_000
    assert sdm.resident_counter_bytes() == ram_before  # Fixed RAM!

    recall = sdm.read("forged sea chart inside brass astrolabe")
    assert recall.found is True
    assert recall.episode is not None
    assert recall.episode.episode_id == "turn_1_astrolabe"
    assert recall.episode.payload["container"] == "brass_astrolabe"


def test_organ3_ppr_top32_sequitur_menger_sybil_and_fca():
    # Bit-Parallel PPR strictly caps working set <= 32 on a 120-node hub graph
    ppr = BitParallelPPRWorkingMemory(max_working_set=32)
    for i in range(120):
        ppr.add_edge("hub_center", f"leaf_{i}")
    ppr.add_edge("seed_target", "neighbor_a")
    ppr.add_edge("neighbor_a", "hub_center")
    ws = ppr.select_working_set(["seed_target"], top_k=32)
    assert len(ws) <= 32
    assert ws[0][0] == "seed_target"
    assert any(node == "neighbor_a" for node, _sc in ws[:4])

    # O(N) Sequitur Predictive-MDL Symbol Minter
    seq = SequiturSymbolMinter()
    tokens = ["open", "lock", "write", "commit", "close"] * 6
    sres = seq.mint(tokens)
    assert sres.mdl_gain_symbols > 0
    assert sres.grammar_length < sres.original_length
    assert len(sres.rules) >= 1

    # Stage 9 Menger Provenance Verifier: Rejects 50-node Sybil farm with 1 bottleneck!
    sybil = MengerProvenanceVerifier(required_k=3)
    sybil.add_trusted_root("root_web")
    sybil.add_attestation_edge("root_web", "compromised_origin")
    for i in range(50):
        blog = f"sybil_blog_{i}"
        sybil.add_attestation_edge("compromised_origin", blog)
        sybil.add_attestation_edge(blog, "poisoned_claim")
    audit_sybil = sybil.verify_claim("poisoned_claim")
    assert audit_sybil.admitted is False
    assert audit_sybil.independent_paths_k == 1
    assert audit_sybil.bottleneck_detected is True

    # Admits 3 genuinely vertex-disjoint independent attestation chains
    honest = MengerProvenanceVerifier(required_k=3)
    for idx in range(3):
        honest.add_trusted_root(f"trusted_root_{idx}")
        honest.add_attestation_edge(f"trusted_root_{idx}", f"independent_journal_{idx}")
        honest.add_attestation_edge(f"independent_journal_{idx}", "verified_claim")
    audit_honest = honest.verify_claim("verified_claim")
    assert audit_honest.admitted is True
    assert audit_honest.independent_paths_k == 3

    # Stage 11 FCA Galois Concept Splitter splits polysemous 'bank' cleanly
    splitter = FCAGaloisConceptSplitter()
    clusters = splitter.split_concept(
        "bank",
        {
            "b1": ["river", "water", "erosion", "mud"],
            "b2": ["river", "water", "erosion", "bridge"],
            "b3": ["deposit", "loan", "interest", "vault"],
            "b4": ["deposit", "loan", "interest", "branch"],
        },
    )
    assert len(clusters) == 2
    extents = {c.extent_instances for c in clusters}
    assert frozenset({"b1", "b2"}) in extents
    assert frozenset({"b3", "b4"}) in extents

    distiller = SubmodularMDLRuleDistiller()
    picked = distiller.distill(
        {
            "r_broad": ({"t1", "t2", "t3", "t4"}, 12),
            "r_dup": ({"t1", "t2"}, 12),
            "r_tail": ({"t5", "t6"}, 10),
        }
    )
    picked_ids = [r for r, _g in picked]
    assert picked_ids == ["r_broad", "r_tail"]


# ===========================================================================
# ORGAN 1: Bilingual CCG-HDC, Tropical-RSA, Reference Frames & Active Inference
# ===========================================================================


def test_organ1_bilingual_ccg_4hop_xor_abduction_and_tropical_rsa():
    parser = BilingualCCGHDCParser()
    en_he = parser.parse_clause("Amina hid the brass astrolabe")
    ur_he = parser.parse_clause("Amina ne brass astrolabe chhupaya hai")
    assert en_he.language == "english" and en_he.who == "amina"
    assert ur_he.language == "roman_urdu" and ur_he.who == "amina"

    # 4-hop Telescoping XOR Abduction (English)
    en_premises = [
        "if transformer overheats, then circuit breaker trips",
        "if circuit breaker trips, then cooling pump stops",
        "if cooling pump stops, then server rack throttles",
        "if server rack throttles, then api latency spikes",
    ]
    abd_en = parser.abduce_k_hop(en_premises, "api latency spikes")
    assert abd_en.found is True
    assert abd_en.hops == 4
    assert abd_en.root_cause == "transformer overheats"
    assert abd_en.xor_verification_q16 == Q16

    # 4-hop Telescoping XOR Abduction (Roman Urdu)
    ur_premises = [
        "agar bijli gayi to router band hua",
        "agar router band hua to internet cut gaya",
        "agar internet cut gaya to sync ruk gaya",
        "agar sync ruk gaya to alert aaya",
    ]
    abd_ur = parser.abduce_k_hop(ur_premises, "alert aaya")
    assert abd_ur.found is True
    assert abd_ur.hops == 4
    assert abd_ur.root_cause == "bijli gayi"
    assert abd_ur.xor_verification_q16 == Q16

    # Stage 5B Tropical-RSA: Scalar Implicature, Indirect Speech Acts (EN + Urdu), Irony
    rsa = TropicalRSAPragmaticEngine()
    sc = rsa.infer("Some of the database replicas survived the outage.")
    assert sc.phenomenon == "scalar_implicature"
    assert "not all" in sc.pragmatic_intent.casefold()

    ind_en = rsa.infer("It's freezing in here.")
    ind_ur = rsa.infer("Yahan bohat thand hai.")
    assert ind_en.phenomenon == "indirect_speech_act" and "window" in ind_en.pragmatic_intent.casefold()
    assert ind_ur.phenomenon == "indirect_speech_act" and "window" in ind_ur.pragmatic_intent.casefold()

    irony = rsa.infer("Oh great, the production database crashed again!")
    assert irony.phenomenon == "irony_sarcasm"
    assert irony.affect_polarity_q16 < 0


def test_organ1_occ22_tom_cmp_and_active_inference_gbs():
    assert len(OCC_22_EMOTIONS) == 22

    amina = CharacterReferenceFrame(char_id="amina")
    tariq = CharacterReferenceFrame(char_id="tariq")
    amina.observe_fact("forged_sea_chart")
    assert amina.knows_fact("forged_sea_chart") is True
    assert tariq.knows_fact("forged_sea_chart") is False

    em = amina.appraise_event(desirability_q16=-25_000, expected_q16=10_000, actor_id="tariq", praiseworthy_q16=-20_000)
    assert em == "anger"
    assert amina.affect_q16[2] > 8_000  # Tension rose

    # Generalized Binary Search exact 50/50 split selection
    ctrl = CorticalActiveInferenceController()
    hyps = [("h1", 10), ("h2", 10), ("h3", 10), ("h4", 10)]
    probe, gain = ctrl.generalized_binary_search_split(
        hyps,
        {"skewed_probe": {"h1"}, "balanced_probe": {"h1", "h2"}},
    )
    assert probe == "balanced_probe"
    assert gain == Q16

    # CMP lateral vote detects overlapping write footprint conflicts
    bad_frames = [
        ModuleReferenceFrame("m1", "a", ("user_prompt",), ("x",), CompactRoaringBitmap(), CompactRoaringBitmap([7]), ()),
        ModuleReferenceFrame("m2", "b", ("x",), ("y",), CompactRoaringBitmap(), CompactRoaringBitmap([7]), ()),
    ]
    bad_vote = CorticalCMPVoter.vote_module_graph(bad_frames)
    assert bad_vote.consensus_reached is False
    assert len(bad_vote.write_conflicts) == 1


# ===========================================================================
# ORGAN 4: Stage 12 Multi-File CEGIS Architect, Bisection Repair & Stitch MDL
# ===========================================================================


def test_organ4_ast_anti_unification_and_stage12_1000line_sandbox_verified(monkeypatch):
    monkeypatch.setenv("MIRROR_SANDBOX_MODE", "process")

    # Stage 3 Plotkin First-Order AST Anti-Unification
    au = FirstOrderASTAntiUnifier()
    res_au = au.anti_unify_sources(
        "def add_tax(price):\n    return price + 15\n",
        "def add_fee(amount):\n    return amount + 40\n",
    )
    assert res_au.similarity_q16 > 40_000
    assert len(res_au.subst_a) >= 2 and len(res_au.subst_b) >= 2

    # Stage 12 End-to-End: 1 vague sentence -> ~900+ line 8-module app passing ZeroTrustSandboxJudge!
    architect = MultiFileCEGISArchitect()
    build = architect.synthesize_architecture(
        "Build a local-first collaborative habit tracker with streak analytics and reminders."
    )
    assert build.module_count == 8
    assert build.total_lines >= 650
    assert build.spec.cmp_vote.consensus_reached is True
    assert build.tsetlin_gate.passed is True
    assert build.sandbox_verified is True, f"Sandbox failed: {build.sandbox_output}"
    assert "ALL_PROPERTY_TESTS_PASSED:7" in build.sandbox_output
    assert build.symbol_sdm_entries >= 10

    # Verify Stitch MDL sleep-phase primitive invention finds U(A) > 0 helpers
    stitch = StitchMDLLibraryLearner()
    primitives = stitch.invent_primitives(build.modules)
    assert len(primitives) >= 1
    assert all(p.mdl_utility_gain > 0 for p in primitives)

    # Verify a second domain (expense ledger with double-entry conservation) & third (task DAG)
    build_ledger = architect.synthesize_architecture(
        "Build an offline-first collaborative expense ledger with budget splits and reminders."
    )
    assert build_ledger.spec.domain_family == "expense_ledger"
    assert build_ledger.sandbox_verified is True
    assert "ALL_PROPERTY_TESTS_PASSED:7" in build_ledger.sandbox_output

    build_dag = architect.synthesize_architecture(
        "Build a local Kanban task board with dependency graphs and reminders."
    )
    assert build_dag.spec.domain_family == "task_board"
    assert build_dag.sandbox_verified is True
    assert "ALL_PROPERTY_TESTS_PASSED:7" in build_dag.sandbox_output


def test_organ4_log_m_call_graph_bisection_repair(monkeypatch):
    monkeypatch.setenv("MIRROR_SANDBOX_MODE", "process")
    architect = MultiFileCEGISArchitect()
    # Inject a deliberate CRDT commutativity tie-break bug into crdt_sync.py!
    build = architect.synthesize_architecture(
        "Build a local-first collaborative habit tracker with streak analytics and reminders.",
        simulate_fault_in_module="crdt_sync.py",
    )
    assert build.bisection_report is not None
    assert build.bisection_report.bug_detected is True
    assert build.bisection_report.culprit_module == "crdt_sync.py"
    assert build.bisection_report.probes_used <= build.bisection_report.max_log_probes
    assert build.bisection_report.repaired is True
    assert build.sandbox_verified is True
    assert "ALL_PROPERTY_TESTS_PASSED:7" in build.sandbox_output


# ===========================================================================
# ORGAN 5: 0-Float-MatMul Ternary LUT, I-LLM Ops, PyTorch Upgrade & Stage 13
# ===========================================================================


def test_organ5_zero_float_matmul_ternary_lut_bit_exact_and_illm_ops():
    rng = random.Random(123)
    out_f, in_f = 19, 37  # Non-multiples of 3 and 4 to test padding edges!
    w = [[rng.choice((-1, 0, 1)) for _ in range(in_f)] for _ in range(out_f)]
    x_int = [rng.randint(-120, 120) for _ in range(in_f)]
    ref = [sum(w[o][i] * x_int[i] for i in range(in_f)) for o in range(out_f)]

    kernel = TernaryLUTLinearKernel(w, kernel_mode="tmac_g4")
    y_tmac = kernel.matvec_int(x_int, mode="tmac_g4")
    y_tl2 = kernel.matvec_int(x_int, mode="tl2")
    assert y_tmac == ref
    assert y_tl2 == ref
    assert kernel.float_matmuls_executed == 0
    assert kernel.lut_lookups_executed > 0

    # Serialization round-trip
    restored = TernaryLUTLinearKernel.from_dict(kernel.to_dict())
    assert restored.matvec_int(x_int, mode="tmac_g4") == ref
    assert restored.matvec_int(x_int, mode="tl2") == ref

    # I-LLM strict integer Q16.16 ops
    logits = [Q16 * 2, Q16, 0, -Q16]
    probs = I_LLM_IntegerOps.di_clipped_softmax_q16(logits)
    assert sum(probs) == Q16
    assert probs[0] > probs[1] > probs[2] > probs[3] > 0

    normed = I_LLM_IntegerOps.di_rmsnorm_q16([Q16 * 2, -Q16 * 2, Q16 * 2, -Q16 * 2])
    assert len(normed) == 4 and abs(normed[0]) > 50_000
    assert I_LLM_IntegerOps.di_silu_q16(Q16 * 2) > Q16


def test_organ5_training_engine_lut_wiring_ast_and_optional_torch():
    # Always verify training/cpu_autonomous_engine.py syntax and 0-float-matmul LUT wiring
    import ast as _ast

    eng_path = Path(__file__).resolve().parents[1] / "training" / "cpu_autonomous_engine.py"
    src = eng_path.read_text(encoding="utf-8")
    tree = _ast.parse(src)
    cls_names = {n.name for n in tree.body if isinstance(n, _ast.ClassDef)}
    assert "TernaryBitLinear" in cls_names
    assert "TernaryLUTLinearKernel" in src
    assert "zero_float_matmul_lut:" in src

    try:
        import torch
    except ImportError:
        return  # Pure-CPU Python stdlib environment without optional PyTorch

    from training.cpu_autonomous_engine import TernaryBitLinear

    torch.manual_seed(0)
    layer = TernaryBitLinear(24, 12, bias=True)
    stats = layer.freeze_for_cpu_inference(kernel_mode="tmac_g4")
    assert stats["zero_float_matmul_ready"] == 1.0
    layer.eval()

    x = torch.randn(2, 24)
    y_lut = layer(x)
    assert layer.last_inference_backend == "zero_float_matmul_lut:tmac_g4"
    assert layer._lut_kernel.float_matmuls_executed == 0
    assert layer._lut_kernel.lut_lookups_executed > 0
    assert y_lut.shape == (2, 12)

    layer.use_lut_inference = False
    y_ref = layer(x)
    max_diff = float((y_lut - y_ref).abs().max().item())
    assert max_diff < 0.08


def test_organ5_stage13_canon_locked_tom_safe_narrative_walkthrough():
    # Turn 1 vs Turn 34 Astrolabe walkthrough from docs/38
    sdm = KanervaSDMInt8(num_hard_locations=256, sdr_dim=256, top_k_sdr=16, value_dim=64)
    sdm.write(
        "turn_1_secret",
        "amina hid forged_sea_chart inside brass_astrolabe in captain cabin",
        {"secret": "forged_sea_chart"},
        salience_q8=512,
        token_count=35_000,
    )
    recalled = sdm.read("brass_astrolabe forged_sea_chart")
    assert recalled.found is True

    amina = CharacterReferenceFrame(char_id="amina", voice_steering_id="style_defensive_subtext")
    amina.observe_fact("forged_sea_chart")
    amina.observe_fact("brass_astrolabe")
    amina.appraise_event(desirability_q16=-22_000, expected_q16=12_000)

    tariq = CharacterReferenceFrame(char_id="tariq", voice_steering_id="style_melancholic_understatement")
    # Tariq was outside the cabin -> does NOT observe forged_sea_chart!
    assert tariq.knows_fact("forged_sea_chart") is False

    world = TwinEGraphDoCalculus()
    world.assert_world_fact("captain_alive", False, world="canon")

    realizer = LTAGKneserNeyRealizer()
    rsa = TropicalRSAPragmaticEngine()
    prag = rsa.infer("The tides are calm tonight, Tariq.")

    # Tariq speaks: must NEVER leak 'forged_sea_chart' (ToM isolation) & canon blocks dead captain speaking!
    res = realizer.realize_narrative_turn(
        scene_prompt="confrontation over missing cargo with quiet tension",
        speaker=tariq,
        listener=amina,
        world_gate=world,
        pragmatic=prag,
        mode="canon",
        forbidden_tom_terms=("forged_sea_chart",),
        candidate_claims=[
            ("grief_was_a_rusted_anchor", True, True),  # figurative -> permitted by M_canon
            ("captain_alive", True, False),  # contradicts story canon -> blocked by M_canon
        ],
    )
    assert "forged_sea_chart" not in res.text.casefold()
    assert res.tom_leaks_blocked >= 1
    assert res.canon_violations_blocked == 1
    assert res.zero_float_matmul_verified is True
    assert res.kneser_ney_score_q16 > 0


# ===========================================================================
# TRANSC0MPILER, SHARD INTEGRITY & BACKEND END-TO-END ROUTING
# ===========================================================================


def test_transcompiler_script_artifact_and_74741_shard_integrity(tmp_path):
    from scripts.transcompile_open_weights_to_lut import build_stage12_13_artifact

    custom_w = tmp_path / "open_weights.json"
    custom_w.write_text(
        json.dumps({"layer0": [[0.9, -0.8, 0.05, 1.1], [-0.7, 0.02, 0.85, -0.95]]}),
        encoding="utf-8",
    )
    out_art = tmp_path / "compiled_lut_sae.json"
    art = build_stage12_13_artifact(out_art, input_weights_json=custom_w)
    assert out_art.is_file()
    assert art["zero_float_matmul_guaranteed"] is True
    assert art["transcompiled_lut_layers"]["layer0"]["bit_exact_verified"] is True

    # Repository artifact exists and Concept Lattice stays exact 74,741 (0 vault overwrites!)
    repo_art = Path(__file__).resolve().parents[1] / "artifacts" / "mirror_stage12_13_lut_sae.json"
    assert repo_art.is_file()
    lat = OpenEndedConceptLatticeEngine()
    try:
        assert lat.concept_count() == 74_741
    finally:
        lat.close()


def test_backend_service_unified_organs_status_and_crystal_ask_routing(monkeypatch):
    monkeypatch.setenv("MIRROR_SANDBOX_MODE", "process")
    from mirror7_backend.service import BackendService

    svc = BackendService(enable_crystal_auto_load=False)
    status = svc.unified_organs_status()
    assert status["ok"] is True
    assert status["hypervector_bits"] == 10_240
    assert status["zero_float_matmul_lut_verified"] is True
    assert len(status["organs"]) == 5
    assert status["lut_sae_artifact_present"] is True

    # End-to-end BackendService.crystal_ask routing for Stages 12, 13, 5B, 7
    r12 = svc.crystal_ask(
        "Synthesize architecture: Build a local-first collaborative habit tracker with streak analytics and reminders"
    )
    assert r12["abstained"] is False
    assert "sandbox_verified=True" in r12["answer"]
    assert "8 modules" in r12["answer"]

    r13 = svc.crystal_ask(
        "Synthesize narrative: Amina speaks politely of calm tides while hiding the brass astrolabe"
    )
    assert r13["abstained"] is False
    assert "amina" in r13["answer"].casefold()

    r5b = svc.crystal_ask("Analyze subtext: It's freezing in here.")
    assert r5b["abstained"] is False
    assert "window" in r5b["answer"].casefold()

    r7 = svc.crystal_ask("Do-calculus: rain->wet, wet->slippery | do(wet=false)")
    assert r7["abstained"] is False
    assert "do(wet=False)" in r7["answer"]
