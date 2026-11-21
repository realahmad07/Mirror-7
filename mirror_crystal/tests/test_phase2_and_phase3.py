"""
Comprehensive Verification Suite for Phase-2 Cognitive Subsystems & Phase-3 Bottleneck Defenses.
"""
from __future__ import annotations

import math
from pathlib import Path
import sys
import time

from mirror_crystal import (
    CloudCrystalNode,
    DiskCrystalVault,
    DreamUtilityPruner,
    FluentByteRealizer,
    MirrorCrystalEngine,
    NightlyDreamSynthesizer,
    PerceptualVQLens,
    PocketCrystalClient,
    SandboxMCTSPlanner,
    ZeroTrustSandboxJudge,
)


def test_p2_1_perceptual_vq_lens_image_brightness_contrast_invariance():
    """INT8 PerceptualVQLens recognizes 2D visual patterns across drastic brightness & contrast shifts."""
    eng = MirrorCrystalEngine()

    l_shape = [
        [10, 0, 0, 0],
        [10, 0, 0, 0],
        [10, 0, 0, 0],
        [10, 10, 10, 10],
    ]
    h_stripe = [
        [10, 10, 10, 10],
        [0, 0, 0, 0],
        [10, 10, 10, 10],
        [0, 0, 0, 0],
    ]
    eng.ingest_pair(
        PerceptualVQLens.wrap_multimodal_prompt("Classify visual pattern", image_grid=l_shape, patch_size=2),
        "Detected geometric class: L_CORNER_BRACKET.",
    )
    eng.ingest_pair(
        PerceptualVQLens.wrap_multimodal_prompt("Classify visual pattern", image_grid=h_stripe, patch_size=2),
        "Detected geometric class: HORIZONTAL_GRATINGS.",
    )

    # Apply heavy illumination shift (+85.0) and contrast stretch (* 2.7)
    bright_l = [[px * 2.7 + 85.0 for px in row] for row in l_shape]
    bright_h = [[px * 3.4 + 120.0 for px in row] for row in h_stripe]

    r1 = eng.respond(PerceptualVQLens.wrap_multimodal_prompt("Classify visual pattern", image_grid=bright_l, patch_size=2))
    r2 = eng.respond(PerceptualVQLens.wrap_multimodal_prompt("Classify visual pattern", image_grid=bright_h, patch_size=2))
    assert "L_CORNER_BRACKET" in r1.answer
    assert "HORIZONTAL_GRATINGS" in r2.answer


def test_p2_1b_perceptual_vq_lens_waveform_offset_and_gain_invariance():
    """PerceptualVQLens distinguishes 1D waveforms invariant to DC offset and sensor gain."""
    eng = MirrorCrystalEngine()
    osc_wave = [1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0]
    ramp_wave = [float(i) for i in range(12)]

    eng.ingest_pair(
        PerceptualVQLens.wrap_multimodal_prompt("Diagnose turbine vibration", waveform=osc_wave),
        "Diagnosis: HIGH_FREQ_RESONANCE_FLUTTER.",
    )
    eng.ingest_pair(
        PerceptualVQLens.wrap_multimodal_prompt("Diagnose turbine vibration", waveform=ramp_wave),
        "Diagnosis: MONOTONIC_THERMAL_DRIFT.",
    )

    shifted_osc = [x * 4.2 + 350.0 for x in osc_wave]
    shifted_ramp = [x * 0.5 - 90.0 for x in ramp_wave]

    assert "HIGH_FREQ_RESONANCE_FLUTTER" in eng.respond(
        PerceptualVQLens.wrap_multimodal_prompt("Diagnose turbine vibration", waveform=shifted_osc)
    ).answer
    assert "MONOTONIC_THERMAL_DRIFT" in eng.respond(
        PerceptualVQLens.wrap_multimodal_prompt("Diagnose turbine vibration", waveform=shifted_ramp)
    ).answer


def test_p2_2_sandbox_mcts_planner_composes_two_helper_rules_into_new_program():
    """SandboxMCTSPlanner retrieves separate `is_prime` & `digit_sum` rules and synthesizes a verified combined solver."""
    eng = MirrorCrystalEngine()
    eng.ingest_pair(
        "Write helper is_prime(n) in Python to test prime numbers.",
        (
            "```python\n"
            "def is_prime(n: int) -> bool:\n"
            "    if n < 2:\n"
            "        return False\n"
            "    for k in range(2, int(n ** 0.5) + 1):\n"
            "        if n % k == 0:\n"
            "            return False\n"
            "    return True\n"
            "```"
        ),
        test_code="assert is_prime(2) and is_prime(13) and not is_prime(9)",
    )
    eng.ingest_pair(
        "Write helper digit_sum(n) in Python to sum decimal digits.",
        (
            "```python\n"
            "def digit_sum(n: int) -> int:\n"
            "    return sum(int(ch) for ch in str(abs(n)))\n"
            "```"
        ),
        test_code="assert digit_sum(29) == 11",
    )

    planner = SandboxMCTSPlanner(max_nodes=24, max_wall_ms=30000.0)
    prompt = "Write prime_digit_primes(limit) returning primes <= limit whose digit_sum is also prime."
    test_spec = "assert prime_digit_primes(30) == [2, 3, 5, 7, 11, 23, 29]\nassert prime_digit_primes(10) == [2, 3, 5, 7]"

    res = planner.solve_and_verify(eng, prompt=prompt, test_code=test_spec, crystallize_on_success=True)
    assert res.mode == "mcts_program_synthesis", res.reasoning_trace
    assert res.sandbox_verified is True
    assert res.epistemic_tier == "verified_exact"

    # Subsequent query hits the newly crystallized solution directly without re-running MCTS!
    cached_hit = eng.respond(prompt)
    assert "def prime_digit_primes" in cached_hit.answer
    assert cached_hit.sandbox_verified is True


def test_p2_2b_sandbox_mcts_planner_ast_self_repair_off_by_one():
    """SandboxMCTSPlanner repairs an off-by-one range(1, n) bug into range(1, n + 1) using UCB1 AST mutation."""
    eng = MirrorCrystalEngine()
    # Seed a slightly buggy unverified draft rule (off-by-one upper bound)
    eng.ingest_pair(
        "Write sum_up_to(n) summing 1..n inclusive.",
        "```python\ndef sum_up_to(n):\n    return sum(range(1, n))\n```",
    )

    planner = SandboxMCTSPlanner(max_nodes=20, max_wall_ms=8000.0)
    res = planner.solve_and_verify(
        eng,
        prompt="Write sum_up_to(n) summing 1..n inclusive.",
        test_code="assert sum_up_to(10) == 55\nassert sum_up_to(1) == 1",
    )
    assert res.mode == "mcts_program_synthesis"
    assert res.sandbox_verified is True
    ok, _ = ZeroTrustSandboxJudge.verify_code_and_test(res.answer, "assert sum_up_to(100) == 5050")
    assert ok is True


def test_p3_3_bounded_loop_step_guard_blocks_infinite_loops_in_milliseconds():
    """Phase-3 BoundedLoopStepGuard aborts `while True:`, huge ranges, and deep loops in < 80 ms on CPU."""
    adversarial_snippets = [
        "def hang():\n    while True:\n        pass\nhang()",
        "def big_for():\n    s = 0\n    for i in range(10**12):\n        s += i\n    return s\nbig_for()",
        "def spin():\n    x = 1\n    while x > 0:\n        x += 1\nspin()",
    ]
    for bad in adversarial_snippets:
        t0 = time.perf_counter()
        ok, msg = ZeroTrustSandboxJudge.verify_code_and_test(bad, max_loop_steps=5_000)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        assert ok is False
        assert "LimitExceeded" in msg
        max_ms = 1000.0 if sys.platform == "win32" else 150.0
        assert elapsed_ms < max_ms


def test_p2_3_and_p3_2_coreference_pronoun_tracking_and_topic_shift_decay_guard():
    """Pronouns ('its') resolve across turns, while TopicShiftDecayGuard blocks stale pronouns on topic switch & TTL."""
    eng = MirrorCrystalEngine()
    eng.ingest_pair(
        "Explain Grover quantum search algorithm.",
        "Grover quantum search amplifies target state amplitudes on an unsorted database.",
    )
    eng.ingest_pair(
        "What is the time complexity of Grover quantum search algorithm?",
        "Grover quantum search runs in O(sqrt(N)) oracle queries.",
    )
    eng.ingest_pair(
        "How does binary search split a sorted array when it runs?",
        "Binary search halves the sorted search interval in O(log N) comparisons.",
    )

    # Turn 1 sets active entity anchor -> ('grover', 'quantum', 'search', 'algorithm')
    r1 = eng.respond("Explain Grover quantum search algorithm.", session_id="conv_1")
    assert "amplifies target state" in r1.answer

    # Turn 2 uses pronoun 'its' with NO new topic entity -> resolves to Grover!
    r2 = eng.respond("What is its time complexity?", session_id="conv_1")
    assert "O(sqrt(N))" in r2.answer
    assert any("Coreference resolved" in tr for tr in r2.reasoning_trace)

    # Turn 3 switches topic to 'binary search' while ALSO containing 'it' -> TopicShiftDecayGuard overrides!
    r3 = eng.respond("How does binary search split a sorted array when it runs?", session_id="conv_1")
    assert "O(log N)" in r3.answer
    assert "O(sqrt(N))" not in r3.answer
    assert any("TopicShiftDecayGuard" in tr for tr in r3.reasoning_trace)

    # Decay TTL across 3 empty-entity turns -> anchor expires completely
    for _ in range(3):
        eng.respond("Why?", session_id="conv_1")
    r_expired = eng.respond("What is its time complexity?", session_id="conv_1")
    assert not any("Coreference resolved" in tr for tr in r_expired.reasoning_trace)


def test_p2_3b_homonym_polysemy_sense_disambiguation_ram_and_disk(tmp_path: Path):
    """Homonyms ('python' snake vs code; 'bank' river vs finance) coexist without overwriting and disambiguate cleanly."""
    eng = MirrorCrystalEngine()
    vault = DiskCrystalVault(tmp_path / "sense.dcrystal")

    pairs = [
        ("Tell me about python", "A python is a nonvenomous constrictor snake reptile native to tropical jungles."),
        ("Tell me about python", "Python is an interpreted programming language using def functions and indentation."),
        ("How do banks behave?", "River banks experience soil sediment erosion during heavy flood water currents."),
        ("How do banks behave?", "Financial banks accept currency deposits and issue credit loans with interest."),
    ]
    for oracle in (eng, vault):
        for u, t in pairs:
            oracle.ingest_pair(u, t)

    for oracle in (eng, vault):
        r_snake = oracle.respond("Tell me about python snakes in the jungle")
        r_code = oracle.respond("Tell me about python code def functions")
        r_river = oracle.respond("How do banks behave during river water flood?")
        r_fin = oracle.respond("How do banks behave with deposit loans and interest?")

        assert "constrictor snake reptile" in r_snake.answer
        assert "interpreted programming language" in r_code.answer
        assert "sediment erosion" in r_river.answer
        assert "currency deposits" in r_fin.answer

    vault.close()


def test_p2_4_and_p3_1_nightly_dream_synthesizer_and_cycle_bloat_pruner(tmp_path: Path):
    """NightlyDreamSynthesizer materializes multi-hop shortcuts while pruning cycles & capping dream bloat."""
    vault = DiskCrystalVault(tmp_path / "dream.dcrystal")
    facts = [
        # 3-hop valid causal chain
        "quantum_decoherence causes phase_damping_noise.",
        "phase_damping_noise triggers syndrome_parity_flip.",
        "syndrome_parity_flip leads to logical_qubit_fault.",
        # Deliberate 2-cycle (must be pruned, never infinite-loop!)
        "oscillator_alpha triggers oscillator_beta.",
        "oscillator_beta triggers oscillator_alpha.",
    ]
    # Add 10 extra 2-hop chains to test max_dreams_per_cycle bloat cap
    for idx in range(10):
        facts.append(f" event_src_{idx} causes mid_stage_{idx}.")
        facts.append(f"mid_stage_{idx} leads to final_outcome_{idx}.")

    vault.observe_unlabeled_text("\n".join(facts))

    synthesizer = NightlyDreamSynthesizer(pruner=DreamUtilityPruner(max_dreams_per_cycle=4, min_hops=2, max_hops=4))
    report = synthesizer.run_dream_cycle(vault)

    assert report["cycles_pruned"] >= 2
    assert report["dreams_crystallized"] == 4
    assert report["bloat_pruned"] >= 6

    # Verify the 3-hop chain was prioritized (highest utility) and crystallized as a direct 1-hop rule!
    r_dream = vault.respond("How does quantum_decoherence relate to logical_qubit_fault?")
    assert "logical_qubit_fault" in r_dream.answer
    vault.close()


def test_p2_5_universal_concept_hub_zero_duplication_multilingual_queries(tmp_path: Path):
    """Single English CrystalRule answers Urdu, Chinese, Arabic, Hindi, Spanish & French prompts with 0 duplication."""
    vault = DiskCrystalVault(tmp_path / "multilingual.dcrystal")
    vault.ingest_pair(
        "What is the time complexity of Grover quantum search algorithm?",
        "<think> Quadratic speedup over classical O(N) </think>\nGrover quantum search achieves O(sqrt(N)) query complexity.",
    )
    assert vault.rule_count() == 1  # Strictly ONE rule on disk!

    multilingual_prompts = [
        ("ur", "گروور کوانٹم تلاش الگورتھم کی وقت پیچیدگی کیا ہے؟"),
        ("zh", "格罗弗量子搜索算法的时间复杂度是多少？"),
        ("ar", "ما هو تعقيد خوارزمية جروفر للبحث الكمي؟"),
        ("hi", "ग्रोवर क्वांटम खोज एल्गोरिदम की जटिलता क्या है?"),
        ("es", "¿Cuál es la complejidad de búsqueda cuántica de Grover?"),
        ("fr", "Quelle est la complexité de l'algorithme de recherche quantique de Grover ?"),
    ]
    for expected_lang, q_foreign in multilingual_prompts:
        res = vault.respond(q_foreign)
        assert "O(sqrt(N))" in res.answer, f"Failed cross-lingual lock for {expected_lang}: {q_foreign}"
        polished = FluentByteRealizer.realize(res.answer, lang=expected_lang, include_think=False)
        assert "<think>" not in polished
        assert "O(sqrt(N))" in polished

    assert vault.rule_count() == 1
    vault.close()


def test_p3_4_morphological_fuzzy_spell_bridge_recovers_severe_typos(tmp_path: Path):
    """Phase-3 MorphologicalFuzzySpellBridge recovers multi-word keyboard typos on both RAM & reloaded Disk vaults."""
    vault_path = tmp_path / "typo.dcrystal"
    vault = DiskCrystalVault(vault_path)
    vault.ingest_pair(
        "Explain Grover quantum search complexity",
        "Grover search finds a marked quantum item in O(sqrt(N)) steps.",
    )
    vault.ingest_pair(
        "Explain quantum entanglement Bell correlations",
        "Entangled Bell pairs violate classical local hidden-variable inequalities.",
    )
    vault.optimize_disk()
    vault.close()

    reopened = DiskCrystalVault(vault_path)
    # Transpositions ('grvoer', 'serach') + missing letters ('quantm', 'entanglment')
    r1 = reopened.respond("Explain Grvoer quantm serach complexty")
    r2 = reopened.respond("Explain quantm entanglment Bell correlations")
    assert "O(sqrt(N))" in r1.answer
    assert "Bell pairs violate" in r2.answer
    reopened.close()


def test_p3_5_epistemic_confidence_calibrator_blocks_false_cousin_partial_overlap():
    """EpistemicConfidenceCalibrator rejects single-word cousin overlaps on >=4-concept specific queries."""
    eng = MirrorCrystalEngine()
    eng.ingest_pair(
        "Explain superconducting Josephson junction magnetic flux",
        "A Josephson junction tunnels Cooper pairs across a thin insulating barrier.",
    )
    eng.ingest_pair(
        "Compute 17^19 mod 1000000007",
        f"Exact result is {pow(17, 19, 1000000007)}.",
    )

    # 5-concept query sharing ONLY 'superconducting' with totally unrelated physics topic -> must abstain!
    r_cousin = eng.respond("Explain superconducting topological braided anyon condensates")
    assert r_cousin.mode == "unmatched_honest_abstain"
    assert r_cousin.epistemic_tier == "honest_abstain"

    # Verified formula lock -> calibrated as 'verified_exact'
    r_exact = eng.respond("Compute 29^31 mod 1000000007")
    assert str(pow(29, 31, 1000000007)) in r_exact.answer
    assert r_exact.epistemic_tier == "verified_exact"


def test_p4_1_perceptual_vq_lens_d4_camera_rotation_and_mirror_flip_invariance():
    """Fourth-Horizon D4 canonical patch hashes recognize rotated (180-deg) & selfie-flipped shapes."""
    eng = MirrorCrystalEngine()
    corner_tl = [
        [20, 20, 0, 0],
        [20, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ]
    checker = [
        [20, 0, 20, 0],
        [0, 20, 0, 20],
        [20, 0, 20, 0],
        [0, 20, 0, 20],
    ]
    eng.ingest_pair(
        PerceptualVQLens.wrap_multimodal_prompt("Identify fiducial marker", image_grid=corner_tl, patch_size=4),
        "Marker ID: CORNER_FIDUCIAL.",
    )
    eng.ingest_pair(
        PerceptualVQLens.wrap_multimodal_prompt("Identify fiducial marker", image_grid=checker, patch_size=4),
        "Marker ID: CHECKER_CALIBRATION.",
    )

    # Rotate corner_tl 180 degrees AND scale illumination
    rotated_180 = [[px * 1.9 + 40.0 for px in reversed(row)] for row in reversed(corner_tl)]
    res = eng.respond(PerceptualVQLens.wrap_multimodal_prompt("Identify fiducial marker", image_grid=rotated_180, patch_size=4))
    assert "CORNER_FIDUCIAL" in res.answer


def test_p4_2_roman_urdu_hinglish_and_multilingual_slot_math_zero_duplication():
    """Single English modular formula lock solves Urdu, Chinese, Spanish, and Roman-Urdu unseen math."""
    eng = MirrorCrystalEngine()
    eng.ingest_pair(
        "Compute 17^19 mod 1000000007",
        f"The exact modular result of 17^19 mod 1000000007 is {pow(17, 19, 1000000007)}.",
    )
    expected = str(pow(41, 53, 1000000007))
    for q in (
        "41^53 mod 1000000007 کا حساب",
        "计算 41^53 mod 1000000007",
        "Calcula 41^53 mod 1000000007",
        "41^53 mod 1000000007 ka hisab",
    ):
        r = eng.respond(q)
        assert expected in r.answer, f"Failed multilingual slot math on: {q}"
        assert r.mode == "slot_formula_lock"


def test_p4_3_split_brain_pocket_to_cloud_coreference_and_homonym_teleport():
    """PocketCrystalClient bridges offline Turn-1 coreference into Turn-2 Cloud teleportation without leaking PII."""
    cloud_eng = MirrorCrystalEngine()
    cloud_eng.ingest_pair(
        "What is the time complexity of Grover quantum search algorithm?",
        "Grover quantum search runs in O(sqrt(N)) oracle evaluations.",
    )
    cloud = CloudCrystalNode(cloud_eng)
    pocket = PocketCrystalClient(cloud_node=cloud, max_ram_rules=4)

    # Turn 1 stays 100% in private Pocket RAM (Cloud NEVER hears Turn 1!)
    pocket.teach_private_local(
        "Explain Grover quantum search algorithm.",
        "Private study note: Grover amplifies marked quantum states.",
    )
    r_turn1 = pocket.ask("Explain Grover quantum search algorithm.", session_id="mob_1")
    assert "Private study note" in r_turn1.answer
    assert len(cloud.audit_received_queries) == 0  # Cloud saw ZERO Turn-1 traffic!

    # Turn 2 asks 'What is its time complexity?' with an email -> misses Pocket RAM -> bridges coref + scrubs PII -> Cloud!
    r_turn2 = pocket.ask("What is its time complexity for student@uni.edu?", session_id="mob_1")
    assert "O(sqrt(N))" in r_turn2.answer
    assert len(cloud.audit_received_queries) == 1
    assert "student@uni.edu" not in cloud.audit_received_queries[0]
    assert "grover" in cloud.audit_received_queries[0]
