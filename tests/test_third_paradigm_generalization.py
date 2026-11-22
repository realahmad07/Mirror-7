"""Verification Suite for Mirror 7's 3rd Mathematical Paradigm:
1. Bit-Sliced Vertical HDC (`FastBitSlicedBundle`) + Abjad Phonetic Skeleton (`abjad_canonical`)
   + Unsupervised Context-Role Slot Induction (`AntiUnificationHDCModel`):
   - 20/20 unseen messy bilingual intent classification
   - 2/2 OOD gibberish rejection
   - 4/4 uppercase and 4/4 lowercase unquoted slot extraction
2. v5 Cross-Modal Contrastive Trigger Discovery + Word-Level HDC Selective Attention
   + Bottom-Up Type-Directed AST Search (`UnsupervisedCrossModalSynthesizer`):
   - 6/6 zero-shot unseen 3-way compositional AST synthesis (sandbox-verified)
3. Phonetic Frame Stripping + Anti-Unification Construction-Grammar Realizer (`induce_construction_frame`, `realize_from_frame`)
4. End-to-End `BackendService.crystal_ask()` verification across bilingual code & concept queries.
"""
from __future__ import annotations

import contextlib
import io

from mirror7_backend.native_code_generation import (
    RAW_CODE_CORPUS,
    UNSEEN_COMPOSITIONAL_TESTS,
    UnsupervisedCrossModalSynthesizer,
    extract_ast_primitives,
)
from mirror7_backend.service import BackendService
from mirror_crystal.grammar_and_sandbox import ZeroTrustSandboxJudge
from mirror_crystal.hdc_associative_engine import (
    LOWERCASE_SLOT_TESTS,
    OOD,
    TEST,
    TRAIN,
    AntiUnificationHDCModel,
    abjad_canonical,
    abjad_strip_query_frame,
    build_print_program,
    induce_construction_frame,
    realize_from_frame,
    verify_program,
)


def test_abjad_phonetic_skeleton_and_20_unseen_bilingual_intents_plus_ood():
    # Verify Abjad phonetic equivalences across Roman-Urdu & English typos/vowel-drops
    assert abjad_canonical("kya") == abjad_canonical("kia")
    assert abjad_canonical("hota") == abjad_canonical("hta")
    assert abjad_canonical("hai") == abjad_canonical("hy") == abjad_canonical("h")
    assert abjad_canonical("kare") == abjad_canonical("kre") == abjad_canonical("kry")
    assert abjad_canonical("karo") == abjad_canonical("kro")
    assert abjad_canonical("batao") == abjad_canonical("btao")
    assert abjad_canonical("print") == abjad_canonical("prnt")
    assert abjad_canonical("differnce") == abjad_canonical("difference")
    assert abjad_canonical("helo") == abjad_canonical("hello")
    assert abjad_canonical("salaam") == abjad_canonical("salam")

    model = AntiUnificationHDCModel().fit(TRAIN)

    # 1. 20/20 Unseen messy bilingual intent classification
    correct = 0
    for text, gold_label, _ in TEST:
        pred_label, score, margin = model.classify(text)
        assert pred_label == gold_label, f"Intent mismatch for {text!r}: gold={gold_label}, pred={pred_label} (score={score}, margin={margin})"
        correct += 1
    assert correct == 20

    # 2. 2/2 OOD gibberish rejection
    ood_rejected = sum(1 for gibberish in OOD if model.is_ood(gibberish))
    assert ood_rejected == 2


def test_unsupervised_context_role_uppercase_and_lowercase_slot_extraction():
    model = AntiUnificationHDCModel().fit(TRAIN)

    # 1. 4/4 Uppercase unquoted slot extraction (with typos & distractor nouns like 'script')
    uppercase_cases = [(text, slot) for text, _, slot in TEST if slot is not None]
    assert len(uppercase_cases) == 4
    for text, gold_slot in uppercase_cases:
        pred_slot = model.extract_surprise_slot(text)
        assert pred_slot == gold_slot, f"Uppercase slot miss for {text!r}: gold={gold_slot!r}, pred={pred_slot!r}"
        code = build_print_program(pred_slot)
        assert verify_program(code, f"{gold_slot}\n") is True

    # 2. 4/4 Lowercase-only unquoted slot extraction (0 capital-letter cues)
    assert len(LOWERCASE_SLOT_TESTS) == 4
    for text, gold_slot in LOWERCASE_SLOT_TESTS:
        pred_slot = model.extract_surprise_slot(text)
        assert pred_slot == gold_slot, f"Lowercase slot miss for {text!r}: gold={gold_slot!r}, pred={pred_slot!r}"
        code = build_print_program(pred_slot)
        assert verify_program(code, f"{gold_slot}\n") is True


def test_v5_cross_modal_selective_attention_and_6_unseen_3way_ast_compositions():
    synth = UnsupervisedCrossModalSynthesizer().fit(RAW_CODE_CORPUS)

    # Verify contrastive trigger discovery automatically induced AST primitive triggers
    assert "FILTER:EVEN" in synth.op_trigger_words and synth.op_trigger_words["FILTER:EVEN"]
    assert "FILTER:ODD" in synth.op_trigger_words and synth.op_trigger_words["FILTER:ODD"]
    assert "MAP:SQUARE" in synth.op_trigger_words and synth.op_trigger_words["MAP:SQUARE"]
    assert "MAP:CUBE" in synth.op_trigger_words and synth.op_trigger_words["MAP:CUBE"]
    assert "REDUCE:SUM" in synth.op_trigger_words and synth.op_trigger_words["REDUCE:SUM"]

    # 6/6 Zero-shot unseen 3-way compositional AST synthesis
    assert len(UNSEEN_COMPOSITIONAL_TESTS) == 6
    for prompt, desc, expected_stdout in UNSEEN_COMPOSITIONAL_TESTS:
        top_programs = synth.bottom_up_type_directed_search(prompt, top_k=2)
        assert top_programs, f"No AST candidates synthesized for {prompt!r}"
        best_score, best_code, best_sig = top_programs[0]
        assert verify_program(best_code, expected_stdout) is True, (
            f"Compositional AST failed verification for {prompt!r} ({desc}):\n"
            f"Synthesized ({best_sig}, score={best_score}):\n{best_code}"
        )
        extracted_ops = extract_ast_primitives(best_code)
        assert extracted_ops


def test_phonetic_frame_stripping_and_anti_unification_grammar_realizer():
    assert abjad_strip_query_frame("sql injection kya hota hai") == "sql injection"
    assert abjad_strip_query_frame("kubernetes kia hta h") == "kubernetes"
    assert abjad_strip_query_frame("machine learning asaan lafzon mein samjha do") == "machine learning"
    assert abjad_strip_query_frame("redis aur memcached mein kya farq hai") == "redis memcached"

    explain_pairs = [
        (
            {"TOPIC": "Docker", "CAT": "container runtime", "MECH": "OS-level virtualization", "GOAL": "apps isolate"},
            "Docker ek container runtime hai jo OS-level virtualization ke zariye apps isolate karta hai.",
        ),
        (
            {"TOPIC": "TLS", "CAT": "cryptographic protocol", "MECH": "asymmetric handshake", "GOAL": "traffic encrypt"},
            "TLS ek cryptographic protocol hai jo asymmetric handshake ke zariye traffic encrypt karta hai.",
        ),
    ]
    frame = induce_construction_frame(explain_pairs)
    assert "<TOPIC>" in frame and "<CAT>" in frame and "<MECH>" in frame and "<GOAL>" in frame
    realized = realize_from_frame(
        frame,
        {
            "TOPIC": "SQL Injection",
            "CAT": "security vulnerability",
            "MECH": "unsanitized query input",
            "GOAL": "database manipulate",
        },
    )
    assert realized == "SQL Injection ek security vulnerability hai jo unsanitized query input ke zariye database manipulate karta hai."


def _exec_captured_stdout(code: str) -> str:
    buf = io.StringIO()
    safe_builtins = {"print": print, "range": range, "sum": sum, "len": len, "max": max, "min": min, "list": list}
    with contextlib.redirect_stdout(buf):
        exec(compile(code, "<e2e_test>", "exec"), {"__builtins__": safe_builtins})  # noqa: S102
    return buf.getvalue()


def test_end_to_end_backend_service_crystal_ask_verification(monkeypatch):
    monkeypatch.setenv("MIRROR_SANDBOX_MODE", "process")
    svc = BackendService()

    # 1. "python k code likho jo Ahmad print kry" -> executable code printing Ahmad
    r1 = svc.crystal_ask("python k code likho jo Ahmad print kry")
    assert r1["abstained"] is False
    code1 = r1["generated_code"] or ZeroTrustSandboxJudge.extract_python_code(r1["answer"])
    assert code1
    assert _exec_captured_stdout(code1).strip() == "Ahmad"

    # 2. "Ahmad ko 3 bar print krne ka code likho" -> executable loop printing Ahmad 3 times
    r2 = svc.crystal_ask("Ahmad ko 3 bar print krne ka code likho")
    assert r2["abstained"] is False
    code2 = r2["generated_code"] or ZeroTrustSandboxJudge.extract_python_code(r2["answer"])
    assert code2
    assert _exec_captured_stdout(code2).strip().splitlines() == ["Ahmad", "Ahmad", "Ahmad"]

    # 3. "1 se 6 tak odd numbers ke squares ka sum print kro" -> executable code printing 35
    r3 = svc.crystal_ask("1 se 6 tak odd numbers ke squares ka sum print kro")
    assert r3["abstained"] is False
    code3 = r3["generated_code"] or ZeroTrustSandboxJudge.extract_python_code(r3["answer"])
    assert code3
    assert _exec_captured_stdout(code3).strip() == "35"

    # 4. Typo'd & lowercase unquoted print prompts via BackendService.crystal_ask()
    for p_typo, expected_name in (
        ("pyhton ka code btao jo usman prnt kre", "usman"),
        ("write a python script printing fatima", "fatima"),
    ):
        rt = svc.crystal_ask(p_typo)
        assert rt["abstained"] is False, f"Abstained on {p_typo!r}: {rt}"
        ct = rt["generated_code"] or ZeroTrustSandboxJudge.extract_python_code(rt["answer"])
        assert ct
        assert _exec_captured_stdout(ct).strip() == expected_name

    # 5. "sql injection kya hota hai" -> grounded SQL Injection explanation
    r4 = svc.crystal_ask("sql injection kya hota hai")
    assert r4["abstained"] is False
    assert "injection" in r4["answer"].casefold()

    # 6. "kubernetes kia hta h" -> grounded Kubernetes explanation
    r5 = svc.crystal_ask("kubernetes kia hta h")
    assert r5["abstained"] is False
    assert "kubernetes" in r5["answer"].casefold()

    # 7. "redis aur memcached mein kya farq hai" -> grounded comparison
    r6 = svc.crystal_ask("redis aur memcached mein kya farq hai")
    assert r6["abstained"] is False
    ans6_low = r6["answer"].casefold()
    assert "redis" in ans6_low and "memcached" in ans6_low
