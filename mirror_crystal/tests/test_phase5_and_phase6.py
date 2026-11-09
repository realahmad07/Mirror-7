from __future__ import annotations

import math
from pathlib import Path
import shutil
import struct
import subprocess
import zlib

import pytest

from mirror_crystal import (
    CloudCrystalNode,
    ContextBitTable,
    CrystalRule,
    DiskCrystalVault,
    FactFidelityGuard,
    MirrorCrystalEngine,
    PNGSecurityError,
    PackedSimdBitKernel,
    PackedTernaryProjection,
    PerceptualVQLens,
    PocketCrystalClient,
    SensoryCodec,
    ShardedCrystalRouter,
    SignedRuleCapsule,
    WasmPocketCompiler,
)


def test_p5_1_symbolic_cas_differentiation_polynomial_and_chain_rule() -> None:
    eng = MirrorCrystalEngine()
    r_poly = eng.respond("Differentiate x^3 + 4*x^2 - 7*x + 5")
    assert r_poly.mode == "symbolic_cas_exact"
    assert r_poly.epistemic_tier == "verified_exact"
    assert r_poly.sandbox_verified is True
    assert "3*x^2 + 8*x - 7" in r_poly.answer

    r_trig = eng.respond("Differentiate sin(3*x)")
    assert r_trig.mode == "symbolic_cas_exact"
    assert r_trig.sandbox_verified is True
    assert "cos" in r_trig.answer and "3" in r_trig.answer


def test_p5_2_symbolic_cas_indefinite_and_definite_integration(tmp_path: Path) -> None:
    vault = DiskCrystalVault(tmp_path / "cas_test.dcrystal")
    r_def = vault.respond("Integrate 3*x^2 + 2*x from 0 to 4")
    assert r_def.mode == "symbolic_cas_exact"
    assert r_def.sandbox_verified is True
    assert r_def.answer.endswith("= 80")

    r_indef = vault.respond("Integrate 6*x^2 - 4*x + 1")
    assert r_indef.mode == "symbolic_cas_exact"
    assert r_indef.sandbox_verified is True
    assert "2*x^3 - 2*x^2 + x + C" in r_indef.answer
    vault.close()


def test_p5_3_symbolic_cas_linear_system_quadratic_and_unit_conversions() -> None:
    eng = MirrorCrystalEngine()
    r_sys = eng.respond("Solve 3*x + 2*y = 12, x - y = 1")
    assert r_sys.mode == "symbolic_cas_exact"
    assert r_sys.sandbox_verified is True
    assert "x = 14/5" in r_sys.answer and "y = 9/5" in r_sys.answer

    r_quad = eng.respond("Solve x^2 - 5*x + 6 = 0")
    assert r_quad.mode == "symbolic_cas_exact"
    assert r_quad.sandbox_verified is True
    assert "2, 3" in r_quad.answer

    r_speed = eng.respond("Convert 72 km/h to m/s")
    assert r_speed.mode == "symbolic_cas_exact"
    assert "= 20 m/s" in r_speed.answer

    r_temp = eng.respond("Convert 100 C to K")
    assert r_temp.mode == "symbolic_cas_exact"
    assert "373.15" in r_temp.answer


def test_p6_b6_3_cas_complexity_explosion_and_dimensional_mismatch_guards() -> None:
    eng = MirrorCrystalEngine()
    r_exp = eng.respond("Differentiate (x + 1)^500")
    assert r_exp.mode == "cas_complexity_guarded"
    assert r_exp.latency_ms < 25.0

    r_dim = eng.respond("Convert 5 kg to km")
    assert r_dim.mode == "cas_dimensional_mismatch"
    assert "Dimensional mismatch" in r_dim.answer


def test_p5_4_packed_ternary_158bit_projection_zero_float_multiplies() -> None:
    matrix = [
        [1, -1, 0, 1, 0, -1, 1, 0],
        [-1, 0, 1, 1, -1, 0, 0, 1],
        [0, 1, 1, -1, 1, 1, -1, -1],
    ]
    proj = PackedTernaryProjection.from_ternary_matrix(matrix)
    assert proj.storage_bytes() == 3 * 2  # 8 ternary weights packed into just 2 bytes per row!
    vec = [10, -4, 7, 3, -9, 2, 11, -5]
    got = proj.project_int(vec)
    expected = [sum(w * x for w, x in zip(row, vec)) for row in matrix]
    assert got == expected


def test_p5_5_bicameral_grounded_compose_and_fact_fidelity_guard() -> None:
    eng = MirrorCrystalEngine()
    fact = "Grover search achieves O(sqrt(N)) query complexity instead of 1000000 linear checks and does not require sorted keys."
    eng.ingest_pair("Explain Grover search query complexity", fact)

    res = eng.respond("Draft a polite client email explaining Grover search query complexity")
    assert res.mode == "bicameral_grounded_compose"
    assert "Subject:" in res.answer
    assert "O(sqrt(N))" in res.answer
    assert "1000000" in res.answer
    assert "not" in res.answer.lower()

    # FactFidelityGuard must reject rewrites that drop negations, alter O(...), or hallucinate numbers
    assert FactFidelityGuard.verify_preservation(fact, "Grover achieves O(N) instead of 1000000 and does not sort.") is False
    assert FactFidelityGuard.verify_preservation(fact, "Grover achieves O(sqrt(N)) in 999 steps and does not sort.") is False
    assert FactFidelityGuard.verify_preservation(fact, "Grover achieves O(sqrt(N)) instead of 1000000 linear checks.") is False


def test_p5_6_bicameral_creative_synthesis_and_bleed_guard() -> None:
    eng = MirrorCrystalEngine()
    eng.ingest_pair("What is 2 + 2?", "4")
    rules_before = len(eng.rule_index.rules)

    r_story = eng.respond("Write a short story about a lighthouse on Mars")
    assert r_story.mode == "bicameral_creative_synthesis"
    assert r_story.epistemic_tier == "creative_synthesis"
    assert r_story.sandbox_verified is False
    assert "lighthouse" in r_story.answer.lower() and "mars" in r_story.answer.lower()
    # CreativeFactualBleedGuard: vault rule count must NEVER grow from creative outputs
    assert len(eng.rule_index.rules) == rules_before

    # Factual unknown questions must STILL abstain honestly (zero hallucination!)
    r_unknown = eng.respond("What is the orbital period of planet Zorgon999?")
    assert r_unknown.mode == "unmatched_honest_abstain"
    assert r_unknown.epistemic_tier == "honest_abstain"


def test_p5_7_and_p6_b6_4_active_curiosity_clarifier_and_loop_guard() -> None:
    eng = MirrorCrystalEngine()
    eng.ingest_pair(
        "How to tune database storage engine postgresql vacuum?",
        "Use PostgreSQL autovacuum scale factor and B-tree fillfactor tuning.",
    )
    eng.ingest_pair(
        "How to tune database storage engine sqlite pragma?",
        "Enable SQLite journal_mode=WAL and mmap_size pragma for concurrent reads.",
    )

    # Turn 1: Ambiguous short query matching both distinct storage branches equally
    r1 = eng.respond("tune database storage engine", session_id="cur_s1")
    assert r1.mode == "clarification_needed"
    assert r1.epistemic_tier == "diagnostic_clarification"
    assert "postgresql" in r1.answer.lower() or "sqlite" in r1.answer.lower()

    # Turn 2: User replies with single clarifying keyword -> locks onto exact SQLite rule (no loop!)
    r2 = eng.respond("sqlite", session_id="cur_s1")
    assert r2.mode == "direct_crystal_lock"
    assert "WAL" in r2.answer


def test_p5_8_and_p6_b6_6_document_and_python_auto_crystallizer_with_boilerplate_guard() -> None:
    raw_doc = """# Table of Contents
1. Intro
2. License

# License
Apache 2.0 boilerplate text that must be skipped by DocBoilerplateAndSafetyGuard.

# Fenwick Prefix Tree Architecture
A Fenwick tree maintains prefix sums in O(log N) time using least-significant set bit jumps across an array.

**Merkle Root**: A cryptographic hash summarizing every leaf transaction in a binary hash tree.
thermal_runaway causes electrolyte_venting.

```python
def cube_plus_two(x: int) -> int:
    \"\"\"Compute x cubed plus two.\"\"\"
    return x ** 3 + 2

assert cube_plus_two(3) == 29

def unsafe_backdoor():
    import os
    return os.getcwd()
```
"""
    eng = MirrorCrystalEngine()
    st = eng.ingest_document(raw_doc, title="SystemsManual")
    assert st["boilerplate_skipped"] >= 2
    assert st["sections_crystallized"] >= 1
    assert st["definitions_crystallized"] >= 1
    assert st["code_blocks_verified"] == 1

    r_sec = eng.respond("Explain Fenwick Prefix Tree Architecture")
    assert "O(log N)" in r_sec.answer

    r_def = eng.respond("What is Merkle Root?")
    assert "cryptographic hash" in r_def.answer.lower()

    r_fn = eng.respond("Write Python function cube_plus_two: Compute x cubed plus two.")
    assert "x ** 3 + 2" in r_fn.answer
    # Code execution is evidence within this bounded check, not a formal proof.
    assert r_fn.epistemic_tier == "high_confidence"


def test_p5_9_pure_stdlib_png_and_wav_sensory_roundtrip_invariance(tmp_path: Path) -> None:
    # Build 12x12 corner fiducial grid + rotated 90 deg brighter version -> real .png bytes!
    base_grid = [[220.0 if (r < 4 and c < 4) else 30.0 for c in range(12)] for r in range(12)]
    rot90_bright = [
        [min(255.0, base_grid[11 - c][r] * 1.1 + 15.0) for c in range(12)]
        for r in range(12)
    ]
    png_a = SensoryCodec.encode_grid_to_png_bytes(base_grid)
    png_b = SensoryCodec.encode_grid_to_png_bytes(rot90_bright)

    vault = DiskCrystalVault(tmp_path / "sensory.dcrystal")
    vault.ingest_pair(
        PerceptualVQLens.wrap_multimodal_prompt("Identify PCB fiducial PNG", png_bytes=png_a),
        "Fiducial Marker Alpha-9 verified.",
    )
    res_img = vault.respond(PerceptualVQLens.wrap_multimodal_prompt("Identify PCB fiducial PNG", png_bytes=png_b))
    assert "Alpha-9" in res_img.answer

    # Real 16-bit PCM .wav roundtrip under amplitude gain change (1.0x vs 0.35x)
    sr = 8000
    wave_loud = [0.85 * math.sin(2.0 * math.pi * 440.0 * i / sr) for i in range(400)]
    wave_quiet = [0.30 * math.sin(2.0 * math.pi * 440.0 * i / sr) for i in range(400)]
    wav_a = SensoryCodec.encode_samples_to_wav_bytes(wave_loud, sample_rate=sr)
    wav_b = SensoryCodec.encode_samples_to_wav_bytes(wave_quiet, sample_rate=sr)

    vault.ingest_pair(
        PerceptualVQLens.wrap_multimodal_prompt("Classify acoustic tone WAV", wav_bytes=wav_a),
        "Calibration 440Hz A4 reference tone.",
    )
    res_wav = vault.respond(PerceptualVQLens.wrap_multimodal_prompt("Classify acoustic tone WAV", wav_bytes=wav_b))
    assert "440Hz" in res_wav.answer
    vault.close()


def test_p6_b6_2_png_decompression_bomb_and_oversized_header_guard() -> None:
    # 1. Oversized 10000x10000 IHDR header must be blocked BEFORE buffer allocation
    ihdr_huge = struct.pack(">IIBBBBB", 10000, 10000, 8, 0, 0, 0, 0)
    fake_png = (
        b"\x89PNG\r\n\x1a\n"
        + SensoryCodec._png_chunk(b"IHDR", ihdr_huge)
        + SensoryCodec._png_chunk(b"IDAT", zlib.compress(b"\x00" * 64))
        + SensoryCodec._png_chunk(b"IEND", b"")
    )
    with pytest.raises(PNGSecurityError):
        SensoryCodec.decode_png_bytes_to_luma_grid(fake_png)

    # 2. Normal 16x16 IHDR concealing a 20 MB zlib decompression bomb inside IDAT
    ihdr_ok = struct.pack(">IIBBBBB", 16, 16, 8, 0, 0, 0, 0)
    bomb_idat = zlib.compress(b"\x00" * (20 * 1024 * 1024), level=9)
    bomb_png = (
        b"\x89PNG\r\n\x1a\n"
        + SensoryCodec._png_chunk(b"IHDR", ihdr_ok)
        + SensoryCodec._png_chunk(b"IDAT", bomb_idat)
        + SensoryCodec._png_chunk(b"IEND", b"")
    )
    with pytest.raises(PNGSecurityError, match="Bomb|oversized"):
        SensoryCodec.decode_png_bytes_to_luma_grid(bomb_png)


def test_p5_10_and_p6_b6_5_sharded_crystal_router_fanout_and_cold_shard_hint(tmp_path: Path) -> None:
    router = ShardedCrystalRouter(tmp_path / "shards", max_open_shards=2)
    s1, _ = router.ingest_pair(
        "Implement binary search in Python function",
        "def bin_search(a, x): ...",
    )
    s2, _ = router.ingest_pair(
        "Explain Grover quantum amplitude amplification speedup",
        "Grover achieves O(sqrt(N)) quantum oracle queries.",
    )
    s3, _ = router.ingest_pair(
        "Configure SQLite WAL database index latency",
        "Set PRAGMA journal_mode=WAL for low-latency concurrent reads.",
    )
    assert s1 == "code"
    assert s2 == "math_quantum"
    assert s3 == "systems"

    router.register_cold_shard(
        "astrophysics_archive",
        "pulsars magnetars neutron star chandrasekhar supernova redshift",
        remote_uri="gdrive://Mirror7_Ultra_Backup/astrophysics.dcrystal",
    )
    router.close()

    # Reopen from disk manifest (0 rules loaded in RAM) and test cross-domain routing + cold shard hint
    reopened = ShardedCrystalRouter(tmp_path / "shards", max_open_shards=2)
    r_cross = reopened.respond("Python function for Grover quantum amplitude speedup")
    assert "O(sqrt(N))" in r_cross.answer or "bin_search" in r_cross.answer

    r_cold = reopened.respond("Explain magnetars neutron star chandrasekhar limit")
    assert any("astrophysics_archive" in t for t in r_cold.reasoning_trace)
    reopened.close()


def test_p5_11_packed_simd_bit_kernel_and_wasm_pocket_binary_v8_execution(tmp_path: Path) -> None:
    bt = ContextBitTable()
    kernel = PackedSimdBitKernel()
    sample_texts = [f"quantum circuit gate layer {i} entanglement qubit {i % 7}" for i in range(300)]
    for t in sample_texts:
        bt.observe_text(t)
    for idx, t in enumerate(sample_texts):
        sig, _ = bt.encode_text_signature(t)
        kernel.add(idx, sig)

    q_sig, _ = bt.encode_text_signature("quantum circuit gate layer 42 entanglement")
    top = kernel.top_k_asymmetric(q_sig, top_k=3)
    assert top[0][1] == 42
    inter = (q_sig & kernel.sigs[42]).bit_count()
    union = q_sig.bit_count() + kernel.sigs[42].bit_count() - inter
    expected_sc = 0.72 * (inter / q_sig.bit_count()) + 0.28 * (inter / union)
    assert abs(top[0][0] - expected_sc) < 1e-9

    wasm_path = WasmPocketCompiler.write_pocket_wasm(tmp_path / "pocket_popcnt.wasm")
    meta = WasmPocketCompiler.validate_wasm_binary(wasm_path.read_bytes())
    assert meta["valid"] is True
    assert meta["byte_size"] < 300

    node_bin = shutil.which("node")
    if node_bin:
        sig_a = kernel.sigs[10]
        sig_b = kernel.sigs[42]
        py_and = (sig_a & sig_b).bit_count()
        py_xor = (sig_a ^ sig_b).bit_count()
        a_hex = sig_a.to_bytes(256, "little").hex()
        b_hex = sig_b.to_bytes(256, "little").hex()
        js = (
            f"const fs=require('fs');"
            f"const inst=new WebAssembly.Instance(new WebAssembly.Module(fs.readFileSync({str(wasm_path)!r})));"
            f"const m=new Uint8Array(inst.exports.memory.buffer);"
            f"m.set(Buffer.from('{a_hex}','hex'),0);"
            f"m.set(Buffer.from('{b_hex}','hex'),256);"
            f"console.log(inst.exports.popcnt2048_and(0,256)+','+inst.exports.popcnt2048_xor(0,256));"
        )
        proc = subprocess.run([node_bin, "-e", js], capture_output=True, text=True, check=True, timeout=10)
        wasm_and, wasm_xor = [int(x) for x in proc.stdout.strip().split(",")]
        assert wasm_and == py_and
        assert wasm_xor == py_xor


def test_p5_12_proof_carrying_rule_capsules_block_buggy_and_unsafe_signed_rules() -> None:
    secret = b"shared-test-hmac-key"
    pocket = PocketCrystalClient(verify_key=secret, max_ram_rules=16)

    # 1. Validly HMAC-signed capsule with WRONG math code failing its own proof_test_code -> BLOCKED!
    bad_rule = CrystalRule(
        rule_id=0,
        signature=12345,
        salient_tokens=("triple",),
        abstract_prompt="Write triple(x)",
        abstract_response="def triple(x):\n    return x * 999",
        abstract_tool="assert triple(4) == 12",
        num_slots=0,
        derived_expr="",
        pass_count=5,
        fail_count=0,
        domain="code",
        trust_tier=1,
    )
    bad_cap = SignedRuleCapsule.sign(bad_rule, secret, proof_test_code="assert triple(4) == 12")
    assert pocket.ingest_signed_capsule(bad_cap.to_wire_dict()) is False
    assert pocket.stats["failed_proof_capsules_blocked"] == 1

    # 2. Validly HMAC-signed capsule smuggling `import os` -> BLOCKED!
    evil_rule = CrystalRule(
        rule_id=1,
        signature=67890,
        salient_tokens=("pwn",),
        abstract_prompt="Run pwn()",
        abstract_response="def pwn():\n    import os\n    return os.getcwd()",
        abstract_tool="",
        num_slots=0,
        derived_expr="",
        pass_count=5,
        fail_count=0,
        domain="code",
        trust_tier=1,
    )
    evil_cap = SignedRuleCapsule.sign(evil_rule, secret)
    assert pocket.ingest_signed_capsule(evil_cap.to_wire_dict()) is False
    assert pocket.stats["failed_proof_capsules_blocked"] == 2

    # 3. Validly signed + mathematically proved capsule -> ACCEPTED & runs offline!
    good_rule = CrystalRule(
        rule_id=2,
        signature=54321,
        salient_tokens=("triple", "number"),
        abstract_prompt="Write Python function triple(x)",
        abstract_response="def triple(x):\n    return x * 3",
        abstract_tool="assert triple(4) == 12",
        num_slots=0,
        derived_expr="",
        pass_count=6,
        fail_count=0,
        domain="code",
        trust_tier=1,
    )
    good_cap = SignedRuleCapsule.sign(good_rule, secret, proof_test_code="assert triple(4) == 12")
    assert pocket.ingest_signed_capsule(good_cap.to_wire_dict()) is True
    pocket.online = False
    r_off = pocket.ask("Write Python function triple(x)")
    assert "x * 3" in r_off.answer

    # End-to-end Cloud -> Pocket proof-carrying teleport
    cloud_eng = MirrorCrystalEngine()
    cloud_eng.ingest_pair(
        "Implement square_plus_one(n)",
        "def square_plus_one(n):\n    return n * n + 1",
        tool="assert square_plus_one(5) == 26",
        test_code="assert square_plus_one(5) == 26",
    )
    node = CloudCrystalNode(cloud_eng, signing_key=secret)
    pocket.cloud = node
    pocket.online = True
    r_tel = pocket.ask("Implement square_plus_one(n)")
    assert "n * n + 1" in r_tel.answer


def test_p6_cross_lingual_symbolic_cas_urdu_chinese_spanish() -> None:
    eng = MirrorCrystalEngine()
    r_zh = eng.respond("求导 x^3 + 5*x")
    assert r_zh.mode == "symbolic_cas_exact"
    assert "3*x^2 + 5" in r_zh.answer

    r_ur = eng.respond("x^3 + 5*x کا مشتق")
    assert r_ur.mode == "symbolic_cas_exact"
    assert "3*x^2 + 5" in r_ur.answer

    r_es = eng.respond("resuelve 3*x + 2 = 14")
    assert r_es.mode == "symbolic_cas_exact"
    assert "x = 4" in r_es.answer
