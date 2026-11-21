"""Tests for Parameter-Free Semantic Frame VM (`SemanticStateVM`) and
Raw-DEFLATE Skeleton-NCD + Analogical Transfer Engine (`compression_analogy_engine`).
"""
from __future__ import annotations

import math
from pathlib import Path
import time

from mirror_crystal import DiskCrystalVault
from mirror_crystal.compression_analogy_engine import (
    AnalogicalTransferEngine,
    FastSkeletonNCDIndex,
    skeletonize,
    symmetric_raw_ncd,
)
from mirror_crystal.nl_uol_router import NaturalLanguageUOLRouter
from mirror_crystal.semantic_frame_vm import SemanticStateVM

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_raw_deflate_skeleton_ncd_symmetry_and_structural_isomorphism():
    s1 = skeletonize("A shirt originally costs $80 and is on sale for 25% off. What is the sale price?")
    s2 = skeletonize("A jacket originally priced at $250 is discounted by 20% off. What is the sale price?")
    s3 = skeletonize("Write a poem about autumn rain in Lahore.")

    assert s1.signature == ("MONEY", "PCT")
    assert s2.signature == ("MONEY", "PCT")
    assert s3.signature == ()

    b1, b2, b3 = s1.skeleton.encode(), s2.skeleton.encode(), s3.skeleton.encode()
    assert symmetric_raw_ncd(b1, b1) == 0.0
    assert math.isclose(symmetric_raw_ncd(b1, b2), symmetric_raw_ncd(b2, b1))
    # Structurally isomorphic word problems must have much smaller NCD than unrelated prose
    assert symmetric_raw_ncd(b1, b2) < symmetric_raw_ncd(b1, b3)


def test_fast_2stage_ncd_funnel_scales_to_1343_blueprints_in_milliseconds():
    vault_path = REPO_ROOT / "artifacts/mirror_uol_skills.dcrystal"
    vault = DiskCrystalVault(vault_path)
    try:
        bps = vault.list_blueprints(limit=2000)
    finally:
        vault.close()
    assert len(bps) >= 1000

    idx = FastSkeletonNCDIndex(coarse_top_k=24)
    idx.add_many((bp.blueprint_id, bp.prompt_pattern, {"domain": bp.domain}) for bp in bps)
    assert idx.size == len(bps)

    sample = bps[42]
    t0 = time.perf_counter()
    hits = idx.query(sample.prompt_pattern, top_k=3)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    assert hits and hits[0].doc_id == sample.blueprint_id
    assert hits[0].ncd_distance == 0.0
    # 2-stage funnel avoids O(N) compression scan and completes well under 50 ms on CPU
    assert elapsed_ms < 50.0


def test_analogical_skeleton_ncd_transfer_on_unseen_nouns_and_abstention():
    engine = AnalogicalTransferEngine()

    # Completely unseen nouns ("laptop", "headphones") & numbers
    r1 = engine.transfer_solve("A laptop priced at $1200 is on sale for 15% off. What is the sale price?")
    assert r1.matched and math.isclose(float(r1.output), 1020.0, rel_tol=1e-6)

    r2 = engine.transfer_solve("Four friends split a hotel bill of $240 equally. How much does each friend pay?")
    assert r2.matched and math.isclose(float(r2.output), 60.0, rel_tol=1e-6)

    # Unrelated prose must NOT trigger analogical transfer
    r_bad = engine.transfer_solve("Explain the philosophy of quantum mechanics.")
    assert r_bad.matched is False


def test_semantic_state_vm_multistep_frames_and_honest_abstention():
    vm = SemanticStateVM()

    # 3-step sequential inventory chain (INIT -> SUB -> ADD -> SUB)
    r_inv = vm.solve(
        "A reservoir holds 1000 liters. 300 liters are drained, then 150 liters are poured in, and 50 liters are removed. How much water is left?"
    )
    assert r_inv.solved and math.isclose(float(r_inv.output), 800.0)
    assert len(r_inv.steps) == 4

    # Buy + change
    r_ch = vm.solve("A bakery sells muffins for $3 each. If Omar buys 6 muffins and pays with $25, how much change does he get?")
    assert r_ch.solved and math.isclose(float(r_ch.output), 7.0)

    # Relative scaling + ledger sum
    r_led = vm.solve("Nadia solves 14 puzzles on Friday and twice as many on Saturday. How many puzzles did she solve in total?")
    assert r_led.solved and math.isclose(float(r_led.output), 42.0)

    # Equal partition
    r_part = vm.solve("If 5 shelves hold 85 books in total, how many books are on 1 shelf?")
    assert r_part.solved and math.isclose(float(r_part.output), 17.0)

    # Non-quantitative / ambiguous -> MUST abstain honestly
    assert vm.solve("Write a song about Karachi.").solved is False
    assert vm.solve("Who won the 2099 Olympics?").solved is False


def test_nl_router_integrates_semantic_vm_and_analogical_transfer():
    router = NaturalLanguageUOLRouter()
    r = router.route("A store sells pens for $2 each. If Ali buys 9 pens and pays with $25, how much change does he get?")
    assert r.matched and math.isclose(float(r.output), 7.0)
    assert r.rule_ids == ("semantic_vm.buy_and_change",)
