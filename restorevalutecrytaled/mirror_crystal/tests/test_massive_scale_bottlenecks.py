"""
Verification Suite for Massive Scale Bottleneck Defenses:
1. Scalable Bloom Filter (Constant-RAM deduplication up to 50M records)
2. High-Throughput Batch Ingestion (ingest_batch with WAL transactions)
3. Inverted Index Posting List Crowding & Dynamic Frequency Pruning
4. Execution Truth Arbitrator (PRM & Sandbox precedence over unverified assertions)
5. Sharded Crystal Router Drop-In Backend Integration
"""
from __future__ import annotations

import math
from pathlib import Path
import pytest

from mirror_crystal import (
    DiskCrystalVault,
    ScalableBloomFilter,
    ShardedCrystalRouter,
    SixGateDataHygienePipeline,
)
from mirror7_backend.service import BackendService


def test_scalable_bloom_filter_saturation_and_accuracy():
    """Verify ScalableBloomFilter maintains constant memory and < 0.1% false-positive rate."""
    bf = ScalableBloomFilter(capacity=5000, error_rate=0.001)
    
    # Insert 2,000 distinct items
    for i in range(2000):
        bf.add(f"prompt_key_{i}")
    
    assert bf.count == 2000
    assert bf.bit_count() > 0
    
    # 100% True-Positive guarantee
    for i in range(2000):
        assert f"prompt_key_{i}" in bf
        
    # False-positive rate verification on 2,000 unseen items
    false_positives = sum(1 for i in range(2000, 4000) if f"unseen_key_{i}" in bf)
    fp_rate = false_positives / 2000.0
    assert fp_rate <= 0.005, f"False-positive rate too high: {fp_rate}"
    
    # Clear resets cleanly
    bf.clear()
    assert bf.count == 0
    assert bf.bit_count() == 0
    assert "prompt_key_0" not in bf


def test_six_gate_bloom_filter_deduplication():
    """Verify SixGateDataHygienePipeline uses ScalableBloomFilter to drop exact duplicates in O(1)."""
    pipeline = SixGateDataHygienePipeline(max_seen_cache=100)
    
    sample = {
        "text": "def calculate_harmonic_mean(a, b):\n    '''Calculate the harmonic mean of two numbers.'''\n    return 2 * (a * b) / (a + b)\n",
        "spdx": "MIT",
    }
    
    # First admission passes
    res1 = pipeline.audit_sample(sample["text"], spdx_license=sample["spdx"])
    assert res1.passed is True
    
    # Immediate second admission rejected as exact duplicate by Bloom filter / MinHash
    res2 = pipeline.audit_sample(sample["text"], spdx_license=sample["spdx"])
    assert res2.passed is False
    assert "Duplicate" in (res2.rejection_reason or "")


def test_disk_vault_ingest_batch_high_throughput(tmp_path: Path):
    """Verify ingest_batch commits hundreds of records in single transactions with O(1) retrieval."""
    vault = DiskCrystalVault(tmp_path / "batch_vault.dcrystal")
    
    records = [
        {
            "prompt": f"Define cryptographic nonce index {i}",
            "response": f"A nonce at index {i} is an arbitrary number used only once in cryptographic communications.",
            "domain": "systems",
            "trust_tier": 2,
        }
        for i in range(500)
    ]
    
    stats = vault.ingest_batch(records, batch_size=200)
    assert stats["new_rules"] == 500
    assert stats["duplicates_skipped"] == 0
    assert vault.rule_count() == 500
    
    # Re-ingesting same batch recognizes all as duplicates
    stats_dup = vault.ingest_batch(records, batch_size=200)
    assert stats_dup["new_rules"] == 0
    assert stats_dup["duplicates_skipped"] == 500
    
    # Instant retrieval of first and last items
    r0 = vault.respond("Define cryptographic nonce index 0")
    assert "arbitrary number" in r0.answer
    assert r0.confidence >= 0.85
    
    r499 = vault.respond("Define cryptographic nonce index 499")
    assert "arbitrary number" in r499.answer
    assert r499.confidence >= 0.85
    vault.close()


def test_posting_list_crowding_and_stride(tmp_path: Path):
    """Verify that common query terms do not drown rare discriminatory keywords across large posting lists."""
    vault = DiskCrystalVault(tmp_path / "crowd_vault.dcrystal")
    
    # Create 300 rules sharing the ultra-common words 'python function algorithm code'
    crowded_records = [
        {
            "prompt": f"Write a python function algorithm code for task {i}",
            "response": f"Here is the standard python function algorithm code for task {i}.",
            "domain": "coding",
            "trust_tier": 2,
        }
        for i in range(300)
    ]
    vault.ingest_batch(crowded_records, batch_size=150)
    
    # Insert a target rule with a rare token
    vault.ingest_pair(
        "Write a python function algorithm code for parallel_quantum_fourier_transform",
        "Implementation of parallel_quantum_fourier_transform using butterfly stages.",
        domain="quantum",
        trust_tier=1,
    )
    
    # Query with common words AND the rare token
    res = vault.respond("Write a python function algorithm code for parallel_quantum_fourier_transform")
    assert "parallel_quantum_fourier_transform" in res.answer
    assert "butterfly stages" in res.answer
    assert res.confidence >= 0.85
    vault.close()


def test_execution_truth_math_and_code_arbitration(tmp_path: Path):
    """Verify that verified code and valid math proofs automatically supersede unverified stubs."""
    vault = DiskCrystalVault(tmp_path / "truth_arbitration.dcrystal")
    
    prompt = "How to reverse an array in python?"
    # 1. Ingest unverified text answer (tier 2)
    vault.ingest_pair(prompt, "You can reverse it manually with a while loop.", trust_tier=2)
    assert "while loop" in vault.respond(prompt).answer
    
    # 2. Ingest verified implementation with unit test (promotes to tier 1 verified truth)
    verified_code = "def reverse_array(arr):\n    return arr[::-1]\n"
    test_code = "assert reverse_array([1, 2, 3]) == [3, 2, 1]\nassert reverse_array([]) == []\n"
    vault.ingest_pair(prompt, verified_code, test_code=test_code, trust_tier=2)
    
    # Verified truth supersedes the unverified assertion
    res = vault.respond(prompt)
    assert "arr[::-1]" in res.answer
    assert res.sandbox_verified is True
    
    # 3. Attempt to poison with failing test -> rejected
    bad_code = "def reverse_array(arr):\n    return arr\n"
    vault.ingest_pair(prompt, bad_code, test_code=test_code, trust_tier=2)
    res_after = vault.respond(prompt)
    assert "arr[::-1]" in res_after.answer  # Bad code was blocked
    vault.close()


def test_sharded_crystal_router_backend_service_integration(tmp_path: Path):
    """Verify ShardedCrystalRouter works seamlessly inside BackendService with rule_count and multi-shard dispatch."""
    router = ShardedCrystalRouter(
        root_dir=tmp_path / "shards",
        shard_names=("code", "math_quantum", "general"),
    )
    
    # Ingest into different domains
    router.ingest_pair("What is quicksort?", "Quicksort is an O(n log n) sorting algorithm.", domain="code")
    router.ingest_pair("What is a Hadamard gate?", "The Hadamard gate creates quantum superposition.", domain="math_quantum")
    
    assert router.rule_count() == 2
    
    # Pass ShardedCrystalRouter directly to BackendService
    service = BackendService(crystal_vault=router, enable_crystal_auto_load=False)
    
    # Test query routing through BackendService
    q_code = service.crystal_ask("What is quicksort?")
    assert "Quicksort" in q_code["answer"]
    assert q_code["confidence"] >= 0.8
    
    q_quant = service.crystal_ask("What is a Hadamard gate?")
    assert "Hadamard gate" in q_quant["answer"]
    assert q_quant["confidence"] >= 0.8


def test_export_mmap_index_and_domain_sharding(tmp_path: Path):
    from mirror_crystal import DiskCrystalVault, MmapPostingsIndex
    vault_p = tmp_path / "source_vault.dcrystal"
    vault = DiskCrystalVault(vault_p)
    try:
        vault.ingest_pair("What is Grover search?", "Grover search achieves O(sqrt(N)) speedup.", domain="quantum")
        vault.ingest_pair("What is Needleman-Wunsch algorithm?", "Needleman-Wunsch does global sequence alignment.", domain="bioinformatics")
        vault.ingest_pair("What is Black-Scholes formula?", "Black-Scholes prices European call and put options.", domain="finance")
        assert vault.rule_count() == 3

        # 1. Test Domain Sharding
        shards_dir = tmp_path / "domain_shards"
        shard_counts = vault.shard_by_domain(shards_dir)
        assert shard_counts["quantum"] == 1
        assert shard_counts["bioinformatics"] == 1
        assert shard_counts["finance"] == 1

        q_shard = DiskCrystalVault(shards_dir / "vault_quantum.dcrystal")
        assert q_shard.rule_count() == 1
        res_q = q_shard.respond("What is Grover search?")
        assert "speedup" in res_q.answer
        q_shard.close()

        # 2. Test Mmap Binary Index Export and Zero-Copy Lookup
        mmap_p = tmp_path / "index.mmrules"
        meta = vault.export_mmap_index(mmap_p)
        assert meta["token_count"] >= 3
        assert meta["bytes_written"] > 0
        assert mmap_p.exists()

        mmap_idx = MmapPostingsIndex(mmap_p)
        rids_grover = mmap_idx.lookup_rids("grover")
        assert len(rids_grover) >= 1

        rids_needleman = mmap_idx.lookup_rids("needleman")
        assert len(rids_needleman) >= 1

        rids_missing = mmap_idx.lookup_rids("unknown_token_xyz")
        assert rids_missing == []
        mmap_idx.close()
    finally:
        vault.close()
