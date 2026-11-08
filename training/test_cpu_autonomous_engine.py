from __future__ import annotations

from pathlib import Path
import torch

from training.cpu_autonomous_engine import (
    ASTSafetyAuditor,
    AutonomousWakeSleepTrainer,
    ProductKeyMemoryBank,
    TernaryBitLinear,
    VerifiedSkillForge,
    benchmark_cpu_scan_vs_sequential_gru,
    parallel_associative_linear_scan,
)


def test_ternary_bitlinear_ste_and_cpu_compression():
    torch.manual_seed(7)
    layer = TernaryBitLinear(256, 256, bias=True)
    x = torch.randn(4, 256, requires_grad=True)
    out = layer(x)
    loss = out.pow(2).mean()
    loss.backward()

    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert layer.weight.grad is not None and torch.isfinite(layer.weight.grad).all()

    layer.eval()
    with torch.no_grad():
        y_before = layer(x)
        stats = layer.freeze_for_cpu_inference()
        y_after = layer(x)

    assert stats["compression_vs_fp32"] >= 15.0
    assert set(layer.frozen_ternary_int8.unique().tolist()).issubset({-1, 0, 1})
    assert torch.allclose(y_before, y_after, atol=1e-5)


def test_product_key_memory_sublinear_lookup_and_zero_backprop_recall():
    torch.manual_seed(11)
    pkm = ProductKeyMemoryBank(state_dim=128, codebook_size=128, top_k=8)
    assert pkm.total_slots == 16384

    q = torch.randn(2, 128)
    out = pkm(q)
    assert out.shape == (2, 128)

    # Instant 0-backprop episodic write & recall
    s_vec = torch.randn(1, 128)
    slot = pkm.write_verified_episode(
        state_vec=s_vec,
        prompt_signature="Implement Dijkstra shortest path",
        response_text="def dijkstra(adj, src): ...",
        reward=1.0,
    )
    assert 0 <= slot < 16384

    noisy_query = s_vec + 0.02 * torch.randn_like(s_vec)
    recalled = pkm.recall_verified_episode(noisy_query, min_cosine=0.90)
    assert recalled is not None
    assert "dijkstra" in recalled.response_text


def test_parallel_associative_scan_matches_exact_sequential_recurrence():
    torch.manual_seed(3)
    bsz, seq_len, dim = 2, 129, 64
    gates = torch.sigmoid(torch.randn(bsz, seq_len, dim))
    vals = torch.tanh(torch.randn(bsz, seq_len, dim))

    # Ground-truth sequential loop
    seq_out = []
    h = torch.zeros(bsz, dim)
    for t in range(seq_len):
        h = gates[:, t] * h + vals[:, t]
        seq_out.append(h)
    expected = torch.stack(seq_out, dim=1)

    # Parallel associative prefix scan in ceil(log2(T)) steps
    actual = parallel_associative_linear_scan(gates, vals)
    assert torch.allclose(expected, actual, atol=1e-5)


def test_verified_skill_forge_blocks_unsafe_code_and_self_upgrades(tmp_path: Path):
    reg_file = tmp_path / "verified_skills.json"
    forge = VerifiedSkillForge(registry_path=reg_file)

    # 1. Unsafe OS / dunder / open code MUST be rejected
    for unsafe_snippet in [
        "import os\ndef bad(): return os.getcwd()",
        "def bad(): return open('secret.txt').read()",
        "def bad(x): return x.__class__.__bases__",
        "def bad(s): return eval(s)",
    ]:
        ok, _ = ASTSafetyAuditor.audit(unsafe_snippet)
        assert not ok
        res = forge.synthesize_and_recruit("bad", unsafe_snippet, "assert True")
        assert not res["recruited"]

    # 2. Recruit an initial O(n^2) LIS implementation (v1)
    slow_o_n2_lis = (
        "def lis_length(nums):\n"
        "  if not nums: return 0\n"
        "  dp = [1] * len(nums)\n"
        "  for i in range(len(nums)):\n"
        "    for j in range(i):\n"
        "      if nums[j] < nums[i] and dp[j] + 1 > dp[i]:\n"
        "        dp[i] = dp[j] + 1\n"
        "  return max(dp)\n"
    )
    test_suite = (
        "assert lis_length([10, 9, 2, 5, 3, 7, 101, 18]) == 4\n"
        "assert lis_length([0, 1, 0, 3, 2, 3]) == 4\n"
        "assert lis_length([]) == 0\n"
    )
    bench_expr = "assert lis_length(list(range(220, 0, -1)) + list(range(120))) == 120"

    r1 = forge.synthesize_and_recruit(
        "lis_length", slow_o_n2_lis, test_suite, keywords=["lis"], benchmark_expr=bench_expr
    )
    assert r1["recruited"] is True
    assert r1["version"] == 1

    # 3. Synthesize O(n log n) patience-sorting LIS -> must self-upgrade to v2 due to faster CPU latency!
    fast_nlogn_lis = (
        "import bisect\n"
        "def lis_length(nums):\n"
        "  tails = []\n"
        "  for x in nums:\n"
        "    idx = bisect.bisect_left(tails, x)\n"
        "    if idx == len(tails): tails.append(x)\n"
        "    else: tails[idx] = x\n"
        "  return len(tails)\n"
    )
    r2 = forge.synthesize_and_recruit(
        "lis_length", fast_nlogn_lis, test_suite, keywords=["lis"], benchmark_expr=bench_expr
    )
    assert r2["recruited"] is True
    assert r2["upgraded"] is True
    assert r2["version"] == 2
    assert r2["benchmark_us"] < r1["benchmark_us"]

    # 4. Re-submitting the slow O(n^2) candidate MUST be rejected so skills never regress
    r3 = forge.synthesize_and_recruit(
        "lis_length", slow_o_n2_lis, test_suite, keywords=["lis"], benchmark_expr=bench_expr
    )
    assert r3["recruited"] is False

    # 5. Recruited skill is live and callable inside sandboxed execution
    out = forge.execute_with_recruited_skills("python: print(lis_length([3, 1, 2, 4, 5]))")
    assert out == "4"
    assert reg_file.is_file()


def test_autonomous_wake_sleep_trainer_end_to_end_on_cpu(tmp_path: Path):
    trainer = AutonomousWakeSleepTrainer(
        state_dim=128, lora_rank=4, registry_path=tmp_path / "auto_skills.json", min_anchor_cosine=0.95
    )
    metrics = trainer.run_autonomous_cycle(dpo_steps=6)

    assert metrics["challenges_evaluated"] == 3
    assert metrics["episodic_instant_writes"] == 3
    assert set(metrics["recruited_skills"]) == {"lis_length", "fast_mod_exp", "topo_sort_kahn"}
    assert metrics["final_preference_margin"] > metrics["initial_preference_margin"]
    assert metrics["regression_rolled_back"] is False

    # Verify recruited tools execute live on CPU
    res = trainer.skill_forge.execute_with_recruited_skills("python: print(fast_mod_exp(17, 19, 1000000007))")
    assert res == "286765645"


def test_cpu_scan_speedup_benchmark():
    bench = benchmark_cpu_scan_vs_sequential_gru(seq_len=1024, hidden_dim=256, repeats=2)
    assert bench["cpu_speedup_x"] > 1.5
