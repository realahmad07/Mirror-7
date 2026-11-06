"""
Mirror 7 — Extreme Advanced Programming, Software Engineering & Polyglot Systems Mastery Pipeline.

Covers 24 Extreme Programming & Computer Science Categories:
1. 01_advanced_graph_algorithms (Dijkstra, Bellman-Ford, Tarjan SCC, A*)
2. 02_dynamic_programming_optimization (0/1 Knapsack, O(n log n) Patience LIS, Bitmask DP)
3. 03_advanced_data_structures (O(1) LRU Cache, Trie, Skip List, Treap)
4. 04_segment_tree_fenwick_range_queries (Fenwick BIT i & -i, Lazy Segment Tree)
5. 05_string_algorithms_kmp_trie (KMP Prefix Automaton, Rabin-Karp, Suffix Array)
6. 06_concurrency_lockfree_atomics (CAS Atomics, ABA Problem, Deadlock Coffman Conditions)
7. 07_rust_ownership_lifetimes_zero_cost (Borrow Checker, Lifetimes 'a, Send/Sync, Pin)
8. 08_cpp20_modern_systems_raii_move (Move Semantics, RAII, Cache-Line False Sharing alignas(64))
9. 09_gpu_cuda_triton_parallel_kernels (Warp Coalescing, Shared Memory Tiling, Online Softmax)
10. 10_compiler_design_ssa_llvm_jit (Lexing, AST, SSA Phi-Nodes, Graph-Coloring Register Allocation)
11. 11_operating_systems_virtual_memory_epoll (4-Level Page Tables, TLB, Copy-on-Write, epoll/io_uring)
12. 12_distributed_systems_raft_consensus (Raft Consensus, CAP/PACELC, Vector Clocks, 2PC)
13. 13_database_internals_btree_lsm_mvcc (B+ Trees vs LSM-Trees, Bloom Filters, MVCC Snapshot Isolation)
14. 14_cryptography_cybersecurity_exploits (Constant-Time HMAC, SQLi Parameterization, Memory Safety)
15. 15_asyncio_event_loop_coroutines (Cooperative Multitasking, Backpressure Semaphores, Epoll Loop)
16. 16_deep_learning_autograd_backprop_engine (Reverse-Mode Autograd DAG, Scaled Dot-Product Attention)
17. 17_functional_programming_monads_tail_call (Pure Functions, Immutability, Monadic Composition, TCO)
18. 18_network_protocols_tcp_bbr_quic_http3 (TCP 3-Way Handshake, BBR vs CUBIC, QUIC/HTTP3 0-RTT)
19. 19_software_architecture_solid_ddd_clean (SOLID, Hexagonal Architecture, Idempotency Keys)
20. 20_automated_debugging_root_cause_sanitizers (ASan/TSan Diagnostics, Mutable Default Bug)
21. 21_bitwise_algorithms_simd_popcount (Kernighan n & (n - 1), XOR Algebra, SIMD Vectorization)
22. 22_union_find_disjoint_set_kruskal (Path Compression & Union by Rank O(alpha(n)))
23. 23_topological_sort_kahn_cycle_detection (Kahn's Indegree Queue & 3-Color DFS Cycle Detection)
24. 24_formal_verification_hoare_logic_invariants (Hoare Triples {P} C {Q}, Loop Invariants, Variant Functions)
"""
from __future__ import annotations

import ast
import re

TWENTY_FOUR_EXTREME_PROGRAMMING_CATEGORIES: tuple[str, ...] = (
    "01_advanced_graph_algorithms",
    "02_dynamic_programming_optimization",
    "03_advanced_data_structures",
    "04_segment_tree_fenwick_range_queries",
    "05_string_algorithms_kmp_trie",
    "06_concurrency_lockfree_atomics",
    "07_rust_ownership_lifetimes_zero_cost",
    "08_cpp20_modern_systems_raii_move",
    "09_gpu_cuda_triton_parallel_kernels",
    "10_compiler_design_ssa_llvm_jit",
    "11_operating_systems_virtual_memory_epoll",
    "12_distributed_systems_raft_consensus",
    "13_database_internals_btree_lsm_mvcc",
    "14_cryptography_cybersecurity_exploits",
    "15_asyncio_event_loop_coroutines",
    "16_deep_learning_autograd_backprop_engine",
    "17_functional_programming_monads_tail_call",
    "18_network_protocols_tcp_bbr_quic_http3",
    "19_software_architecture_solid_ddd_clean",
    "20_automated_debugging_root_cause_sanitizers",
    "21_bitwise_algorithms_simd_popcount",
    "22_union_find_disjoint_set_kruskal",
    "23_topological_sort_kahn_cycle_detection",
    "24_formal_verification_hoare_logic_invariants",
)


def extract_executable_python(response_text: str) -> str:
    """Strip <think>...</think> blocks and return executable Python source code."""
    return re.sub(r"<think>.*?</think>", "", response_text, flags=re.DOTALL).strip()


def verify_python_ast(response_text: str) -> bool:
    """Verify that the code body outside <think>...</think> parses cleanly with Python's AST compiler."""
    code_str = extract_executable_python(response_text)
    if not code_str:
        return False
    try:
        ast.parse(code_str)
        return True
    except SyntaxError:
        return False
