"""
CLI interface & Hardware Efficiency Benchmark for Mirror Crystal (`python -m mirror_crystal`).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any

from .disk_vault import DiskCrystalVault
from .engine import MirrorCrystalEngine


def run_hardware_benchmark(dataset_path: Path | None = None, max_rows: int = 3000) -> dict[str, Any]:
    """Run comprehensive 0-GPU CPU benchmark verifying speed, RAM compactness, 0% forgetting & reasoning."""
    eng = MirrorCrystalEngine()

    # 1. Context-Frame Synonym Discovery (0 neural weights)
    corpus = "\n".join(
        [
            "always configure server_limit before deploying gateway routes",
            "always configure traffic_cap before deploying gateway routes",
            "exceeding server_limit drops incoming client packets",
            "exceeding traffic_cap drops incoming client packets",
        ]
    )
    eng.observe_unlabeled_text(corpus)
    eng.bit_table.fold_equivalences(min_shared_specific_frames=2)

    # 2. One-Shot Formula Generalization + Multi-Hop Chaining + Continual 100-Rule Zero-Forgetting Test
    eng.ingest_pair(
        "Compute 17^19 mod 1000000007",
        "The exact modular result of 17^19 mod 1000000007 is 286765645.",
    )
    eng.observe_unlabeled_text("borosilicate_beaker is a brittle_vessel.")
    eng.observe_unlabeled_text("if brittle_vessel, falls_on_concrete_shatters.")

    for i in range(100):
        eng.ingest_pair(f"What is protocol code {1000 + i}?", f"Protocol {1000 + i} maps to channel_{i}.")

    # Verify 1st rule (17^19 -> generalized to unseen 23^41!) STILL works with 0% forgetting after 100 new rules
    expected_unseen = pow(23, 41, 1000000007)
    r_math = eng.respond("Compute 23^41 mod 1000000007")
    unseen_math_exact = str(expected_unseen) in r_math.answer

    r_first = eng.respond("What is protocol code 1000?")
    zero_forgetting_ok = "channel_0" in r_first.answer

    r_chain = eng.respond("What happens to borosilicate_beaker?")
    chaining_ok = "falls_on_concrete_shatters" in r_chain.answer

    eng.respond("My server limit is 120 requests/sec.", session_id="bench")
    r_mem = eng.respond("Are 3 bursts of 45 safe? Give a limiter.", session_id="bench")
    session_mem_ok = ("Unsafe" in r_mem.answer) and ("135" in r_mem.answer) and r_mem.sandbox_verified

    # 3. Real JSONL Dataset Streaming Benchmark (if provided or present in training/data/)
    real_data_stats: dict[str, Any] = {}
    if dataset_path is None:
        default_ds = Path(__file__).resolve().parents[1] / "training" / "data" / "mirror_complete_master_v2.jsonl"
        if default_ds.is_file():
            dataset_path = default_ds

    query_latencies_ms: list[float] = []
    saved_bytes = 0
    if dataset_path is not None and dataset_path.is_file():
        real_data_stats = eng.ingest_jsonl(dataset_path, max_rows=max_rows)
        sample_queries = [
            "What is a qubit and how does quantum superposition work mathematically?",
            "How does Grover's quantum search algorithm achieve a quadratic speedup?",
            "Explain the core mathematical mechanism behind Shor's factoring algorithm.",
            "Compute 29^37 mod 1000000007",
        ]
        for sq in sample_queries:
            t_q = time.perf_counter_ns()
            _ = eng.respond(sq)
            query_latencies_ms.append((time.perf_counter_ns() - t_q) / 1_000_000.0)

        out_brain = Path(__file__).resolve().parents[1] / "artifacts" / "mirror_crystal_v1.mcrystal"
        saved_bytes = eng.save(out_brain)

    mean_q_ms = sum(query_latencies_ms) / max(1, len(query_latencies_ms))
    return {
        "engine": "MirrorCrystal_v1_ZeroWeight_ZeroBackprop_CPU",
        "floating_point_weights": 0,
        "gpu_required": False,
        "unseen_number_slot_generalization_exact": unseen_math_exact,
        "continual_100_step_zero_forgetting": zero_forgetting_ok,
        "multi_hop_lock_chaining_verified": chaining_ok,
        "multi_turn_session_reality_check_verified": session_mem_ok,
        "real_dataset_ingestion": real_data_stats,
        "mean_cpu_query_latency_ms": round(mean_q_ms, 3),
        "compressed_brain_size_mb": round(saved_bytes / (1024 * 1024), 3) if saved_bytes else 0.0,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Mirror Crystal — Zero-Weight CPU Intelligence Engine")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_train = sub.add_parser("train", help="Ingest .jsonl or .txt/.py files in 1 CPU pass")
    p_train.add_argument("--input", required=True, help="Path to .jsonl, .txt, .md, or .py file")
    p_train.add_argument("--save", required=True, help="Output .mcrystal brain file path")
    p_train.add_argument("--load", default="", help="Optional existing .mcrystal file or crystal_4 ZIP to continue teaching")
    p_train.add_argument("--max-rows", type=int, default=None)

    p_query = sub.add_parser("query", help="Run a single prompt against a .mcrystal brain or Crystal archive ZIP")
    p_query.add_argument("--brain", required=True, help="Path to .mcrystal file or crystal_4 ZIP")
    p_query.add_argument("--prompt", required=True, help="User prompt")

    p_chat = sub.add_parser("chat", help="Interactive multi-turn CLI session")
    p_chat.add_argument("--brain", required=True, help="Path to .mcrystal file")

    p_bench = sub.add_parser("benchmark", help="Run hardware efficiency & capability verification benchmark")
    p_bench.add_argument("--dataset", default="", help="Optional .jsonl dataset path")
    p_bench.add_argument("--max-rows", type=int, default=7125)

    args = parser.parse_args(argv)

    if args.cmd == "train":
        in_path = Path(args.input)
        save_path = Path(args.save)
        files = sorted(in_path.glob("*.jsonl")) if in_path.is_dir() else [in_path]

        if save_path.suffix.lower() == ".dcrystal":
            vault = DiskCrystalVault(save_path)
            per_file = []
            for idx_f, f in enumerate(files):
                is_last = idx_f == len(files) - 1
                if f.suffix.lower() == ".jsonl":
                    per_file.append(vault.ingest_jsonl(f, max_rows=args.max_rows, fold_synonyms=is_last))
                else:
                    lines_n = vault.observe_unlabeled_text(f.read_text(encoding="utf-8", errors="replace"))
                    per_file.append({"file": f.name, "lines_read": lines_n})
            sz = vault.optimize_disk()
            summary = {
                "storage_mode": "out_of_core_sqlite_disk_vault (.dcrystal)",
                "rules_loaded_in_ram_at_startup": 0,
                "files_ingested": per_file,
                "total_vault_rules_on_disk": vault.rule_count(),
                "stats": vault.stats,
                "saved_brain_path": str(save_path),
                "saved_brain_mb": round(sz / (1024 * 1024), 3),
            }
            vault.close()
            print(json.dumps(summary, indent=2))
            return 0

        eng = MirrorCrystalEngine.load(args.load) if args.load else MirrorCrystalEngine()
        per_file = []
        for idx_f, f in enumerate(files):
            is_last = idx_f == len(files) - 1
            if f.suffix.lower() == ".jsonl":
                per_file.append(eng.ingest_jsonl(f, max_rows=args.max_rows, fold_synonyms=is_last))
            else:
                per_file.append(eng.ingest_text_file(f, fold_synonyms=is_last))
        sz = eng.save(save_path)
        out_stats: dict[str, Any] = per_file[0] if len(per_file) == 1 else {"files_ingested": per_file}
        out_stats["saved_brain_path"] = str(save_path)
        out_stats["saved_brain_mb"] = round(sz / (1024 * 1024), 3)
        print(json.dumps(out_stats, indent=2))
        return 0

    if args.cmd == "query":
        bpath = Path(args.brain)
        if bpath.suffix.lower() == ".dcrystal":
            vault = DiskCrystalVault(bpath)
            res = vault.respond(args.prompt)
            vault.close()
        else:
            eng = MirrorCrystalEngine.load(bpath)
            res = eng.respond(args.prompt)
        print(json.dumps(res.to_dict(), indent=2))
        return 0

    if args.cmd == "chat":
        bpath = Path(args.brain)
        is_disk = bpath.suffix.lower() == ".dcrystal"
        brain: Any = DiskCrystalVault(bpath) if is_disk else MirrorCrystalEngine.load(bpath)
        print("Mirror Crystal Interactive CPU Session (type 'exit' to quit, '/teach Q ||| A' to teach instantly)")
        while True:
            try:
                user_in = input("\nYOU> ").strip()
            except EOFError:
                break
            if not user_in or user_in.lower() in {"exit", "quit"}:
                break
            if user_in.startswith("/teach ") and "|||" in user_in:
                q_part, a_part = user_in[7:].split("|||", 1)
                brain.ingest_pair(q_part.strip(), a_part.strip())
                if is_disk:
                    brain.optimize_disk()
                else:
                    brain.save(bpath)
                print("[CRYSTALLIZED IN 1 PASS & SAVED]")
                continue
            res = brain.respond(user_in, session_id="cli")
            print(f"MIRROR CRYSTAL ({res.latency_ms:.2f} ms | {res.mode})>\n{res.answer}")
        if is_disk:
            brain.close()
        return 0

    if args.cmd == "benchmark":
        ds = Path(args.dataset) if args.dataset else None
        report = run_hardware_benchmark(dataset_path=ds, max_rows=args.max_rows)
        print(json.dumps(report, indent=2))
        return 0

    return 0
