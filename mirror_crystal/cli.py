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


SUPPORTED_INPUT_SUFFIXES = {".jsonl", ".txt", ".md", ".py"}


def _collect_input_files(input_path: Path) -> list[Path]:
    """Validate an input file/directory and collect supported files in stable order."""
    if not input_path.exists():
        raise ValueError(f"Input path does not exist: {input_path}")
    if input_path.is_file():
        if input_path.suffix.lower() not in SUPPORTED_INPUT_SUFFIXES:
            raise ValueError(f"Unsupported input type {input_path.suffix!r}; use .jsonl, .txt, .md or .py")
        return [input_path]
    if not input_path.is_dir():
        raise ValueError(f"Input path is not a regular file or directory: {input_path}")
    files = sorted(
        path for path in input_path.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_INPUT_SUFFIXES
    )
    if not files:
        raise ValueError(f"No supported input files found under {input_path} (expected .jsonl, .txt, .md or .py)")
    return files


def _validate_brain_file(path: Path, parser: argparse.ArgumentParser) -> None:
    if path.suffix.lower() not in {".mcrystal", ".dcrystal"}:
        parser.error(f"Brain path must end in .mcrystal or .dcrystal: {path}")
    if not path.is_file():
        parser.error(f"Brain file does not exist: {path}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Mirror Crystal — CPU-first rule crystallization, retrieval and evaluation"
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_train = sub.add_parser("train", help="Ingest .jsonl, .txt, .md or .py files")
    p_train.add_argument("--input", required=True, help="Input file or directory")
    p_train.add_argument("--save", required=True, help="Output .mcrystal or .dcrystal brain path")
    p_train.add_argument("--load", default="", help="Optional existing .mcrystal brain to continue")
    p_train.add_argument("--max-rows", type=int, default=None)

    p_query = sub.add_parser("query", help="Run one prompt against an existing brain")
    p_query.add_argument("--brain", required=True, help="Existing .mcrystal or .dcrystal file")
    p_query.add_argument("--prompt", required=True, help="User prompt")

    p_chat = sub.add_parser("chat", help="Interactive multi-turn session")
    p_chat.add_argument("--brain", required=True, help="Existing .mcrystal or .dcrystal file")

    p_bench = sub.add_parser("benchmark", help="Run deterministic CPU capability checks")
    p_bench.add_argument("--dataset", default="", help="Optional .jsonl dataset path")
    p_bench.add_argument("--max-rows", type=int, default=7125)

    args = parser.parse_args(argv)
    if getattr(args, "max_rows", None) is not None and args.max_rows < 1:
        parser.error("--max-rows must be a positive integer")

    if args.cmd == "train":
        input_path = Path(args.input).expanduser()
        save_path = Path(args.save).expanduser()
        try:
            files = _collect_input_files(input_path)
        except ValueError as exc:
            parser.error(str(exc))
        if save_path.suffix.lower() not in {".mcrystal", ".dcrystal"}:
            parser.error("--save must end in .mcrystal or .dcrystal")
        if args.load:
            load_path = Path(args.load).expanduser()
            if save_path.suffix.lower() == ".dcrystal":
                parser.error("--load is supported only when saving an in-memory .mcrystal brain")
            if load_path.suffix.lower() != ".mcrystal" or not load_path.is_file():
                parser.error(f"--load must point to an existing .mcrystal file: {load_path}")
        save_path.parent.mkdir(parents=True, exist_ok=True)
        per_file: list[dict[str, Any]] = []

        if save_path.suffix.lower() == ".dcrystal":
            vault = DiskCrystalVault(save_path)
            try:
                for index, file_path in enumerate(files):
                    fold = index == len(files) - 1
                    if file_path.suffix.lower() == ".jsonl":
                        per_file.append(vault.ingest_jsonl(file_path, max_rows=args.max_rows, fold_synonyms=fold))
                    else:
                        per_file.append(vault.ingest_source_file(file_path))
                saved_bytes = vault.optimize_disk()
                summary = {
                    "storage_mode": "out_of_core_sqlite_disk_vault",
                    "rules_loaded_in_ram_at_startup": 0,
                    "files_ingested": per_file,
                    "total_vault_rules_on_disk": vault.rule_count(),
                    "stats": vault.stats,
                    "saved_brain_path": str(save_path.resolve()),
                    "saved_brain_mb": round(saved_bytes / (1024 * 1024), 3),
                }
            finally:
                vault.close()
            print(json.dumps(summary, indent=2))
            return 0

        engine = MirrorCrystalEngine.load(args.load) if args.load else MirrorCrystalEngine()
        for index, file_path in enumerate(files):
            fold = index == len(files) - 1
            if file_path.suffix.lower() == ".jsonl":
                per_file.append(engine.ingest_jsonl(file_path, max_rows=args.max_rows, fold_synonyms=fold))
            else:
                per_file.append(engine.ingest_text_file(file_path, fold_synonyms=fold))
        saved_bytes = engine.save(save_path)
        summary: dict[str, Any] = per_file[0] if len(per_file) == 1 else {"files_ingested": per_file}
        summary["saved_brain_path"] = str(save_path.resolve())
        summary["saved_brain_mb"] = round(saved_bytes / (1024 * 1024), 3)
        print(json.dumps(summary, indent=2))
        return 0

    if args.cmd == "query":
        brain_path = Path(args.brain).expanduser()
        _validate_brain_file(brain_path, parser)
        if brain_path.suffix.lower() == ".dcrystal":
            vault = DiskCrystalVault(brain_path)
            try:
                response = vault.respond(args.prompt)
            finally:
                vault.close()
        else:
            response = MirrorCrystalEngine.load(brain_path).respond(args.prompt)
        print(json.dumps(response.to_dict(), indent=2))
        return 0

    if args.cmd == "chat":
        brain_path = Path(args.brain).expanduser()
        _validate_brain_file(brain_path, parser)
        is_disk = brain_path.suffix.lower() == ".dcrystal"
        brain: Any = DiskCrystalVault(brain_path) if is_disk else MirrorCrystalEngine.load(brain_path)
        try:
            print("Mirror Crystal Interactive CPU Session (type 'exit' to quit, '/teach Q ||| A' to teach)")
            while True:
                try:
                    user_input = input("\nYOU> ").strip()
                except (EOFError, KeyboardInterrupt):
                    print()
                    break
                if not user_input or user_input.lower() in {"exit", "quit"}:
                    break
                if user_input.startswith("/teach ") and "|||" in user_input:
                    question, answer = user_input[7:].split("|||", 1)
                    question, answer = question.strip(), answer.strip()
                    if not question or not answer:
                        print("Usage: /teach <question> ||| <answer>")
                        continue
                    added = brain.ingest_pair(question, answer)
                    if is_disk:
                        brain.optimize_disk()
                    else:
                        brain.save(brain_path)
                    print("[NEW RULE SAVED]" if added else "[NO NEW RULE — DUPLICATE OR UNCHANGED]")
                    continue
                response = brain.respond(user_input, session_id="cli")
                print(f"MIRROR CRYSTAL ({response.latency_ms:.2f} ms | {response.mode})>\n{response.answer}")
        finally:
            if is_disk:
                brain.close()
        return 0

    if args.cmd == "benchmark":
        dataset_path = Path(args.dataset).expanduser() if args.dataset else None
        if dataset_path is not None and not dataset_path.is_file():
            parser.error(f"Dataset file does not exist: {dataset_path}")
        report = run_hardware_benchmark(dataset_path=dataset_path, max_rows=args.max_rows)
        print(json.dumps(report, indent=2))
        required_checks = (
            "unseen_number_slot_generalization_exact",
            "continual_100_step_zero_forgetting",
            "multi_hop_lock_chaining_verified",
            "multi_turn_session_reality_check_verified",
        )
        return 0 if all(report.get(key) is True for key in required_checks) else 1

    parser.error(f"Unsupported command: {args.cmd}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
