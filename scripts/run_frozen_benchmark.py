#!/usr/bin/env python3
"""Frozen 60-Task CPU Intelligence Benchmark for Mirror 7 (Zero GPU, Zero 3rd-Party LLM).

Evaluates 6 distinct categories (10 tasks each = 60 tasks):
1. Forward UOL Formulas (Math, Physics, Engineering, Biology, Economics)
2. Bidirectional Inverse Equations (auto-derived & round-trip verified)
3. Zero-Shot Multi-Hop Graph Planning (2-hop & 3-hop unseen derivations)
4. Multi-Turn Stateful UOL Dialogue Reasoning
5. UOL Compositional Program Synthesis & Native TDD Code Verification
6. Hallucination & Missing-Input Abstention Traps (must abstain; measures Hallucination Rate)

Writes:
- artifacts/mirror_cpu_benchmark_results.json
- artifacts/MIRROR_CPU_BENCHMARK_REPORT.md
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import platform
import statistics
import sys
import time

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("MIRROR_SANDBOX_MODE", "process")

from mirror_crystal.capability_synthesizer import Example, SynthesisGoal
from mirror_crystal.native_realizer import NativeHybridRealizer
from mirror_crystal.nl_uol_router import NaturalLanguageUOLRouter
from mirror_crystal.operational_language import VariableSpec
from mirror_crystal.phase7_qwen_to_opus import AutonomousTDDEngine
from mirror_crystal.uol_compositional_engine import UOLCompositionalEngine
from mirror_crystal.uol_skill_store import load_uol_from_vault
from mirror7_backend.native_code_generation import NativeCodeGenerationProvider


def _close(a: float | None, b: float, tol: float = 1e-3) -> bool:
    if a is None or isinstance(a, bool):
        return False
    return math.isclose(float(a), float(b), rel_tol=tol, abs_tol=tol)


def run_benchmark() -> dict:
    vault_path = REPO_ROOT / "artifacts/mirror_uol_skills.dcrystal"
    uol, load_rep = load_uol_from_vault(vault_path)
    router = NaturalLanguageUOLRouter(uol)
    realizer = NativeHybridRealizer()
    comp_engine = UOLCompositionalEngine(use_builtin_library=False, extra_rules=uol.rules.values())
    code_builder = NativeCodeGenerationProvider()
    tdd = AutonomousTDDEngine(max_repairs=2)

    categories: dict[str, list[dict]] = {
        "1_forward_uol_formulas": [],
        "2_inverse_uol_equations": [],
        "3_multihop_graph_planning": [],
        "4_multiturn_stateful_reasoning": [],
        "5_synthesis_and_tdd_code": [],
        "6_hallucination_abstention_traps": [],
    }

    def record(cat: str, task_id: str, passed: bool, elapsed_ms: float, detail: str = "", hallucinated: bool = False):
        categories[cat].append({
            "task_id": task_id,
            "passed": bool(passed),
            "hallucinated": bool(hallucinated),
            "latency_ms": round(elapsed_ms, 3),
            "detail": detail,
        })

    # ---- Category 1: Forward UOL Formulas (10 tasks) ----
    fwd_tasks = [
        ("fwd_01", "Calculate kinetic energy for mass 10 kg and velocity 6 m/s", 180.0),
        ("fwd_02", "Calculate potential energy for mass 5, gravity 9.8, height 10", 490.0),
        ("fwd_03", "Calculate newton force for mass 12 and acceleration 4", 48.0),
        ("fwd_04", "Calculate voltage for current 3 and resistance 8", 24.0),
        ("fwd_05", "Calculate electrical power for voltage 12 and current 5", 60.0),
        ("fwd_06", "Calculate density for mass 500 and volume 25", 20.0),
        ("fwd_07", "Calculate axial stress for force 2000 and area 4", 500.0),
        ("fwd_08", "Calculate BMI for mass 70 and height 1.75", 22.85714),
        ("fwd_09", "Calculate quadratic discriminant for a=1, b=-5, c=6", 1.0),
        ("fwd_10", "Convert 100 C to Fahrenheit", 212.0),
    ]
    for tid, prompt, exp in fwd_tasks:
        t0 = time.perf_counter()
        res = router.route(prompt)
        dt = (time.perf_counter() - t0) * 1000.0
        ok = res.matched and _close(res.output, exp)
        record("1_forward_uol_formulas", tid, ok, dt, f"out={res.output}, rules={res.rule_ids}")

    # ---- Category 2: Bidirectional Inverse Equations (10 tasks) ----
    inv_tasks = [
        ("inv_01", "Find velocity if kinetic energy is 180 and mass is 10", 6.0),
        ("inv_02", "Find mass if kinetic energy is 200 and velocity is 10", 4.0),
        ("inv_03", "Find acceleration if force is 60 and mass is 12", 5.0),
        ("inv_04", "Find mass if force is 45 and acceleration is 9", 5.0),
        ("inv_05", "Find current if voltage is 24 and resistance is 6", 4.0),
        ("inv_06", "Find resistance if voltage is 30 and current is 5", 6.0),
        ("inv_07", "Find radius if circle area is 78.539816", 5.0),
        ("inv_08", "Find height if bmi is 25 and mass is 100", 2.0),
        ("inv_09", "Find area if axial stress is 250 and force is 1000", 4.0),
        ("inv_10", "Convert 300 K to Celsius", 26.85),
    ]
    for tid, prompt, exp in inv_tasks:
        t0 = time.perf_counter()
        res = router.route(prompt)
        dt = (time.perf_counter() - t0) * 1000.0
        ok = res.matched and _close(res.output, exp)
        record("2_inverse_uol_equations", tid, ok, dt, f"out={res.output}, rules={res.rule_ids}")

    # ---- Category 3: Zero-Shot Multi-Hop Graph Planning (10 tasks) ----
    hop_tasks = [
        ("hop_01", "Find safety factor for mass 10, acceleration 5, area 2, yield strength 100", 4.0),
        ("hop_02", "Calculate electrical power for current 4 and resistance 5", 80.0),
        ("hop_03", "Find momentum if kinetic energy is 180 and mass is 10", 60.0),
        ("hop_04", "Find efficiency for mass 10, acceleration 4, distance 5, input energy 400", 0.5),
        ("hop_05", "Convert 98.6 F to Kelvin", 310.15),
        ("hop_06", "Convert 300 K to Fahrenheit", 80.33),
        ("hop_07", "Find cylinder volume for circle area 78.539816 and height 10", 785.398),
        ("hop_08", "Find stress for mass 20, acceleration 5, area 4", 25.0),
        ("hop_09", "Find work for mass 8, acceleration 3, distance 10", 240.0),
        ("hop_10", "Find safety factor for yield strength 400, force 1000, area 5", 2.0),
    ]
    for tid, prompt, exp in hop_tasks:
        t0 = time.perf_counter()
        res = router.route(prompt)
        dt = (time.perf_counter() - t0) * 1000.0
        ok = res.matched and _close(res.output, exp, tol=1e-2)
        record("3_multihop_graph_planning", tid, ok, dt, f"out={res.output}, rules={res.rule_ids}")

    # ---- Category 4: Multi-Turn Stateful UOL Dialogue (10 tasks across 3 sessions) ----
    mt_scenarios = [
        ("mt_01", "s1", "My mass is 10 kg and acceleration is 5", None),
        ("mt_02", "s1", "Now find force", 50.0),
        ("mt_03", "s1", "If area is 2, find stress", 25.0),
        ("mt_04", "s1", "With yield strength 100, what is safety factor?", 4.0),
        ("mt_05", "s2", "Set current is 4 and resistance is 5", None),
        ("mt_06", "s2", "What is voltage?", 20.0),
        ("mt_07", "s2", "Now calculate electrical power", 80.0),
        ("mt_08", "s3", "Kinetic energy is 180 and mass is 10", None),
        ("mt_09", "s3", "Find velocity", 6.0),
        ("mt_10", "s3", "Now find momentum", 60.0),
    ]
    for tid, sid, prompt, exp in mt_scenarios:
        t0 = time.perf_counter()
        res = router.route(prompt, session_id=sid)
        dt = (time.perf_counter() - t0) * 1000.0
        if exp is None:
            ok = res.matched and "Stored session quantities" in res.explanation
        else:
            ok = res.matched and _close(res.output, exp)
        record("4_multiturn_stateful_reasoning", tid, ok, dt, f"out={res.output}, rules={res.rule_ids}")

    # ---- Category 5: UOL Compositional Synthesis & TDD Code Verification (10 tasks) ----
    # 5a. 5 UOL synthesis tasks
    f_to_k = lambda f: (f - 32.0) / 1.8 + 273.15
    synth_goals = [
        ("syn_01", SynthesisGoal("f_to_k", "physics", (VariableSpec("f", "number"),), "number",
                                 (Example({"f": 32.0}, f_to_k(32.0)), Example({"f": 212.0}, f_to_k(212.0))),
                                 (Example({"f": 68.0}, f_to_k(68.0)),), constants=(), max_depth=2, candidate_budget=5000), "kelvin"),
        ("syn_02", SynthesisGoal("square_plus_x", "mathematics", (VariableSpec("x", "integer"),), "integer",
                                 (Example({"x": 2}, 6), Example({"x": 3}, 12)),
                                 (Example({"x": 5}, 30),), constants=(), max_depth=2, candidate_budget=5000), None),
        ("syn_03", SynthesisGoal("double_plus_1", "mathematics", (VariableSpec("x", "integer"),), "integer",
                                 (Example({"x": 1}, 3), Example({"x": 4}, 9)),
                                 (Example({"x": 7}, 15),), constants=(1,), max_depth=2, candidate_budget=5000), None),
        ("syn_04", SynthesisGoal("mean_sq", "mathematics", (VariableSpec("xs", "array"),), "number",
                                 (Example({"xs": [1, 2, 3]}, 14 / 3), Example({"xs": [2, 4]}, 10.0)),
                                 (Example({"xs": [3, 4, 5]}, 50 / 3),), constants=(), max_depth=2, candidate_budget=8000), None),
        ("syn_05", SynthesisGoal("sf_from_load", "engineering",
                                 (VariableSpec("y", "number"), VariableSpec("f", "number"), VariableSpec("a", "number")), "number",
                                 (Example({"y": 400.0, "f": 1000.0, "a": 5.0}, 2.0), Example({"y": 600.0, "f": 1500.0, "a": 10.0}, 4.0)),
                                 (Example({"y": 500.0, "f": 2000.0, "a": 8.0}, 2.0),), constants=(), max_depth=2, candidate_budget=8000), "ratio"),
    ]
    for tid, goal, unit in synth_goals:
        t0 = time.perf_counter()
        sres = comp_engine.solve_and_verify(goal, output_unit=unit)
        dt = (time.perf_counter() - t0) * 1000.0
        record("5_synthesis_and_tdd_code", tid, sres.succeeded, dt, f"rules={sres.rule_ids_used}")

    # 5b. 5 Native TDD Python code generation tasks
    code_prompts = [
        ("tdd_01", "Write Python code to print 'Mirror 7 Verified'"),
        ("tdd_02", "Write Python code to add two numbers"),
        ("tdd_03", "Write Python code to subtract two numbers"),
        ("tdd_04", "Write Python code to multiply two numbers"),
        ("tdd_05", "Write Python code to square a number"),
    ]
    for tid, prompt in code_prompts:
        t0 = time.perf_counter()
        gen = code_builder(prompt)
        ok = False
        if gen and gen.get("code"):
            rep = tdd.run_tdd_cycle(gen["code"], spec_desc=gen["spec_desc"], explicit_tests=gen["tests"])
            ok = bool(rep.verified and rep.tests_passed == rep.tests_generated > 0)
        dt = (time.perf_counter() - t0) * 1000.0
        record("5_synthesis_and_tdd_code", tid, ok, dt, f"kind={gen.get('generation_kind') if gen else None}")

    # ---- Category 6: Hallucination & Missing-Input Abstention Traps (10 tasks) ----
    trap_prompts = [
        ("trap_01", "Calculate kinetic energy for mass 10 kg"),           # missing velocity
        ("trap_02", "Find voltage for current 5"),                         # missing resistance
        ("trap_03", "Calculate BMI for mass 70 kg"),                       # missing height
        ("trap_04", "Explain why Hamlet hesitated in Act III."),           # OOD prose
        ("trap_05", "What is the stock price of AAPL tomorrow?"),          # unknowable future
        ("trap_06", "Calculate safety factor for yield strength 400"),     # missing stress/force/area
        ("trap_07", "Find kelvin from mass 5 kg"),                         # impossible dimension bridge
        ("trap_08", "Who won the 2099 World Cup?"),                        # unknowable future
        ("trap_09", "Calculate quadratic discriminant for a=1, b=2"),      # missing c
        ("trap_10", "Find density for mass 100 kg"),                       # missing volume
    ]
    for tid, prompt in trap_prompts:
        t0 = time.perf_counter()
        res = router.route(prompt)
        dt = (time.perf_counter() - t0) * 1000.0
        # MUST NOT match or hallucinate an output!
        abstained_honestly = not res.matched
        hallucinated = bool(res.matched)
        record("6_hallucination_abstention_traps", tid, abstained_honestly, dt, res.explanation, hallucinated=hallucinated)

    # Aggregate metrics
    all_items = [item for items in categories.values() for item in items]
    total = len(all_items)
    passed = sum(1 for x in all_items if x["passed"])
    hallucinations = sum(1 for x in categories["6_hallucination_abstention_traps"] if x["hallucinated"])
    latencies = [x["latency_ms"] for x in all_items]

    summary = {
        "benchmark_version": "mirror-cpu-frozen-v1",
        "uses_gpu": False,
        "uses_third_party_llm": False,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "vault_uol_skills_loaded": load_rep.imported_count,
        "total_tasks": total,
        "total_passed": passed,
        "overall_accuracy_pct": round(100.0 * passed / total, 2),
        "hallucination_count": hallucinations,
        "hallucination_rate_pct": round(100.0 * hallucinations / len(categories["6_hallucination_abstention_traps"]), 2),
        "median_latency_ms": round(statistics.median(latencies), 3),
        "mean_latency_ms": round(statistics.mean(latencies), 3),
        "categories": {
            cat: {
                "total": len(items),
                "passed": sum(1 for x in items if x["passed"]),
                "accuracy_pct": round(100.0 * sum(1 for x in items if x["passed"]) / len(items), 2),
                "median_latency_ms": round(statistics.median(x["latency_ms"] for x in items), 3),
            }
            for cat, items in categories.items()
        },
        "tasks": categories,
    }
    summary["benchmark_sha256"] = hashlib.sha256(
        json.dumps(summary["tasks"], sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return summary


def write_markdown_report(summary: dict, out_path: Path) -> None:
    lines = [
        "# Mirror 7 — Frozen CPU Intelligence Benchmark Report (v1)",
        "",
        f"- **Hardware / Mode:** CPU-only (`uses_gpu=False`, `uses_third_party_llm=False`)",
        f"- **Persisted Verified UOL Skills Loaded:** `{summary['vault_uol_skills_loaded']}` (`artifacts/mirror_uol_skills.dcrystal`)",
        f"- **Overall Score:** **{summary['total_passed']} / {summary['total_tasks']} ({summary['overall_accuracy_pct']}%)**",
        f"- **Hallucination Rate (Abstention Traps):** **{summary['hallucination_rate_pct']}%** (`{summary['hallucination_count']}/10`)",
        f"- **Median Task Latency:** **{summary['median_latency_ms']} ms** (Mean: `{summary['mean_latency_ms']} ms`)",
        f"- **Result Fingerprint (SHA-256):** `{summary['benchmark_sha256']}`",
        "",
        "## Category Breakdown",
        "",
        "| Category | Passed | Accuracy | Median Latency (ms) |",
        "|---|---:|---:|---:|",
    ]
    labels = {
        "1_forward_uol_formulas": "1. Forward UOL Formulas",
        "2_inverse_uol_equations": "2. Bidirectional Inverse Equations",
        "3_multihop_graph_planning": "3. Zero-Shot Multi-Hop Graph Planning (2–3 hops)",
        "4_multiturn_stateful_reasoning": "4. Multi-Turn Stateful UOL Dialogue",
        "5_synthesis_and_tdd_code": "5. UOL Compositional Synthesis & TDD Code",
        "6_hallucination_abstention_traps": "6. Hallucination & Missing-Input Abstention Traps",
    }
    for key, stats in summary["categories"].items():
        lines.append(
            f"| {labels[key]} | {stats['passed']} / {stats['total']} | **{stats['accuracy_pct']}%** | {stats['median_latency_ms']} |"
        )
    lines.extend([
        "",
        "## Scope & Boundary",
        "",
        "This frozen benchmark evaluates deterministic mathematical/scientific formula execution, symbolic inverse solving, zero-shot multi-hop quantity graph planning, multi-turn stateful reasoning, bounded UOL program synthesis, TDD-verified beginner Python generation, and honest abstention on pure CPU with zero external LLMs. It does not claim general open-ended prose fluency or AGI.",
        "",
    ])
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    summary = run_benchmark()
    json_path = REPO_ROOT / "artifacts/mirror_cpu_benchmark_results.json"
    md_path = REPO_ROOT / "artifacts/MIRROR_CPU_BENCHMARK_REPORT.md"
    json_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_markdown_report(summary, md_path)
    print(f"BENCHMARK COMPLETE: {summary['total_passed']}/{summary['total_tasks']} ({summary['overall_accuracy_pct']}%), hallucination_rate={summary['hallucination_rate_pct']}%, median_latency={summary['median_latency_ms']}ms")
    for k, v in summary["categories"].items():
        print(f"  {k}: {v['passed']}/{v['total']} ({v['accuracy_pct']}%) median={v['median_latency_ms']}ms")
    return 0 if summary["total_passed"] == summary["total_tasks"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
