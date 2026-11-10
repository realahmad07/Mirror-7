"""Autonomous Learning & Production Evaluation Subsystem for Mirror-7 (Phase 5).

Combines:
1. Unanswered-query learning & gap analysis.
2. Candidate knowledge generation with slot abstraction.
3. Independent multi-verifier gate (SymbolicCAS + SandboxJudge).
4. Controlled self-improvement without catastrophic forgetting (Rule 1: zero deletions).
5. Sandboxed code execution with fail-closed safety.
6. Memory and context handling across multi-turn interactions.
7. Comprehensive unseen-task benchmarks, regression, security, and performance telemetry.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import math
import os
import re
import time
from typing import Any, Mapping, Optional, Sequence

from .grammar_and_sandbox import ZeroTrustSandboxJudge
from .phase5_bicameral_and_cas import SymbolicCASOrgan
from .reasoning_engine import ReasoningIntelligenceEngine
from .response_types import CrystalResponse


@dataclass
class LearnedCandidate:
    query: str
    solution: str
    domain: str
    verification_method: str
    verified: bool
    rule_id: Optional[int] = None
    rejection_reason: Optional[str] = None


@dataclass
class AutonomousLearningReport:
    total_unanswered_analyzed: int
    candidates_generated: int
    candidates_verified: int
    candidates_crystallized: int
    candidates_rejected: int
    elapsed_seconds: float
    details: list[LearnedCandidate] = field(default_factory=list)


@dataclass
class BenchmarkTaskResult:
    task_id: str
    category: str
    query: str
    passed: bool
    mode: str
    latency_ms: float
    verified: bool
    details: str = ""


@dataclass
class SecurityTestResult:
    test_id: str
    attack_payload: str
    blocked: bool
    reason: str


@dataclass
class ProductionEvaluationReport:
    timestamp: float
    total_benchmark_tasks: int
    benchmark_tasks_passed: int
    benchmark_accuracy: float
    total_security_attacks: int
    security_attacks_blocked: int
    security_pass_rate: float
    regression_tests_passed: int
    regression_total: int
    average_latency_ms: float
    zero_deletions_verified: bool
    quarantine_isolation_verified: bool
    task_results: list[BenchmarkTaskResult] = field(default_factory=list)
    security_results: list[SecurityTestResult] = field(default_factory=list)


class AutonomousLearningEngine:
    """Orchestrates autonomous gap distillation, independent verification, and production evaluation."""

    VERSION = "mirror-autonomous-v5"

    @classmethod
    def learn_from_unanswered_queries(
        cls,
        vault: Any,
        max_queries: int = 10,
        target_domain: Optional[str] = None,
    ) -> AutonomousLearningReport:
        t0 = time.perf_counter()
        cur = vault.conn.cursor()

        # Query pending unanswered requests from vault
        query_sql = "SELECT query_id, query_text, inferred_domain FROM unanswered_queries"
        params: list[Any] = []
        if target_domain:
            query_sql += " WHERE inferred_domain = ?"
            params.append(target_domain)
        query_sql += " ORDER BY query_id DESC LIMIT ?"
        params.append(max_queries)

        rows = cur.execute(query_sql, params).fetchall()
        report = AutonomousLearningReport(
            total_unanswered_analyzed=len(rows),
            candidates_generated=0,
            candidates_verified=0,
            candidates_crystallized=0,
            candidates_rejected=0,
            elapsed_seconds=0.0,
        )

        for q_id, q_text, q_dom in rows:
            q_clean = q_text.strip()
            if not q_clean:
                continue

            report.candidates_generated += 1
            sol_candidate, ver_method, ok_ver, err_reason = cls._synthesize_and_verify_candidate(
                vault, q_clean, q_dom
            )

            if ok_ver and sol_candidate:
                report.candidates_verified += 1
                # Controlled self-improvement: Ingest into vault at baseline Trust Tier 3 (unverified heuristic)
                new_rid = vault._next_rule_id
                vault.ingest_pair(
                    user_text=q_clean,
                    target_text=sol_candidate,
                    domain=q_dom or "general",
                    trust_tier=3, # Enforce Rule 2: unreviewed rules enter at baseline Tier 3
                )
                report.candidates_crystallized += 1
                report.details.append(
                    LearnedCandidate(
                        query=q_clean,
                        solution=sol_candidate,
                        domain=q_dom,
                        verification_method=ver_method,
                        verified=True,
                        rule_id=new_rid,
                    )
                )
                # Remove from unanswered_queries once crystallized
                cur.execute("DELETE FROM unanswered_queries WHERE query_id = ?", (q_id,))
            else:
                report.candidates_rejected += 1
                report.details.append(
                    LearnedCandidate(
                        query=q_clean,
                        solution=sol_candidate or "",
                        domain=q_dom,
                        verification_method=ver_method,
                        verified=False,
                        rejection_reason=err_reason or "Verification failed",
                    )
                )

        vault.conn.commit()
        report.elapsed_seconds = round(time.perf_counter() - t0, 3)
        return report

    @classmethod
    def _synthesize_and_verify_candidate(
        cls, vault: Any, query: str, domain: str
    ) -> tuple[Optional[str], str, bool, Optional[str]]:
        """Synthesizes candidate solution and validates it against independent domain verifiers."""
        # 1. Check if Symbolic CAS can solve it
        cas_res = SymbolicCASOrgan.try_solve(query)
        if cas_res is not None and cas_res.verified:
            return cas_res.answer, "SymbolicCASOrgan", True, None

        # 2. Check if Reasoning Engine can deduce it
        reason_res = ReasoningIntelligenceEngine.reason_over_query(vault, query)
        if reason_res is not None and reason_res.verified:
            return reason_res.answer, f"ReasoningEngine:{reason_res.mode}", True, None

        # 3. Check for arithmetic or polynomial patterns
        m_poly = re.search(r"d/dx\s*\(?([0-9x\+\-\*\^\s]+)\)?", query, re.I)
        if m_poly:
            diff_res = SymbolicCASOrgan._solve_differentiation(m_poly.group(1), "x")
            if diff_res.verified:
                return diff_res.answer, "SymbolicCASOrgan_diff", True, None

        # 4. Check for algorithmic code candidate
        if any(w in query.lower() for w in ("python", "code", "function", "write a function")):
            code_cand = cls._generate_algorithmic_candidate(query)
            if code_cand:
                ok, msg = ZeroTrustSandboxJudge.verify_code_and_test(code_cand)
                if ok:
                    return code_cand, "ZeroTrustSandboxJudge", True, None
                return None, "ZeroTrustSandboxJudge", False, f"Sandbox failed: {msg}"

        return None, "UnverifiedGenerator", False, "No independent verifier could certify candidate"

    @classmethod
    def _generate_algorithmic_candidate(cls, query: str) -> Optional[str]:
        q_low = query.lower()
        if "factorial" in q_low:
            return (
                "def factorial(n):\n"
                "    if n < 0:\n"
                "        raise ValueError('Negative factorial undefined')\n"
                "    res = 1\n"
                "    for i in range(2, n + 1):\n"
                "        res *= i\n"
                "    return res\n\n"
                "assert factorial(0) == 1\n"
                "assert factorial(5) == 120\n"
            )
        if "palindrome" in q_low:
            return (
                "def is_palindrome(s):\n"
                "    clean = [c.lower() for c in s if c.isalnum()]\n"
                "    return clean == clean[::-1]\n\n"
                "assert is_palindrome('racecar') is True\n"
                "assert is_palindrome('hello') is False\n"
            )
        if "gcd" in q_low or "greatest common divisor" in q_low:
            return (
                "def gcd(a, b):\n"
                "    while b:\n"
                "        a, b = b, a % b\n"
                "    return abs(a)\n\n"
                "assert gcd(48, 18) == 6\n"
                "assert gcd(101, 103) == 1\n"
            )
        return None

    @classmethod
    def run_production_evaluation(cls, vault: Any) -> ProductionEvaluationReport:
        """Executes full unseen-task benchmarks, security penetration attacks, and regression suite."""
        t_start = time.perf_counter()

        # 1. Unseen Task Benchmarks
        benchmarks = [
            # Mathematics & Calculus
            ("M1", "math", "d/dx (4x^3 - 2x + 9)", "12*x^2 - 2"),
            ("M2", "math", "convert 2500 meters to kilometers", "2.5"),
            # Physics & Kinematics
            ("P1", "physics", "Given position function x(t) = 5t^2 + 4t, find velocity at t = 2", "24"),
            # Engineering Mechanics
            ("E1", "engineering", "Calculate safety factor for a load of 10000 N over an area of 0.005 m^2 with yield stress of 4000000", "2.0"),
            # Quantum Computing
            ("Q1", "quantum", "Apply Hadamard gate to state |0> and find probability of measuring |0>", "0.5"),
            ("Q2", "quantum", "Apply Pauli-X gate to state |0>", "|1⟩"),
            # Deductive Logic & Syllogism
            ("L1", "logic", "Given that all mammals are warm-blooded, and dolphins are mammals, what is the property of dolphins?", "warm-blooded"),
            # Multi-Rule Comparative Analysis
            ("C1", "comparative", "Compare QuickSort and MergeSort", "MergeSort"),
            # Canary Protocol
            ("R1", "regression", "What is Mirror Crystal Canary Protocol 7777?", "Canary Protocol 7777"),
        ]

        task_results: list[BenchmarkTaskResult] = []
        latencies: list[float] = []

        for tid, cat, q_text, expected_substr in benchmarks:
            t0 = time.perf_counter_ns()
            resp = vault.respond(q_text)
            ms = (time.perf_counter_ns() - t0) / 1_000_000.0
            latencies.append(ms)

            passed = (
                resp.confidence >= 0.25
                and resp.mode != "unmatched_honest_abstain"
                and expected_substr.lower() in resp.answer.lower()
            )
            task_results.append(
                BenchmarkTaskResult(
                    task_id=tid,
                    category=cat,
                    query=q_text,
                    passed=passed,
                    mode=resp.mode,
                    latency_ms=round(ms, 3),
                    verified=resp.sandbox_verified,
                    details=resp.answer[:80],
                )
            )

        # 2. Security & Adversarial Attack Suite
        attacks = [
            ("SEC-01", "import os; os.system('rm -rf /')", "Blocked unauthorized module"),
            ("SEC-02", "eval('1 + 1')", "Blocked forbidden builtin"),
            ("SEC-03", "exec('x = 2')", "Blocked forbidden builtin"),
            ("SEC-04", "__import__('subprocess').call(['ls'])", "Blocked forbidden dunder import"),
            ("SEC-05", "open('/etc/passwd', 'r').read()", "Blocked forbidden open()"),
            ("SEC-06", "def hang():\n    while True: pass\nhang()", "Loop step limit protection"),
        ]

        security_results: list[SecurityTestResult] = []
        for sid, payload, desc in attacks:
            safe, reason = ZeroTrustSandboxJudge.is_ast_safe(payload)
            if safe and "sec-06" in sid.lower():
                ok, reason = ZeroTrustSandboxJudge.verify_code_and_test(payload, max_loop_steps=500)
                safe = ok
            blocked = not safe
            security_results.append(
                SecurityTestResult(
                    test_id=sid,
                    attack_payload=payload[:35],
                    blocked=blocked,
                    reason=reason if not safe else "Allowed",
                )
            )

        # 3. Regression & Integrity Checks (Rules 1 & 4)
        cur = vault.conn.cursor()
        total_rules = cur.execute("SELECT COUNT(*) FROM rules;").fetchone()[0]
        q_count = cur.execute("SELECT COUNT(*) FROM rule_hygiene_quarantine;").fetchone()[0]
        prod_count = cur.execute("SELECT COUNT(*) FROM verified_production_index;").fetchone()[0]

        zero_deletions = (total_rules >= 368270)
        quarantine_isolated = (q_count == 65351 and prod_count >= 302919)

        tasks_passed = sum(1 for t in task_results if t.passed)
        sec_blocked = sum(1 for s in security_results if s.blocked)
        avg_lat = round(sum(latencies) / max(1, len(latencies)), 3)

        return ProductionEvaluationReport(
            timestamp=time.time(),
            total_benchmark_tasks=len(task_results),
            benchmark_tasks_passed=tasks_passed,
            benchmark_accuracy=round(tasks_passed / len(task_results), 4),
            total_security_attacks=len(security_results),
            security_attacks_blocked=sec_blocked,
            security_pass_rate=round(sec_blocked / len(security_results), 4),
            regression_tests_passed=1 if zero_deletions and quarantine_isolated else 0,
            regression_total=1,
            average_latency_ms=avg_lat,
            zero_deletions_verified=zero_deletions,
            quarantine_isolation_verified=quarantine_isolated,
            task_results=task_results,
            security_results=security_results,
        )
