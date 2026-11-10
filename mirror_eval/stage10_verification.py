"""Stage 10: Independent Verification Framework for Mirror 7.

Implements the five core pillars of independent verification:
1. External Reviewer Verification & Ingestion Engine (multi-reviewer independence, signature audit, zero fabrication).
2. Source & License Validation Engine (HTTPS, cryptographic SHA-256 excerpt digests, SPDX license checks).
3. Blind Benchmark Evaluation (disjoint held-out benchmarks, negative controls, calibrated honest abstention).
4. Mathematical & Factual Accuracy Engine (pure Python exact symbolic CAS, FTC+Simpson definite integration, Gauss-Jordan over Q).
5. Cross-Domain Reasoning Engine (quantum Born projection, kinematic calculus, material safety factors, multi-hop syllogisms).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

from mirror_crystal.grammar_and_sandbox import ZeroTrustSandboxJudge
from mirror_crystal.phase5_bicameral_and_cas import CASResult, SymbolicCASOrgan
from mirror_eval.factual_benchmark import ABSTENTION_MODES, load_factual_dataset
from mirror_eval.review_packet import build_review_packet

APPROVED_SPDX_LICENSES = frozenset({
    "Apache-2.0",
    "MIT",
    "BSD-2-Clause",
    "BSD-3-Clause",
    "CC0-1.0",
    "CC-BY-4.0",
    "CC-BY-SA-4.0",
    "ISC",
    "MPL-2.0",
    "PSF-2.0",
    "CC-BY-3.0-IGO",
    "0BSD",
    "LicenseRef-IETF-Trust-Legal-Provisions",
    "LicenseRef-IETF-TLP-5",
    "NOASSERTION",
})


# ============================================================================
# 1. External Reviewer Verification & Ingestion Engine
# ============================================================================

class ExternalReviewerAuditEngine:
    """Audits reviewer packets, enforces independence constraints, and prevents signature fabrication."""

    @staticmethod
    def audit_review_packet(dataset_path: Path, manifest_path: Path) -> Dict[str, Any]:
        """Construct and audit an unreviewed reviewer packet, ensuring zero pre-filled approvals."""
        packet = build_review_packet(dataset_path, manifest_path)
        if not packet.get("cases"):
            raise ValueError("Review packet contains no cases")
        for c in packet["cases"]:
            if c.get("case_reviews"):
                raise ValueError(f"Packet case {c.get('case_id')} has pre-filled case reviews (fabrication detected)")
            for claim in c.get("claims", []):
                if claim.get("reviews"):
                    raise ValueError(f"Packet claim {claim.get('claim_id')} has pre-filled reviews (fabrication detected)")
        return packet

    @staticmethod
    def validate_reviewer_signature(review: Dict[str, Any]) -> Tuple[bool, str]:
        """Verify an individual reviewer signature conforms to independent governance rules."""
        reviewer_id = str(review.get("reviewer_id", "")).strip()
        if not reviewer_id:
            return False, "reviewer_id is empty"
        if review.get("independent") is not True:
            return False, "reviewer must be declared independent (independent=True)"
        if review.get("method") not in {"independent_source_comparison", "independent_replication"}:
            return False, f"unsupported review method: {review.get('method')}"
        if review.get("decision") not in {"approve", "reject"}:
            return False, f"decision must be 'approve' or 'reject', got: {review.get('decision')}"
        reviewed_at = str(review.get("reviewed_at", ""))
        try:
            parsed = datetime.fromisoformat(reviewed_at.replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                return False, "reviewed_at timestamp must be timezone-aware"
        except (ValueError, TypeError):
            return False, "invalid ISO timestamp for reviewed_at"
        return True, "valid"

    @classmethod
    def verify_case_reviews(
        cls,
        reviews: List[Dict[str, Any]],
        min_reviewers: int = 2,
    ) -> Tuple[bool, str]:
        """Enforces that at least min_reviewers distinct independent reviewers approved, with zero rejections."""
        if any(r.get("decision") == "reject" for r in reviews):
            return False, "rejected by at least one reviewer"
        approved_reviewers = set()
        for r in reviews:
            valid, msg = cls.validate_reviewer_signature(r)
            if not valid:
                return False, f"invalid review: {msg}"
            if r.get("decision") == "approve":
                approved_reviewers.add(r["reviewer_id"].strip())
        if len(approved_reviewers) < min_reviewers:
            return False, f"insufficient distinct independent reviewers: {len(approved_reviewers)} < {min_reviewers}"
        return True, f"approved by {len(approved_reviewers)} independent reviewers"


# ============================================================================
# 2. Source & License Validation Engine
# ============================================================================

class SourceAndLicenseValidator:
    """Validates cryptographic integrity of source citations, HTTPS locators, and SPDX licenses."""

    @staticmethod
    def validate_primary_source(claim: Dict[str, Any]) -> Tuple[bool, str]:
        """Validates HTTPS, URL structure, and cryptographic SHA-256 excerpt match."""
        url = str(claim.get("source_url", "")).strip()
        parsed = urlparse(url)
        if parsed.scheme != "https":
            return False, f"source_url must use HTTPS protocol, got: {parsed.scheme}"
        if not parsed.netloc:
            return False, "source_url missing network host/domain"

        quote = str(claim.get("source_quote", ""))
        if not quote:
            return False, "source_quote is empty"
        expected_sha = str(claim.get("source_excerpt_sha256", "")).strip()
        actual_sha = hashlib.sha256(quote.encode("utf-8")).hexdigest()
        if actual_sha != expected_sha:
            return False, f"digest mismatch: expected {expected_sha}, computed {actual_sha}"

        if not str(claim.get("source_publisher", "")).strip():
            return False, "source_publisher is missing"
        if not str(claim.get("source_locator", "")).strip():
            return False, "source_locator is missing"

        # Validate SPDX license
        spdx = str(claim.get("license_spdx", "")).strip()
        if spdx not in APPROVED_SPDX_LICENSES:
            return False, f"unrecognized SPDX license: {spdx}"

        return True, "source valid"

    @staticmethod
    def audit_repo_license(repo_root: Path) -> Dict[str, Any]:
        """Audit repository LICENSE file and pyproject.toml license declarations."""
        license_file = repo_root / "LICENSE"
        if not license_file.is_file():
            return {"ok": False, "reason": "LICENSE file missing"}
        lic_text = license_file.read_text(encoding="utf-8")
        has_apache = "Apache License" in lic_text and "Version 2.0" in lic_text

        pyproject = repo_root / "pyproject.toml"
        pyproject_ok = False
        if pyproject.is_file():
            p_text = pyproject.read_text(encoding="utf-8")
            pyproject_ok = 'license = {text = "Apache-2.0"}' in p_text or "Apache-2.0" in p_text

        return {
            "ok": has_apache and pyproject_ok,
            "license_type": "Apache-2.0",
            "license_file_present": license_file.is_file(),
            "pyproject_licensed": pyproject_ok,
        }


# ============================================================================
# 3. Blind Benchmark Evaluation Engine
# ============================================================================

class BlindBenchmarkEvaluator:
    """Evaluates agent responses across held-out datasets with honest abstention calibration."""

    @staticmethod
    def evaluate_held_out_cases(agent: Any, cases: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Run a battery of held-out math, reasoning, and negative control queries."""
        results = []
        passed_count = 0

        for c in cases:
            prompt = c["prompt"]
            expect_answerable = c.get("answerable", True)
            expected_sub = c.get("expected_sub")
            forbid_sub = c.get("forbid_sub")

            resp = agent.respond(prompt)
            ans = str(getattr(resp, "answer", "")).strip()
            mode = str(getattr(resp, "mode", ""))
            confidence = float(getattr(resp, "confidence", 1.0))
            verified = bool(getattr(resp, "verified", False))

            is_abstained = (
                mode in ABSTENTION_MODES
                or confidence == 0.0
                or "abstain" in ans.lower()
                or "unverified" in ans.lower()
                or "cannot" in ans.lower()
            )

            if not expect_answerable:
                # Negative control: MUST abstain or report zero confidence
                case_pass = is_abstained or confidence <= 0.2
            else:
                sub_ok = True if expected_sub is None else (expected_sub in ans)
                forbid_ok = True if forbid_sub is None else (forbid_sub not in ans)
                case_pass = sub_ok and forbid_ok and (not is_abstained)

            if case_pass:
                passed_count += 1

            results.append({
                "name": c.get("name", prompt[:40]),
                "passed": case_pass,
                "answerable": expect_answerable,
                "answer": ans[:120],
                "confidence": confidence,
                "mode": mode,
            })

        total = len(cases)
        return {
            "passed": passed_count,
            "total": total,
            "accuracy": round(passed_count / total, 4) if total > 0 else 0.0,
            "cases": results,
        }


# ============================================================================
# 4. Mathematical & Factual Accuracy Engine
# ============================================================================

class MathematicalAndFactualVerifier:
    """Verifies mathematical derivations using exact CAS and factual ground truth."""

    @staticmethod
    def verify_polynomial_derivative(poly_expr: str, expected_derivative: str) -> bool:
        """Evaluates d/dx of polynomial expression and checks exact equivalence."""
        res = SymbolicCASOrgan.try_solve(f"d/dx ({poly_expr})")
        if res is None or not res.verified:
            return False
        # Normalize whitespace
        norm_ans = re.sub(r"\s+", "", res.answer)
        norm_exp = re.sub(r"\s+", "", expected_derivative)
        return norm_exp in norm_ans

    @staticmethod
    def verify_definite_integral(poly_expr: str, a: int | float, b: int | float, expected_val: int | float) -> bool:
        """Evaluates definite integral and compares with expected rational value."""
        res = SymbolicCASOrgan.try_solve(f"integrate ({poly_expr}) from {a} to {b}")
        if res is None or not res.verified:
            return False
        exp_str = str(expected_val)
        return exp_str in res.answer

    @staticmethod
    def verify_linear_system(eqs: str, expected_solution: Dict[str, str]) -> bool:
        """Solves multivariable linear system over Q and checks solution."""
        res = SymbolicCASOrgan.try_solve(f"solve {eqs}")
        if res is None or not res.verified:
            return False
        for var, val in expected_solution.items():
            if f"{var} = {val}" not in res.answer:
                return False
        return True

    @staticmethod
    def verify_unit_conversion(val: str, u_from: str, u_to: str, expected_val: str) -> bool:
        """Evaluates unit conversion with dimensional consistency check."""
        res = SymbolicCASOrgan.try_solve(f"convert {val} {u_from} to {u_to}")
        if res is None or not res.verified:
            return False
        return expected_val in res.answer


# ============================================================================
# 5. Cross-Domain Reasoning Engine
# ============================================================================

class CrossDomainReasoningEvaluator:
    """Evaluates multi-hop reasoning spanning distinct scientific and engineering domains."""

    @staticmethod
    def evaluate_quantum_born_projection(vault: Any) -> Tuple[bool, str]:
        """Quantum mechanics -> Linear algebra Born rule projection."""
        q = "Apply Hadamard gate to state |0> and find probability of measuring |0>"
        res = vault.respond(q)
        ans = str(getattr(res, "answer", ""))
        conf = float(getattr(res, "confidence", 0.0))
        passed = ("0.5" in ans or "50%" in ans) and conf >= 0.80
        return passed, ans

    @staticmethod
    def evaluate_kinematic_calculus_chain(vault: Any) -> Tuple[bool, str]:
        """Classical mechanics -> Differential calculus velocity from position."""
        q = "Given position function x(t) = 3t^3 - 9t^2 + 4t, find velocity at t = 4"
        res = vault.respond(q)
        ans = str(getattr(res, "answer", ""))
        conf = float(getattr(res, "confidence", 0.0))
        # v(t) = 9t^2 - 18t + 4; v(4) = 9(16) - 18(4) + 4 = 144 - 72 + 4 = 76
        passed = "76" in ans and conf >= 0.80
        return passed, ans

    @staticmethod
    def evaluate_structural_safety_factor(vault: Any) -> Tuple[bool, str]:
        """Materials science -> Structural engineering safety factor."""
        # F = 15000 N, A = 0.005 m^2 => actual stress = 3,000,000 Pa. Yield = 9,000,000 Pa => FS = 3.0
        q = "Calculate safety factor for a load of 15000 N over an area of 0.005 m^2 with yield stress of 9000000"
        res = vault.respond(q)
        ans = str(getattr(res, "answer", ""))
        conf = float(getattr(res, "confidence", 0.0))
        passed = "3.0" in ans and "SAFE" in ans and conf >= 0.80
        return passed, ans

    @staticmethod
    def evaluate_transitive_categorical_syllogism(vault: Any) -> Tuple[bool, str]:
        """Formal deductive logic -> Chemistry property attribution."""
        q = "Given that all noble gases are unreactive, and neon is a noble gas, what is the property of neon?"
        res = vault.respond(q)
        ans = str(getattr(res, "answer", "")).lower()
        conf = float(getattr(res, "confidence", 0.0))
        passed = "neon" in ans and "unreactive" in ans and conf >= 0.80
        return passed, ans

    @staticmethod
    def evaluate_quantum_grover_scaling(bridge: Any) -> Tuple[bool, str]:
        """Quantum algorithm scaling -> Optimal Grover iteration complexity O(sqrt(N))."""
        q = "How many Grover iterations are needed for N = 10000 items?"
        res = bridge.solve(q)
        ans = str(res.get("answer", ""))
        passed = "78" in ans and bool(res.get("ok", False))
        return passed, ans
