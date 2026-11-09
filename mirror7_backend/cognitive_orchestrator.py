"""Unified evidence orchestration across Mirror's existing cognitive organs.

This module coordinates existing components; it does not pretend that a source
was consulted when its dependency is unavailable, and it abstains on conflicts.
"""
from __future__ import annotations

from typing import Any, Callable


class UnifiedCognitiveOrchestrator:
    """Collect and rank memory, Crystal knowledge, math, and code-verification evidence."""

    def __init__(
        self,
        *,
        vault_provider: Callable[[], Any | None],
        memory_provider: Callable[[], Any],
        reasoning_provider: Callable[[], Any],
        code_provider: Callable[[], Any],
        metrics: Any | None = None,
        math_confidence_threshold: float = 0.80,
    ) -> None:
        if not 0.0 <= math_confidence_threshold <= 1.0:
            raise ValueError("math_confidence_threshold must be between 0 and 1")
        self._vault_provider = vault_provider
        self._memory_provider = memory_provider
        self._reasoning_provider = reasoning_provider
        self._code_provider = code_provider
        self.metrics = metrics
        self.math_confidence_threshold = math_confidence_threshold

    @staticmethod
    def _is_math_query(prompt: str) -> bool:
        text = prompt.lower()
        markers = (
            "calculate", "compute", "equation", "integral", "derivative",
            "eigenvalue", "probability", "solve", "plus", "minus", "times",
            "divided by", "math", "matrix", "how many",
        )
        return any(marker in text for marker in markers)

    @staticmethod
    def _candidate(source: str, answer: Any, confidence: float, tier: str,
                   verified: bool, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        return {
            "source": source,
            "answer": str(answer),
            "confidence": max(0.0, min(1.0, float(confidence))),
            "epistemic_tier": tier,
            "verified": bool(verified),
            "metadata": metadata or {},
        }

    def ask(
        self,
        prompt: str,
        *,
        session_id: str = "default",
        candidate_code: str | None = None,
        spec_desc: str = "",
        explicit_tests: str = "",
        consent_log: bool = False,
    ) -> dict[str, Any]:
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a non-empty string")

        evidence: list[dict[str, Any]] = []
        errors: list[dict[str, str]] = []
        memory = self._memory_provider()
        vault = self._vault_provider()

        # Gather memory and knowledge independently instead of short-circuiting.
        try:
            memory_context = memory.resolve_memory_context(session_id, prompt)
            if memory_context:
                evidence.append(self._candidate(
                    "cross_session_memory", memory_context, 1.0,
                    "remembered_context", False,
                ))
            memory.auto_extract_and_remember(session_id, prompt)
        except Exception as exc:
            errors.append({"source": "memory", "error": type(exc).__name__})

        vault_result = None
        if vault is not None:
            try:
                vault_result = vault.respond(prompt, session_id=session_id)
                tier = str(getattr(vault_result, "epistemic_tier", "unknown"))
                answer = getattr(vault_result, "answer", None)
                if answer and tier != "honest_abstain":
                    evidence.append(self._candidate(
                        "crystal_knowledge", answer,
                        getattr(vault_result, "confidence", 0.0),
                        tier,
                        tier == "verified_exact" and bool(
                            getattr(vault_result, "sandbox_verified", False)
                        ),
                        {"matched_rule_id": getattr(vault_result, "matched_rule_id", None)},
                    ))
            except Exception as exc:
                errors.append({"source": "crystal_knowledge", "error": type(exc).__name__})

        math_result = None
        if self._is_math_query(prompt) or vault_result is None or getattr(
            vault_result, "epistemic_tier", None
        ) == "honest_abstain":
            try:
                math_result = self._reasoning_provider().solve(prompt)
                reward = float(math_result.get("prm_reward", 0.0))
                if math_result.get("ok") and reward >= self.math_confidence_threshold:
                    answer = (
                        math_result.get("formatted_solution")
                        or math_result.get("formatted_response")
                        or math_result.get("answer")
                    )
                    if answer:
                        evidence.append(self._candidate(
                            "cerebrum_reasoning", answer, reward, "verified_exact", True,
                            {"strategy": math_result.get("strategy")},
                        ))
            except Exception as exc:
                errors.append({"source": "reasoning", "error": type(exc).__name__})

        code_verification = None
        if candidate_code is not None:
            try:
                code_verification = self._code_provider().run_tdd_cycle(
                    candidate_code,
                    spec_desc=spec_desc,
                    explicit_tests=explicit_tests,
                )
                if hasattr(code_verification, "__dataclass_fields__"):
                    from dataclasses import asdict
                    code_verification = asdict(code_verification)
                elif not isinstance(code_verification, dict):
                    code_verification = {"result": str(code_verification)}
                # Code checks are reported separately; they never masquerade as
                # proof of the natural-language answer.
                code_verification = dict(code_verification)
            except Exception as exc:
                errors.append({"source": "code_verification", "error": type(exc).__name__})
                code_verification = {"verified": False, "error": type(exc).__name__}

        # A code result is verified only when the TDD engine explicitly reports
        # a passing test run. Missing/ambiguous status is not treated as success.
        if code_verification is not None:
            passed = any(code_verification.get(key) is True for key in (
                "passed", "success", "tests_passed", "all_tests_passed"
            ))
            code_verification["verified"] = passed

        verified = [item for item in evidence if item["verified"]]
        distinct_answers = {item["answer"].strip() for item in verified}
        conflict = len(distinct_answers) > 1
        selected = None
        if verified and not conflict:
            selected = max(verified, key=lambda item: item["confidence"])
        elif not conflict and evidence:
            selected = max(evidence, key=lambda item: item["confidence"])

        abstained = selected is None or conflict
        if self.metrics is not None:
            try:
                self.metrics.increment("crystal_queries_total")
                self.metrics.increment(
                    "crystal_queries_abstained" if abstained else "crystal_queries_answered"
                )
            except Exception:
                pass

        return {
            "answer": (
                "Conflicting verified sources; Mirror abstains."
                if conflict else (selected["answer"] if selected else "Insufficient verified evidence.")
            ),
            "confidence": selected["confidence"] if selected and not conflict else 0.0,
            "mode": "unified_evidence_orchestration",
            "epistemic_tier": (
                "conflicting_verified_evidence" if conflict
                else (selected["epistemic_tier"] if selected else "honest_abstain")
            ),
            "verified": bool(selected and selected["verified"] and not conflict),
            "abstained": abstained,
            "sources_consulted": [
                name for name, used in (
                    ("memory", memory is not None),
                    ("crystal_knowledge", vault is not None),
                    ("reasoning", math_result is not None),
                    ("code_verification", candidate_code is not None),
                ) if used
            ],
            "evidence": evidence,
            "code_verification": code_verification,
            "conflict": conflict,
            "errors": errors,
            "can_teach": abstained,
            "teach_template": ({
                "question": prompt,
                "domain": "general",
                "suggested_action": "Teach Mirror to eliminate this abstention permanently",
            } if abstained else None),
            "consent_log_requested": bool(consent_log),
        }
