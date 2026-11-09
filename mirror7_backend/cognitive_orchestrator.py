"""Unified, trust-aware orchestration across Mirror's cognitive components.

Memory is context, never factual proof. Math outputs remain computed candidates
unless an explicit independent verification certificate is attached. Verified
claims are synthesized before a final answer is returned.
"""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
import re
from typing import Any, Callable, Mapping

from .evidence_synthesis import EvidenceSynthesisEngine


class UnifiedCognitiveOrchestrator:
    """Route requests, gather evidence, synthesize verified claims, and abstain safely."""

    def __init__(
        self,
        *,
        vault_provider: Callable[[], Any | None],
        memory_provider: Callable[[], Any],
        reasoning_provider: Callable[[], Any],
        code_provider: Callable[[], Any],
        metrics: Any | None = None,
        math_confidence_threshold: float = 0.80,
        code_generation_provider: Callable[[str], Any] | None = None,
        synthesis_provider: Callable[[dict[str, Any]], Any] | None = None,
    ) -> None:
        if not 0.0 <= math_confidence_threshold <= 1.0:
            raise ValueError("math_confidence_threshold must be between 0 and 1")
        self._vault_provider = vault_provider
        self._memory_provider = memory_provider
        self._reasoning_provider = reasoning_provider
        self._code_provider = code_provider
        self._code_generation_provider = code_generation_provider
        self._synthesis_provider = synthesis_provider
        self.metrics = metrics
        self.math_confidence_threshold = math_confidence_threshold
        self.synthesizer = EvidenceSynthesisEngine()

    @staticmethod
    def _is_math_query(prompt: str) -> bool:
        text = prompt.casefold()
        math_terms = (
            "calculate", "compute", "equation", "integral", "derivative",
            "eigenvalue", "probability", "solve", "mathematics", "math",
            "matrix", "modulo", "remainder", "differentiate", "integrate",
            "grover", "quantum", "iteration", "iterations",
        )
        return (
            any(re.search(r"\b" + re.escape(term) + r"\b", text) for term in math_terms)
            or bool(re.search(r"\d\s*(?:\+|\-|\*|/|\^|%)\s*\d", text))
        )

    @staticmethod
    def _is_code_request(prompt: str) -> bool:
        text = prompt.casefold()
        markers = (
            "write code", "generate code", "implement", "build a function",
            "write a function", "create a class", "write a script",
            "program this", "code this", "fix this code", "implement in python",
            "implement in javascript", "create an api", "build an api",
        )
        return any(marker in text for marker in markers)

    @staticmethod
    def _should_extract_memory(prompt: str) -> bool:
        text = prompt.strip().casefold()
        if not text or "?" in text:
            return False
        explicit_subject = text.startswith((
            "my ", "our ", "the project ", "we use ", "i use ",
            "app ", "api ", "web ", "backend ", "server ", "database ", "db ",
        ))
        configuration_statement = bool(re.search(
            r"\b(port|bucket|database|framework|region|server|configured|running)\b", text
        ))
        return explicit_subject and configuration_statement

    @staticmethod
    def _candidate(
        source: str,
        answer: Any,
        confidence: float,
        tier: str,
        verified: bool,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        try:
            score = float(confidence)
        except (TypeError, ValueError):
            score = 0.0
        return {
            "source": source,
            "answer": str(answer),
            "confidence": max(0.0, min(1.0, score)),
            "epistemic_tier": tier,
            "verified": bool(verified),
            "metadata": metadata or {},
        }

    @staticmethod
    def _is_verified_crystal_result(result: Any) -> bool:
        tier = str(getattr(result, "epistemic_tier", "unknown"))
        mode = str(getattr(result, "mode", ""))
        matched_rule_id = getattr(result, "matched_rule_id", None)
        if mode == "blueprint_composition_reliable" and tier == "high_confidence":
            # This mode is returned only after source/provenance and claim-review gates.
            return True
        # A sandbox pass is execution evidence, not general factual proof. Restrict
        # the exact fast path to direct deterministic operations without legacy rules.
        return (
            tier == "verified_exact"
            and bool(getattr(result, "sandbox_verified", False))
            and matched_rule_id is None
            and mode != "session_constraint_lock"
        )

    @staticmethod
    def _code_report_dict(report: Any) -> dict[str, Any]:
        if is_dataclass(report):
            value = asdict(report)
        elif isinstance(report, Mapping):
            value = dict(report)
        else:
            value = {"result": str(report)}
        generated = int(value.get("tests_generated", 0) or 0)
        passed = int(value.get("tests_passed", 0) or 0)
        strict_pass = (
            value.get("verified") is True
            and generated > 0
            and passed == generated
        )
        value["verified"] = strict_pass
        value["tests_generated"] = generated
        value["tests_passed"] = passed
        return value

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
        sources_consulted: list[str] = []
        memory_context = None
        computed_candidates: list[dict[str, Any]] = []
        generated_code = None
        code_generation = None
        code_verification = None

        try:
            memory = self._memory_provider()
            sources_consulted.append("memory")
            memory_context = memory.resolve_memory_context(session_id, prompt)
            if self._should_extract_memory(prompt):
                memory.auto_extract_and_remember(session_id, prompt)
        except Exception as exc:
            memory = None
            errors.append({"source": "memory", "error": type(exc).__name__})

        try:
            vault = self._vault_provider()
            if vault is not None:
                sources_consulted.append("crystal_knowledge")
                vault_result = vault.respond(prompt, session_id=session_id)
                tier = str(getattr(vault_result, "epistemic_tier", "unknown"))
                answer = getattr(vault_result, "answer", None)
                if answer and tier != "honest_abstain":
                    is_verified = self._is_verified_crystal_result(vault_result)
                    evidence.append(self._candidate(
                        "crystal_knowledge",
                        answer,
                        getattr(vault_result, "confidence", 0.0),
                        "verified_claims" if is_verified else "unverified_knowledge",
                        is_verified,
                        {
                            "mode": getattr(vault_result, "mode", None),
                            "matched_rule_id": getattr(vault_result, "matched_rule_id", None),
                            "reasoning_trace": list(getattr(vault_result, "reasoning_trace", []) or []),
                        },
                    ))
        except Exception as exc:
            errors.append({"source": "crystal_knowledge", "error": type(exc).__name__})

        if self._is_math_query(prompt):
            try:
                sources_consulted.append("reasoning")
                math_result = self._reasoning_provider().solve(prompt)
                reward = float(math_result.get("prm_reward", 0.0))
                answer = (
                    math_result.get("formatted_solution")
                    or math_result.get("formatted_response")
                    or math_result.get("answer")
                )
                valid_steps = math_result.get("prm_steps") or []
                structurally_consistent = bool(valid_steps) and all(
                    isinstance(step, Mapping) and step.get("is_valid") is True
                    for step in valid_steps
                )
                if math_result.get("ok") and answer and reward >= self.math_confidence_threshold:
                    computed_candidates.append(self._candidate(
                        "cerebrum_reasoning",
                        answer,
                        reward,
                        "computed_candidate",
                        False,
                        {
                            "strategy": math_result.get("strategy"),
                            "prm_reward": reward,
                            "structural_steps_valid": structurally_consistent,
                            "verification_note": (
                                "Symbolic/PRM result; not independently certified by a second verifier."
                            ),
                        },
                    ))
            except Exception as exc:
                errors.append({"source": "reasoning", "error": type(exc).__name__})

        should_generate_code = candidate_code is None and self._is_code_request(prompt)
        if should_generate_code:
            if self._code_generation_provider is not None:
                try:
                    sources_consulted.append("code_generation")
                    generated = self._code_generation_provider(prompt)
                    if isinstance(generated, Mapping):
                        generated_code = generated.get("code") or generated.get("final_code")
                    else:
                        generated_code = generated
                    if generated_code is not None:
                        generated_code = str(generated_code).strip()
                    code_generation = {"available": True, "generated": bool(generated_code)}
                except Exception as exc:
                    errors.append({"source": "code_generation", "error": type(exc).__name__})
                    code_generation = {"available": True, "generated": False}
            else:
                code_generation = {"available": False, "generated": False}

        code_to_test = candidate_code or generated_code
        if code_to_test:
            try:
                sources_consulted.append("code_verification")
                raw_report = self._code_provider().run_tdd_cycle(
                    str(code_to_test),
                    spec_desc=spec_desc,
                    explicit_tests=explicit_tests,
                )
                code_verification = self._code_report_dict(raw_report)
            except Exception as exc:
                errors.append({"source": "code_verification", "error": type(exc).__name__})
                code_verification = {"verified": False, "error": type(exc).__name__}

        synthesis = self.synthesizer.synthesize(
            prompt,
            evidence,
            memory_context=memory_context,
            generator=self._synthesis_provider,
        )
        conflict = bool(synthesis["conflict"])
        verified_answer = bool(synthesis["verified"] and synthesis["answer"] and not conflict)
        abstained = not verified_answer
        candidate = max(
            evidence + computed_candidates,
            key=lambda item: item["confidence"],
            default=None,
        )

        if self.metrics is not None:
            try:
                self.metrics.increment("crystal_queries_total")
                self.metrics.increment(
                    "crystal_queries_abstained" if abstained else "crystal_queries_answered"
                )
                self.metrics.increment(
                    "cognitive_synthesis_conflicts" if conflict else "cognitive_synthesis_completed"
                )
            except Exception:
                pass

        return {
            "answer": (
                "Conflicting verified claims; Mirror abstains."
                if conflict else (
                    synthesis["answer"] if verified_answer
                    else (
                        "UNVERIFIED CANDIDATE — not independently verified.\\n\\n"
                        + candidate["answer"]
                        if candidate else "Insufficient independently verified evidence."
                    )
                )
            ),
            "answer_qualification": (
                "verified" if verified_answer else (
                    "conflicting_evidence" if conflict else (
                        "unverified_candidate" if candidate else "no_candidate"
                    )
                )
            ),
            "confidence": 0.0 if abstained else min(
                (item["confidence"] for item in evidence if item["verified"]),
                default=0.0,
            ),
            "mode": "unified_claim_synthesis",
            "epistemic_tier": "conflicting_verified_evidence" if conflict else (
                "verified_claim_synthesis" if verified_answer else "honest_abstain"
            ),
            "verified": verified_answer,
            "abstained": abstained,
            "verification_status": "verified" if verified_answer else "not_independently_verified",
            "candidate_answer": candidate["answer"] if candidate else None,
            "candidate_source": candidate["source"] if candidate else None,
            "memory_context": memory_context,
            "memory_used_as_evidence": False,
            "sources_consulted": list(dict.fromkeys(sources_consulted)),
            "evidence": evidence,
            "computed_candidates": computed_candidates,
            "synthesis": synthesis,
            "generated_code": generated_code,
            "code_generation": code_generation,
            "code_verification": code_verification,
            "conflict": conflict,
            "errors": errors,
            "can_teach": abstained,
            "teach_template": ({
                "question": prompt,
                "domain": "general",
                "suggested_action": "Submit reviewed source evidence or a tested rule for this unanswered question",
            } if abstained else None),
            "consent_log_requested": bool(consent_log),
        }
