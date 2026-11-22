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
        universal_intelligence_provider: Callable[[], Any] | None = None,
        uol_router_provider: Callable[[], Any] | None = None,
    ) -> None:
        if not 0.0 <= math_confidence_threshold <= 1.0:
            raise ValueError("math_confidence_threshold must be between 0 and 1")
        self._vault_provider = vault_provider
        self._memory_provider = memory_provider
        self._reasoning_provider = reasoning_provider
        self._code_provider = code_provider
        self._code_generation_provider = code_generation_provider
        self._synthesis_provider = synthesis_provider
        self._universal_intelligence_provider = universal_intelligence_provider
        self._uol_router_provider = uol_router_provider
        self._default_uol_router: Any | None = None
        self.metrics = metrics
        self.math_confidence_threshold = math_confidence_threshold
        self.synthesizer = EvidenceSynthesisEngine()

        # Native CPU intelligence organs
        from mirror_crystal.native_intelligence import (
            PersonalFactMemory,
            ConversationOrgan,
            LogicOrgan,
            WordProblemSolver,
            CuratedKnowledgeOrgan,
            RetrievalGuard,
            KnowledgeGraphEngine,
            MultiHopChainingEngine,
            MicroEmbeddingEngine,
            DocumentAutoCrystallizerEngine,
            ContextReasoner,
        )
        self.fact_memory = PersonalFactMemory()
        self.conversation = ConversationOrgan()
        self.logic = LogicOrgan()
        self.word_problems = WordProblemSolver()
        self.curated_knowledge = CuratedKnowledgeOrgan()
        self.retrieval_guard = RetrievalGuard
        self.knowledge_graph = KnowledgeGraphEngine()
        self.multi_hop = MultiHopChainingEngine(kg=self.knowledge_graph)
        self.semantic_search = MicroEmbeddingEngine()
        self.auto_crystallizer = DocumentAutoCrystallizerEngine(kg=self.knowledge_graph)
        self.context_reasoner = ContextReasoner()

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
    def _is_unified_13stage_command(prompt: str) -> bool:
        low = prompt.strip().casefold()
        return low.startswith(
            (
                "synthesize architecture:",
                "architect app:",
                "synthesize narrative:",
                "canon scene:",
                "analyze subtext:",
                "pragmatic meaning:",
                "do-calculus:",
                "intervene do(",
            )
        )

    @staticmethod
    def _is_code_request(prompt: str) -> bool:
        if UnifiedCognitiveOrchestrator._is_unified_13stage_command(prompt):
            return False
        from mirror_crystal.byte_intent_router import get_default_byte_intent_router

        return get_default_byte_intent_router().is_code_request(prompt)

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
        universal_options: Mapping[str, Any] | None = None,
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
        universal_report = None

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
                    rule_id = getattr(vault_result, "matched_rule_id", None)
                    # Filter spurious vault matches that have zero or low topical overlap
                    if not is_verified and rule_id is not None:
                        is_rel, _, _ = self.retrieval_guard.is_relevant(prompt, prompt, str(answer))
                        if not is_rel:
                            answer = None
                    if answer:
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

        if self._universal_intelligence_provider is not None:
            try:
                universal_pipeline = self._universal_intelligence_provider()
                if universal_pipeline is not None:
                    options = dict(universal_options or {})
                    options.setdefault("execute", True)
                    universal_report = universal_pipeline.analyze(prompt, **options)
                    sources_consulted.append("universal_intelligence")
            except Exception as exc:
                errors.append({"source": "universal_intelligence", "error": type(exc).__name__})

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

        generation_metadata: dict[str, Any] = {}
        generation_needs_input = None
        if candidate_code is None and self._is_code_request(prompt):
            if self._code_generation_provider is not None:
                try:
                    sources_consulted.append("code_generation")
                    generated = self._code_generation_provider(prompt)
                    if isinstance(generated, Mapping):
                        generation_metadata = dict(generated)
                        generated_code = generated.get("code") or generated.get("final_code")
                    else:
                        generated_code = generated
                    if generated_code is not None:
                        generated_code = str(generated_code).strip()
                    generation_needs_input = generation_metadata.get("needs_input")
                    code_generation = {
                        "available": True,
                        "generated": bool(generated_code),
                        "generation_kind": generation_metadata.get("generation_kind"),
                    }
                    if generation_needs_input:
                        code_generation["needs_input"] = str(generation_needs_input)
                    if not generated_code and generation_metadata.get("reason"):
                        code_generation["reason"] = str(generation_metadata["reason"])
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
                    spec_desc=spec_desc or str(generation_metadata.get("spec_desc") or ""),
                    explicit_tests=explicit_tests or str(generation_metadata.get("tests") or ""),
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
        universal_execution = (
            universal_report.get("execution", {})
            if isinstance(universal_report, Mapping) else {}
        )
        task_execution_verified = bool(
            isinstance(universal_execution, Mapping)
            and universal_execution.get("status") == "succeeded"
            and universal_execution.get("verified") is True
            and universal_execution.get("verification", {}).get("verified") is True
        )
        task_code_verified = bool(
            generated_code
            and code_verification
            and code_verification.get("verified") is True
            and int(code_verification.get("tests_generated", 0) or 0) > 0
            and int(code_verification.get("tests_passed", 0) or 0)
                == int(code_verification.get("tests_generated", 0) or 0)
        )
        uol_route_payload: dict[str, Any] | None = None
        task_uol_verified = False
        if not (task_code_verified or task_execution_verified or verified_answer) and (
            not self._is_code_request(prompt) or "def " in prompt
        ):
            try:
                router = self._uol_router_provider() if self._uol_router_provider is not None else self._default_uol_router
                if router is None:
                    from mirror_crystal.nl_uol_router import NaturalLanguageUOLRouter
                    router = NaturalLanguageUOLRouter()
                    self._default_uol_router = router
                routed = router.route(prompt, session_id=session_id)
                if routed.matched:
                    task_uol_verified = True
                    sources_consulted.append("uol_auto_router")
                    uol_route_payload = {
                        "matched": True,
                        "rule_ids": list(routed.rule_ids),
                        "inputs": dict(routed.inputs),
                        "output": routed.output,
                        "unit": routed.unit,
                        "explanation": routed.explanation,
                        "steps": list(routed.steps),
                        "realizer_engine": "mirror-native-hybrid-realizer-v1",
                        "uses_third_party_model": False,
                    }
            except Exception as exc:
                errors.append(f"uol_router:{type(exc).__name__}")

        task_result_verified = task_execution_verified or task_code_verified or task_uol_verified
        clarification_required = bool(generation_needs_input)
        abstained = not (verified_answer or task_result_verified or clarification_required)
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

        user_name = self.fact_memory.get(session_id, "name")
        native_hit = None
        fallback_hit = None
        if not evidence:
            fallback_hit = self.knowledge_graph.lookup_from_prompt(prompt) or self.curated_knowledge.lookup(prompt)

        if candidate_code is None and not self._is_code_request(prompt) and not self._is_math_query(prompt):
            native_hit = (
                self.fact_memory.try_store(session_id, prompt)
                or self.fact_memory.try_recall(session_id, prompt)
                or self.conversation.respond(prompt, session_id=session_id, user_name=user_name)
                or self.multi_hop.resolve_chain(prompt)
                or self.context_reasoner.respond(prompt)
                or self.logic.respond(prompt)
                or self.word_problems.solve(prompt)
                or fallback_hit
            )
        elif candidate_code is None and not self._is_code_request(prompt):
            # For queries flagged with math keywords that might be word problems or curated definitions
            native_hit = (
                self.word_problems.solve(prompt)
                or self.context_reasoner.respond(prompt)
                or fallback_hit
            )

        if native_hit is not None and not task_uol_verified:
            return {
                "answer": native_hit.answer,
                "answer_qualification": native_hit.qualification,
                "confidence": native_hit.confidence,
                "mode": f"native_{native_hit.kind}",
                "epistemic_tier": "verified_native_result" if native_hit.verified else "conversational_context",
                "verified": native_hit.verified,
                "task_result_verified": False,
                "task_code_verified": False,
                "task_execution_verified": False,
                "abstained": False,
                "verification_status": "verified" if native_hit.verified else "contextual_response",
                "candidate_answer": native_hit.answer,
                "candidate_source": f"native_{native_hit.kind}",
                "memory_context": memory_context,
                "memory_used_as_evidence": False,
                "sources_consulted": list(dict.fromkeys(sources_consulted + [f"native_{native_hit.kind}"])),
                "evidence": evidence,
                "computed_candidates": computed_candidates,
                "synthesis": synthesis,
                "generated_code": generated_code,
                "code_generation": code_generation,
                "code_verification": code_verification,
                "universal_intelligence": universal_report,
                "task_execution": universal_execution,
                "conflict": False,
                "errors": errors,
                "can_teach": False,
                "teach_template": None,
                "consent_log_requested": bool(consent_log),
                "steps": list(native_hit.steps),
            }

        if task_code_verified:
            final_code = str(code_verification.get("final_code") or generated_code)
            final_answer = (
                "Generated and tested Python code:\n\n"
                + (chr(96) * 3) + "python\n" + final_code + "\n" + (chr(96) * 3)
            )
            qualification = "verified_task_result"
        elif task_execution_verified:
            final_answer = (
                "The requested task was completed and independently verified. "
                "Verified outputs: " + repr(universal_execution.get("verified_outputs", {}))
            )
            qualification = "verified_task_result"
        elif task_uol_verified and uol_route_payload is not None:
            final_answer = str(uol_route_payload["explanation"])
            qualification = "verified_task_result"
        elif clarification_required:
            final_answer = str(generation_needs_input)
            qualification = "clarification_required"
        elif conflict:
            final_answer = "Conflicting verified claims; Mirror abstains."
            qualification = "conflicting_evidence"
        elif verified_answer:
            final_answer = synthesis["answer"]
            qualification = "verified"
        elif code_generation and code_generation.get("available") and not code_generation.get("generated"):
            final_answer = str(
                code_generation.get("reason")
                or "Mirror's native code writer does not support this request yet."
            )
            qualification = "unsupported_task"
        elif (
            generated_code
            and isinstance(code_verification, Mapping)
            and any(
                "Docker or Podman is required" in str(diag)
                for diag in (code_verification.get("diagnostics") or [])
            )
        ):
            final_answer = (
                "Code was synthesized, but Mirror's Zero-Trust container sandbox could not "
                "execute the verification tests because Docker or Podman is not running. "
                "Start Docker/Podman for container isolation, or set "
                "MIRROR_SANDBOX_MODE=process for local development."
            )
            qualification = "sandbox_unavailable"
        elif candidate:
            final_answer = (
                "UNVERIFIED CANDIDATE — not independently verified.\n\n"
                + candidate["answer"]
            )
            qualification = "unverified_candidate"
        else:
            final_answer = "Insufficient independently verified evidence."
            qualification = "no_candidate"

        return {
            "answer": final_answer,
            "answer_qualification": qualification,
            "confidence": 0.0 if abstained else min(
                (item["confidence"] for item in evidence if item["verified"]),
                default=0.0,
            ),
            "mode": "unified_task_and_claim_synthesis",
            "epistemic_tier": (
                "conflicting_verified_evidence" if conflict else (
                    "verified_claim_synthesis" if verified_answer else (
                        "verified_task_result" if task_result_verified else "honest_abstain"
                    )
                )
            ),
            "verified": verified_answer,
            "task_result_verified": task_result_verified,
            "task_code_verified": task_code_verified,
            "task_execution_verified": task_execution_verified,
            "abstained": abstained,
            "verification_status": (
                "verified" if verified_answer else (
                    "task_verified" if task_result_verified else (
                        "clarification_required" if clarification_required else "not_independently_verified"
                    )
                )
            ),
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
            "universal_intelligence": universal_report,
            "uol_route": uol_route_payload,
            "task_execution": universal_execution,
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
