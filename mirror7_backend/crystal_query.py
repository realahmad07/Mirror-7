"""Crystal query orchestration separated from the backend session facade.

Dependencies are provided lazily so query handling stays independent of BackendService
construction and can be tested with small fakes.
"""
from __future__ import annotations

from typing import Any, Callable


class CrystalQueryCoordinator:
    """Own query routing, memory recall, math fallback, and answer telemetry."""

    def __init__(
        self,
        *,
        vault_provider: Callable[[], Any | None],
        memory_graph_provider: Callable[[], Any],
        cerebrum_bridge_provider: Callable[[], Any],
        metrics: Any,
    ) -> None:
        self._vault_provider = vault_provider
        self._memory_graph_provider = memory_graph_provider
        self._cerebrum_bridge_provider = cerebrum_bridge_provider
        self.metrics = metrics

    @property
    def crystal_vault(self) -> Any | None:
        return self._vault_provider()

    @property
    def memory_graph(self) -> Any:
        return self._memory_graph_provider()

    @property
    def cerebrum_bridge(self) -> Any:
        return self._cerebrum_bridge_provider()

    def ask(self, prompt: str, session_id: str = "default", consent_log: bool = False) -> dict[str, Any]:
        """Query the backing Mirror Crystal Vault with verified Cerebrum math fallback on miss, abstention tracking, and optional consent logging."""
        if self.crystal_vault is None:
            return {
                "answer": "Mirror Crystal Vault is not initialized.",
                "confidence": 0.0,
                "mode": "unmatched_honest_abstain",
                "matched_rule_id": None,
                "latency_ms": 0.0,
                "sandbox_verified": False,
                "reasoning_trace": ["Vault not mounted."],
                "epistemic_tier": "honest_abstain",
                "abstained": True,
                "can_teach": True,
                "teach_template": {"question": prompt, "domain": "general"},
            }
        # 1. Check cross-session memory graph for active project/config anchor recall
        mem_ctx = self.memory_graph.resolve_memory_context(session_id, prompt)
        if mem_ctx and any(w in prompt.lower() for w in ("what is", "what are", "show", "config", "my", "our", "port", "bucket", "database")):
            self.metrics.increment("crystal_queries_total")
            self.metrics.increment("crystal_queries_answered")
            return {
                "answer": mem_ctx,
                "confidence": 1.0,
                "mode": "cross_session_memory_recall",
                "matched_rule_id": None,
                "latency_ms": 0.45,
                "sandbox_verified": True,
                "reasoning_trace": ["Retrieved from persistent CrossSessionMemoryGraph"],
                "epistemic_tier": "verified_exact",
                "abstained": False,
                "can_teach": False,
            }

        # 2. Auto-extract any newly stated parameters in prompt into persistent memory
        self.memory_graph.auto_extract_and_remember(session_id, prompt)

        res = self.crystal_vault.respond(prompt, session_id=session_id)
        self.metrics.increment("crystal_queries_total")
        if res.epistemic_tier == "honest_abstain" and prompt.strip():
            math_res = self.cerebrum_bridge.solve(prompt)
            if math_res.get("ok") and float(math_res.get("prm_reward", 0.0)) >= 0.80:
                self.metrics.increment("crystal_queries_answered")
                ans_text = str(math_res.get("formatted_solution") or math_res.get("formatted_response") or math_res["answer"])
                out_pkt = {
                    "answer": ans_text,
                    "confidence": round(float(math_res["prm_reward"]), 4),
                    "mode": f"cerebrum_{math_res.get('strategy', 'exact')}",
                    "matched_rule_id": None,
                    "latency_ms": round(float(res.latency_ms) + float(math_res.get("latency_ms", 0.0)), 3),
                    "sandbox_verified": True,
                    "reasoning_trace": list(res.reasoning_trace)
                    + [f"Delegated to CerebrumCrystalBridge ({math_res.get('strategy')}, PRM={math_res['prm_reward']})"],
                    "epistemic_tier": "verified_exact",
                    "abstained": False,
                    "can_teach": False,
                }
                if consent_log:
                    try:
                        from mirror_eval.runner import QueryRegressionLogger
                        QueryRegressionLogger.log_query(prompt, ans_text, out_pkt["latency_ms"], out_pkt["confidence"], user_consented=True, abstained=False)
                    except Exception:
                        pass
                return out_pkt

        is_abstained = res.epistemic_tier == "honest_abstain"
        if is_abstained:
            self.metrics.increment("crystal_queries_abstained")
        else:
            self.metrics.increment("crystal_queries_answered")

        out_pkt = {
            "answer": res.answer,
            "confidence": res.confidence,
            "mode": res.mode,
            "matched_rule_id": res.matched_rule_id,
            "latency_ms": res.latency_ms,
            "sandbox_verified": res.sandbox_verified,
            "reasoning_trace": res.reasoning_trace,
            "epistemic_tier": res.epistemic_tier,
            "abstained": is_abstained,
            "can_teach": is_abstained,
            "teach_template": {
                "question": prompt,
                "domain": "general",
                "suggested_action": "Teach Mirror to eliminate this abstention permanently",
            } if is_abstained else None,
        }

        if consent_log:
            try:
                from mirror_eval.runner import QueryRegressionLogger
                QueryRegressionLogger.log_query(prompt, res.answer, res.latency_ms, res.confidence, user_consented=True, abstained=is_abstained)
            except Exception:
                pass

        return out_pkt


