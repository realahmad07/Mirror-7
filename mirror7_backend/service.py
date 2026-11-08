from __future__ import annotations

from dataclasses import replace
from threading import Condition, RLock
from typing import Any, Callable, Mapping

from .actions import ActionGateway
from .observability import BackendMetrics
from .persistence import CheckpointStore
from .runtime import BackendResult, BackendSession
from phase359_response_generation_adapter import make_response_generator
from phase362_response_generation_config import ResponseGenerationConfig
from phase364_response_generation_validation import validate_generated_response
from phase357_response_generation_interface import (
    build_response_generation_request,
    generate_response,
)


class BackendService:
    """UI-independent orchestration facade for the Mirror 7 backend with optional Mirror Crystal Engine."""

    VERSION = 5

    def __init__(
        self,
        *,
        engine_factory: Callable[[], Any] | None = None,
        checkpoint_store: CheckpointStore | None = None,
        action_gateway: ActionGateway | None = None,
        response_generator: Callable[[Any], Any] | None = None,
        generation_config: ResponseGenerationConfig | None = None,
        max_sessions: int = 128,
        metrics: BackendMetrics | None = None,
        crystal_vault: Any | None = None,
        crystal_node: Any | None = None,
        enable_crystal_auto_load: bool = True,
    ):
        if max_sessions < 1:
            raise ValueError("max_sessions must be positive")
        self.engine_factory = engine_factory
        self.checkpoint_store = checkpoint_store
        self.action_gateway = action_gateway
        self.response_generator = response_generator
        self.generation_config = generation_config or ResponseGenerationConfig()
        self.max_sessions = max_sessions
        self.metrics = metrics or BackendMetrics()
        self._sessions: dict[str, BackendSession] = {}
        self._active: dict[str, int] = {}
        self._closing: set[str] = set()
        self._condition = Condition(RLock())

        self.crystal_vault = crystal_vault
        self.crystal_node = crystal_node
        self._cerebrum_bridge: Any | None = None
        self._tdd_engine: Any | None = None
        self._hygiene_pipeline: Any | None = None
        if self.crystal_vault is None and enable_crystal_auto_load:
            self._init_default_crystal()
        if self.crystal_vault is not None and self.crystal_node is None:
            from mirror_crystal import CloudCrystalNode
            self.crystal_node = CloudCrystalNode(self.crystal_vault)

    @property
    def cerebrum_bridge(self):
        if self._cerebrum_bridge is None:
            from mirror_crystal import CerebrumCrystalBridge
            self._cerebrum_bridge = CerebrumCrystalBridge()
        return self._cerebrum_bridge

    @property
    def tdd_engine(self):
        if self._tdd_engine is None:
            from mirror_crystal import AutonomousTDDEngine
            self._tdd_engine = AutonomousTDDEngine()
        return self._tdd_engine

    @property
    def hygiene_pipeline(self):
        if self._hygiene_pipeline is None:
            from mirror_crystal import SixGateDataHygienePipeline
            self._hygiene_pipeline = SixGateDataHygienePipeline()
        return self._hygiene_pipeline

    def _init_default_crystal(self) -> None:
        try:
            from pathlib import Path
            from mirror_crystal import DiskCrystalVault
            vault_p = Path("artifacts/mirror_crystal_all_datasets.dcrystal")
            if vault_p.exists():
                self.crystal_vault = DiskCrystalVault(vault_p, lru_rule_cache_size=256)
        except Exception:
            self.crystal_vault = None

    def crystal_ask(self, prompt: str, session_id: str = "default") -> dict[str, Any]:
        """Query the backing Mirror Crystal Vault or fallback to empty abstention."""
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
            }
        res = self.crystal_vault.respond(prompt, session_id=session_id)
        self.metrics.increment("crystal_queries_total")
        return {
            "answer": res.answer,
            "confidence": res.confidence,
            "mode": res.mode,
            "matched_rule_id": res.matched_rule_id,
            "latency_ms": res.latency_ms,
            "sandbox_verified": res.sandbox_verified,
            "reasoning_trace": res.reasoning_trace,
            "epistemic_tier": res.epistemic_tier,
        }

    def crystal_teach(self, question: str, answer: str, test_code: str = "", domain: str = "general") -> bool:
        """Instantly crystallize a verified rule into the live DiskCrystalVault."""
        from mirror_crystal import SixGateDataHygienePipeline, ZeroTrustSandboxJudge

        combined = f"{question}\n{answer}\n{test_code}"
        has_pii, pii_reason = SixGateDataHygienePipeline.scan_for_pii(combined)
        if has_pii:
            raise ValueError(f"Rejected teaching payload containing PII/secret: {pii_reason}")
        lower_combined = combined.lower()
        for probe in SixGateDataHygienePipeline.BENCHMARK_PROBES:
            if probe.lower() in lower_combined:
                raise ValueError(f"Rejected contaminated teaching payload (benchmark probe '{probe}')")
        for block in (ZeroTrustSandboxJudge.extract_python_code(answer), test_code.strip()):
            if block:
                safe, reason = ZeroTrustSandboxJudge.is_ast_safe(block)
                if not safe:
                    raise ValueError(f"Rejected unsafe Python code in teaching payload: {reason}")

        if self.crystal_vault is None:
            from mirror_crystal import MirrorCrystalEngine
            self.crystal_vault = MirrorCrystalEngine()
            from mirror_crystal import CloudCrystalNode
            self.crystal_node = CloudCrystalNode(self.crystal_vault)
        added = self.crystal_vault.ingest_pair(question, answer, test_code=test_code, domain=domain)
        self.metrics.increment("crystal_rules_taught")
        return added

    def crystal_teleport(self, scrubbed_prompt: str, device_id: str = "pocket_web") -> dict[str, Any]:
        """Teleport an HMAC-signed rule capsule to edge devices with token-bucket rate limiting."""
        if self.crystal_node is None:
            if self.crystal_vault is not None:
                from mirror_crystal import CloudCrystalNode
                self.crystal_node = CloudCrystalNode(self.crystal_vault)
            else:
                return {"status": "error", "answer": "No crystal node active", "capsule": None, "cdn_cached": False, "cloud_latency_ms": 0.0}
        pkt = self.crystal_node.handle_teleport_query(scrubbed_prompt, device_id=device_id)
        self.metrics.increment("crystal_teleports_total")
        return pkt

    def crystal_auto_ingest_document(self, text: str, title: str = "WebDocument") -> dict[str, int]:
        """Auto-crystallize unstructured Markdown, textbook text, or Python code into the vault."""
        from mirror_crystal import SixGateDataHygienePipeline

        has_pii, pii_reason = SixGateDataHygienePipeline.scan_for_pii(text)
        if has_pii:
            raise ValueError(f"Rejected document containing PII/secret: {pii_reason}")
        lower_text = text.lower()
        for probe in SixGateDataHygienePipeline.BENCHMARK_PROBES:
            if probe.lower() in lower_text:
                raise ValueError(f"Rejected contaminated document (benchmark probe '{probe}')")

        if self.crystal_vault is None:
            from mirror_crystal import MirrorCrystalEngine
            self.crystal_vault = MirrorCrystalEngine()
        if hasattr(self.crystal_vault, "ingest_document"):
            return self.crystal_vault.ingest_document(text, title=title)
        from mirror_crystal import DocumentAutoCrystallizer
        return DocumentAutoCrystallizer.ingest_document_text(self.crystal_vault, text, title=title)

    def cerebrum_solve(self, prompt: str) -> dict[str, Any]:
        """Dual-Brain Neural-Symbolic Solver (Qwen-2.5-Math parity)."""
        return self.cerebrum_bridge.solve(prompt)

    def tdd_synthesize(self, candidate_code: str, spec_desc: str = "", explicit_tests: str = "") -> dict[str, Any]:
        """Compiler-Verified Test-Driven Synthesizer with Autonomous Self-Repair (Opus Coder parity)."""
        import dataclasses
        rep = self.tdd_engine.run_tdd_cycle(candidate_code, spec_desc=spec_desc, explicit_tests=explicit_tests)
        return dataclasses.asdict(rep)

    def six_gate_audit(self, text: str, spdx_license: str = "MIT") -> dict[str, Any]:
        """'Quality Over Quantity' 6-Gate Permissive Data Hygiene Auditor."""
        import dataclasses
        res = self.hygiene_pipeline.audit_record(text, spdx_license=spdx_license)
        return dataclasses.asdict(res)

    def _register(self, session_id: str, session: BackendSession) -> BackendSession:
        if session_id in self._sessions:
            raise ValueError("session already exists")
        if len(self._sessions) >= self.max_sessions:
            raise RuntimeError("backend session capacity exhausted")
        self._sessions[session_id] = session
        self._active[session_id] = 0
        return session

    def create_session(self, session_id: str, *, max_history: int = 512) -> BackendSession:
        with self._condition:
            engine = self.engine_factory() if self.engine_factory is not None else None
            session = self._register(
                session_id,
                BackendSession(session_id, engine=engine, max_history=max_history),
            )
            self.metrics.increment("sessions_created")
            return session

    def get_session(self, session_id: str) -> BackendSession:
        with self._condition:
            try:
                return self._sessions[session_id]
            except KeyError as exc:
                raise KeyError(f"unknown session: {session_id}") from exc

    def _begin_step(self, session_id: str) -> BackendSession:
        with self._condition:
            if session_id in self._closing:
                raise RuntimeError("session is closing")
            try:
                session = self._sessions[session_id]
            except KeyError as exc:
                raise KeyError(f"unknown session: {session_id}") from exc
            self._active[session_id] += 1
            self.metrics.increment("steps_started")
            return session

    def _end_step(self, session_id: str, *, succeeded: bool) -> None:
        with self._condition:
            self._active[session_id] -= 1
            self.metrics.increment("steps_succeeded" if succeeded else "steps_failed")
            self._condition.notify_all()

    def step(
        self,
        session_id: str,
        observation: Any,
        *,
        goal: str | None = None,
        research_tasks: tuple[Any, ...] = (),
        views: tuple[Any, ...] = (),
    ) -> BackendResult:
        session = self._begin_step(session_id)
        succeeded = False
        try:
            result = session.step(
                observation, goal=goal, research_tasks=research_tasks, views=views
            )
            if self.response_generator is not None:
                request = build_response_generation_request(result.realization_contract)
                if request is not None:
                    response = generate_response(request, self.response_generator)
                    contract = result.realization_contract
                    response = validate_generated_response(
                        response,
                        mode=getattr(contract, "mode", None),
                        config=self.generation_config,
                    )
                    result = replace(result, response=response)
            succeeded = True
            return result
        finally:
            self._end_step(session_id, succeeded=succeeded)

    def set_generation_config(self, config: ResponseGenerationConfig) -> None:
        """Replace response-generation policy after validating the immutable config."""
        if not isinstance(config, ResponseGenerationConfig):
            raise TypeError("config must be a ResponseGenerationConfig")
        self.generation_config = config

    def set_response_model(self, model: Any | None) -> None:
        """Install or remove the response model adapter at runtime."""
        self.response_generator = None if model is None else make_response_generator(model)

    def save(self, session_id: str):
        if self.checkpoint_store is None:
            raise RuntimeError("checkpoint store is not configured")
        session = self.get_session(session_id)
        return self.checkpoint_store.save(session_id, session.snapshot())

    def restore(self, session_id: str, *, max_history: int = 512) -> BackendSession:
        if self.checkpoint_store is None:
            raise RuntimeError("checkpoint store is not configured")
        snapshot = self.checkpoint_store.load(session_id)
        with self._condition:
            engine = self.engine_factory() if self.engine_factory is not None else None
            session = BackendSession(session_id, engine=engine, max_history=max_history)
            session.restore(snapshot)
            return self._register(session_id, session)

    def execute_action(self, name: str, *args, **kwargs) -> Any:
        if self.action_gateway is None:
            raise RuntimeError("action gateway is not configured")
        return self.action_gateway.execute(name, *args, **kwargs)

    def close_session(self, session_id: str) -> None:
        with self._condition:
            if session_id not in self._sessions:
                raise KeyError(f"unknown session: {session_id}")
            self._closing.add(session_id)
            try:
                while self._active[session_id]:
                    self._condition.wait()
                del self._sessions[session_id]
                del self._active[session_id]
                self.metrics.increment("sessions_closed")
            finally:
                self._closing.discard(session_id)
                self._condition.notify_all()

    def list_sessions(self) -> tuple[str, ...]:
        with self._condition:
            return tuple(sorted(self._sessions))

    def session_snapshot(self, session_id: str) -> Mapping[str, Any]:
        return self.get_session(session_id).snapshot()

    def has_checkpoint(self, session_id: str) -> bool:
        if self.checkpoint_store is None:
            raise RuntimeError("checkpoint store is not configured")
        return self.checkpoint_store.path_for(session_id).is_file()

    def delete_checkpoint(self, session_id: str) -> None:
        if self.checkpoint_store is None:
            raise RuntimeError("checkpoint store is not configured")
        self.checkpoint_store.delete(session_id)

    def status(self) -> Mapping[str, Any]:
        with self._condition:
            crystal_info: dict[str, Any] = {
                "active": self.crystal_vault is not None,
                "metrics": self.metrics.crystal_snapshot(),
            }
            if self.crystal_vault is not None:
                if hasattr(self.crystal_vault, "rule_count"):
                    crystal_info["rule_count"] = self.crystal_vault.rule_count()
                elif hasattr(self.crystal_vault, "rule_index"):
                    crystal_info["rule_count"] = len(self.crystal_vault.rule_index.rules)
                if hasattr(self.crystal_vault, "db_path"):
                    crystal_info["vault_path"] = str(self.crystal_vault.db_path)
            return {
                "version": self.VERSION,
                "sessions": len(self._sessions),
                "max_sessions": self.max_sessions,
                "session_ids": tuple(sorted(self._sessions)),
                "active_steps": sum(self._active.values()),
                "closing_sessions": len(self._closing),
                "persistence": self.checkpoint_store is not None,
                "actions": self.action_gateway is not None,
                "response_generation": self.response_generator is not None,
                "crystal": crystal_info,
                "metrics": self.metrics.snapshot(),
            }
