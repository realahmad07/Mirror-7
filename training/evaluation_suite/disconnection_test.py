"""Teacher-disconnection and native-checkpoint independence verification.

This audit checks offline checkpoint loading, actual local generation, and
independent operation of the existing cognitive backend. It does not treat a
missing checkpoint as a successful load and does not confuse a completed
benchmark run with a high benchmark score.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from mirror7_backend.service import BackendService
from phase140_unified_cognitive_runtime.mirror7_phase140 import UnifiedCognitiveRuntime


class TeacherDependencyViolation(Exception):
    """Raised when teacher generation is invoked during the audit."""


@dataclass(frozen=True)
class DisconnectionAuditResult:
    is_fully_independent: bool
    checkpoint_loaded_standalone: bool
    zero_teacher_calls: bool
    zero_credential_requirements: bool
    backend_subsystems_operational: bool
    unseen_evaluation_handled: bool
    audit_notes: tuple[str, ...] = ()
    held_out_overall_score: float | None = None
    held_out_cases: int = 0
    checkpoint_format: str | None = None


class TeacherDisconnectionVerifier:
    """Verify native inference and Mirror's existing backend without a teacher."""

    def __init__(
        self,
        checkpoint_path: str | Path = "artifacts/mirror7_native_conversation_v5.pt",
    ):
        self.checkpoint_path = Path(checkpoint_path)

    def _load_checkpoint(self) -> tuple[Callable[[str], str], str]:
        import torch

        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(f"native checkpoint not found: {self.checkpoint_path}")
        payload = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
        checkpoint_format = payload.get("format")

        if checkpoint_format == "mirror7-native-conversation-v5-input-conditioned":
            from training.native_conversation_v5 import load_checkpoint

            model, _ = load_checkpoint(self.checkpoint_path, device="cpu")
            model.eval()
            return (
                lambda prompt: model.generate_text(
                    prompt, max_new_tokens=96, temperature=0.7
                ),
                checkpoint_format,
            )

        if checkpoint_format == "mirror7-native-conversation-v3-cross-attention":
            from training.native_conversation import (
                NativeConversationConfig,
                build_native_conversation,
                EOS,
                PAD,
            )

            config = NativeConversationConfig(**payload["config"])
            model = build_native_conversation(config)
            model.load_state_dict(payload["state_dict"], strict=True)
            model.eval()

            def generate(prompt: str) -> str:
                raw = list(prompt.encode("utf-8")[: config.max_bytes])
                if raw:
                    ids = torch.tensor([raw], dtype=torch.long)
                    mask = torch.ones_like(ids, dtype=torch.float32)
                else:
                    ids = torch.tensor([[PAD]], dtype=torch.long)
                    mask = torch.zeros_like(ids, dtype=torch.float32)
                tokens = model.generate(
                    ids, mask, max_new_tokens=96, temperature=0.7, top_k=40
                )[0].tolist()
                values: list[int] = []
                for token in tokens:
                    if token == EOS:
                        break
                    if 0 <= token <= 255:
                        values.append(token)
                return bytes(values).decode("utf-8", errors="replace")

            return generate, checkpoint_format

        raise ValueError(f"unsupported native checkpoint format: {checkpoint_format!r}")

    def run_full_disconnection_audit(self) -> DisconnectionAuditResult:
        notes: list[str] = []
        checkpoint_loaded = False
        backend_operational = False
        unseen_evaluation_handled = False
        held_out_score: float | None = None
        held_out_cases = 0
        checkpoint_format: str | None = None
        generate_fn: Callable[[str], str] | None = None

        missing = object()
        credential_names = ("HF_TOKEN", "HUGGINGFACEHUB_API_TOKEN", "OPENAI_API_KEY")
        saved_env = {key: os.environ.pop(key, missing) for key in credential_names}
        try:
            try:
                generate_fn, checkpoint_format = self._load_checkpoint()
                notes.append(f"Strictly loaded native checkpoint format {checkpoint_format!r} offline.")
            except Exception as exc:
                notes.append(f"Native checkpoint load failed: {type(exc).__name__}: {exc}")

            from training.knowledge_transfer.teacher_interface import TeacherProvider

            patched: list[tuple[type, object]] = []
            teacher_calls_detected = 0

            def trap_generate(_self, *args, **kwargs):
                nonlocal teacher_calls_detected
                teacher_calls_detected += 1
                raise TeacherDependencyViolation(
                    "teacher generation was called during native inference"
                )

            provider_types = [TeacherProvider]
            pending = list(TeacherProvider.__subclasses__())
            while pending:
                cls = pending.pop()
                provider_types.append(cls)
                pending.extend(cls.__subclasses__())
            for cls in provider_types:
                original = cls.__dict__.get("generate")
                if original is not None:
                    patched.append((cls, original))
                    setattr(cls, "generate", trap_generate)

            try:
                # Run real native inference only after all teacher entry points
                # have been trapped.
                if generate_fn is not None:
                    sample = generate_fn("Reply with a short greeting.")
                    if not isinstance(sample, str):
                        raise TypeError("native generation did not return text")
                    checkpoint_loaded = True
                    notes.append(f"Native model generated locally ({len(sample)} characters).")

                service = BackendService(engine_factory=UnifiedCognitiveRuntime)
                service.create_session("disconnection-audit-session")
                first = service.step(
                    session_id="disconnection-audit-session",
                    observation="Compute battery load and report status.",
                    goal="system_status",
                )
                second = service.step(
                    session_id="disconnection-audit-session",
                    observation="New event received: temperature sensor alert.",
                    goal="safety_check",
                )
                assert first.sequence == 1 and first.engine_result is not None
                assert first.state_digest
                assert second.sequence == 2 and second.engine_result is not None
                snapshot = service.session_snapshot("disconnection-audit-session")
                assert snapshot["sequence"] == 2
                backend_operational = True
                notes.append(
                    "Backend service and cognitive runtime completed two stateful turns and snapshot verification."
                )

                if generate_fn is not None:
                    from training.evaluation_suite import IndependentEvaluator

                    evaluator = IndependentEvaluator(partition="held_out")
                    report = evaluator.evaluate(
                        generate_fn, model_name=self.checkpoint_path.name
                    )
                    held_out_score = report.overall_score
                    held_out_cases = sum(
                        score.total_cases for score in report.dimension_scores.values()
                    )
                    unseen_evaluation_handled = held_out_cases > 0
                    notes.append(
                        f"Held-out benchmark executed: {held_out_cases} cases; "
                        f"score={held_out_score:.3f}. This records evaluation, not a pass claim."
                    )
            except Exception as exc:
                notes.append(f"Offline inference/backend/evaluation failed: {type(exc).__name__}: {exc}")
            finally:
                for cls, original in reversed(patched):
                    setattr(cls, "generate", original)

            zero_teacher = teacher_calls_detected == 0
            zero_credentials = all(os.environ.get(key) is None for key in credential_names)
            fully_independent = (
                checkpoint_loaded
                and zero_teacher
                and zero_credentials
                and backend_operational
                and unseen_evaluation_handled
            )
            return DisconnectionAuditResult(
                is_fully_independent=fully_independent,
                checkpoint_loaded_standalone=checkpoint_loaded,
                zero_teacher_calls=zero_teacher,
                zero_credential_requirements=zero_credentials,
                backend_subsystems_operational=backend_operational,
                unseen_evaluation_handled=unseen_evaluation_handled,
                audit_notes=tuple(notes),
                held_out_overall_score=held_out_score,
                held_out_cases=held_out_cases,
                checkpoint_format=checkpoint_format,
            )
        finally:
            for key, value in saved_env.items():
                if value is not missing:
                    os.environ[key] = value
