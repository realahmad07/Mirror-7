"""Systems under test (SUT) for the NovelGen benchmark.

``native_chain`` reproduces the native-organ routing order used by
``mirror7_backend.cognitive_orchestrator.UnifiedCognitiveOrchestrator.ask`` so the
benchmark measures what Mirror actually answers, without needing the vault,
backend providers or network.
"""
from __future__ import annotations

from typing import Callable

from .scoring import Response

# NativeAnswer.kind values that come from a general in-context inference engine
# rather than from topic/pattern-specific handlers.
REASONER_KINDS = frozenset({"context_reasoning"})


def _source_for(kind: str) -> str:
    return "reasoner" if kind in REASONER_KINDS else "handler"


def native_chain(*, include_context_reasoner: bool = True) -> Callable[[str], Response]:
    from mirror_crystal import native_intelligence as ni

    memory = ni.PersonalFactMemory()
    chat = ni.ConversationOrgan()
    logic = ni.LogicOrgan()
    words = ni.WordProblemSolver()
    curated = ni.CuratedKnowledgeOrgan()
    kg = ni.KnowledgeGraphEngine()
    multi_hop = ni.MultiHopChainingEngine(kg=kg)
    reasoner = None
    if include_context_reasoner:
        reasoner = getattr(ni, "ContextReasoner", None)
        reasoner = reasoner() if reasoner is not None else None

    def organs(prompt: str):
        # Same order as the orchestrator's non-math branch; the context reasoner
        # runs before pattern handlers because it only answers when every premise
        # needed for the question was parsed into its logical form.
        yield lambda: memory.try_store("eval", prompt)
        yield lambda: memory.try_recall("eval", prompt)
        yield lambda: chat.respond(prompt, session_id="eval")
        if reasoner is not None:
            yield lambda: reasoner.respond(prompt)
        yield lambda: multi_hop.resolve_chain(prompt)
        yield lambda: logic.respond(prompt)
        yield lambda: words.solve(prompt)
        yield lambda: kg.lookup_from_prompt(prompt) or curated.lookup(prompt)

    def sut(prompt: str) -> Response:
        for call in organs(prompt):
            try:
                hit = call()
            except Exception:
                hit = None
            if hit is not None:
                return Response(text=hit.answer, source=_source_for(hit.kind))
        return Response(text="", source="abstain")

    return sut


def always_yes() -> Callable[[str], Response]:
    """Trivial baseline: shows the floor a yes-biased pattern matcher reaches."""
    return lambda prompt: Response(text="Yes", source="handler")
