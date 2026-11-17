"""Multi-Hop Rule and Knowledge Graph Chaining Engine.

Traverses relational paths across the entity graph and rule database:
  Query: (Subject A) -> [Predicate 1] -> (Entity B) -> [Predicate 2] -> (Object C)
Enables multi-step reasoning such as:
  "What is the currency of the country whose capital is Paris?"
  -> Hop 1: Capital is Paris => Country is France
  -> Hop 2: France's currency => Euro (€)
"""
from __future__ import annotations

import re
from typing import Any

from .knowledge_graph import KnowledgeGraphEngine
from .text import clean_entity, normalize
from .types import NativeAnswer


class MultiHopChainingEngine:
    """Discovers and evaluates multi-hop reasoning chains across knowledge triples."""

    def __init__(self, kg: KnowledgeGraphEngine | None = None) -> None:
        self.kg = kg or KnowledgeGraphEngine()

    def resolve_chain(self, prompt: str) -> NativeAnswer | None:
        text = normalize(prompt).strip(" ?!.").casefold()

        # Pattern 1: "What is the [Pred2] of the country/entity whose [Pred1] is [Val1]?"
        # e.g.: "What is the currency of the country whose capital is Paris?"
        # e.g.: "What is the continent of the nation whose capital is Tokyo?"
        m = re.search(
            r"\b(?:what(?:'s| is| are)?|tell me)\s+(?:the\s+)?([a-z_]+)\s+of\s+(?:the\s+)?(?:country|nation|entity|place|state)\s+whose\s+([a-z_]+)\s+is\s+([a-zA-Z\s.-]+)",
            text,
            re.I,
        )
        if m:
            pred2 = clean_entity(m.group(1)).casefold()
            pred1 = clean_entity(m.group(2)).casefold()
            val1 = clean_entity(m.group(3))

            # Hop 1: Find intermediate entity (subject where pred1 = val1)
            hop1_candidates = self.kg.query_by_object(val1, predicate=pred1)
            if not hop1_candidates:
                # Fallback: check without strict predicate filter
                hop1_candidates = self.kg.query_by_object(val1)

            if hop1_candidates:
                intermediate_subject = hop1_candidates[0][0]

                # Hop 2: Query pred2 on intermediate entity
                hop2_results = self.kg.query(intermediate_subject, pred2)
                if hop2_results:
                    final_object, conf = hop2_results[0]
                    steps = (
                        f"Hop 1: Identified entity whose {pred1} is '{val1}' -> {intermediate_subject.title()}",
                        f"Hop 2: Queried {pred2} of {intermediate_subject.title()} -> {final_object}",
                    )
                    answer = (
                        f"{val1.title()} is the {pred1} of **{intermediate_subject.title()}**, "
                        f"and the {pred2} of {intermediate_subject.title()} is **{final_object}**.\n\n"
                        f"**Multi-Hop Evidence Chain:**\n"
                        f"• {steps[0]}\n"
                        f"• {steps[1]}"
                    )
                    return NativeAnswer(
                        kind="multi_hop_chain",
                        answer=answer,
                        qualification="deductive_proof",
                        confidence=conf,
                        verified=True,
                        steps=steps,
                        metadata={
                            "hops": 2,
                            "intermediate": intermediate_subject,
                            "final": final_object,
                            "path": [val1, intermediate_subject, final_object],
                        },
                    )

        # Pattern 2: "What is the [Pred2] of the capital of [Country]?"
        # e.g.: "What is the continent of the capital of France?" (France -> capital = Paris -> continent = Europe)
        m2 = re.search(
            r"\b(?:what(?:'s| is| are)?|tell me)\s+(?:the\s+)?([a-z_]+)\s+of\s+(?:the\s+)?([a-z_]+)\s+of\s+([a-zA-Z\s.-]+)",
            text,
            re.I,
        )
        if m2:
            pred2 = clean_entity(m2.group(1)).casefold()
            pred1 = clean_entity(m2.group(2)).casefold()
            subj = clean_entity(m2.group(3)).casefold()

            hop1_res = self.kg.query(subj, pred1)
            if hop1_res:
                intermediate = hop1_res[0][0]
                # Look up pred2 on intermediate or on original subject
                hop2_res = self.kg.query(intermediate, pred2) or self.kg.query(subj, pred2)
                if hop2_res:
                    final_obj, conf = hop2_res[0]
                    steps = (
                        f"Hop 1: {pred1.title()} of {subj.title()} -> {intermediate}",
                        f"Hop 2: {pred2.title()} -> {final_obj}",
                    )
                    answer = (
                        f"The {pred1} of {subj.title()} is **{intermediate}**, "
                        f"whose {pred2} is **{final_obj}**.\n\n"
                        f"**Multi-Hop Evidence Chain:**\n"
                        f"• {steps[0]}\n"
                        f"• {steps[1]}"
                    )
                    return NativeAnswer(
                        kind="multi_hop_chain",
                        answer=answer,
                        qualification="deductive_proof",
                        confidence=conf,
                        verified=True,
                        steps=steps,
                        metadata={"intermediate": intermediate, "final": final_obj},
                    )

        return None
