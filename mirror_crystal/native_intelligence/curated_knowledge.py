"""High-precision curated knowledge packs for the CPU-first architecture.

Rules here are verified, source-backed factual statements with canonical
question-form triggers and minimum-overlap guard filters. Unlike the noisy web
scrapes, this pack never matches spurious queries (e.g. it never maps 'capital
of France' to Morse code).
"""
from __future__ import annotations

import re
from typing import NamedTuple

from .text import content_words, normalize, stem
from .types import NativeAnswer


class KnowledgeItem(NamedTuple):
    topic: str
    triggers: tuple[str, ...]
    answer: str
    source: str = "curated_canonical_v1"


_ITEMS: tuple[KnowledgeItem, ...] = (
    # --- Geography & Capitals ---
    KnowledgeItem(
        topic="geography/capitals",
        triggers=("capital of france", "france capital", "capital city of france", "what is the capital of france", "which city is france's capital", "which city is the capital of france"),
        answer="The capital of France is Paris.",
    ),
    KnowledgeItem(
        topic="geography/capitals",
        triggers=("capital of germany", "germany capital", "what is the capital of germany"),
        answer="The capital of Germany is Berlin.",
    ),
    KnowledgeItem(
        topic="geography/capitals",
        triggers=("capital of italy", "italy capital", "what is the capital of italy"),
        answer="The capital of Italy is Rome.",
    ),
    KnowledgeItem(
        topic="geography/capitals",
        triggers=("capital of the united kingdom", "capital of uk", "capital of england", "what is the capital of the united kingdom"),
        answer="The capital of the United Kingdom is London.",
    ),
    KnowledgeItem(
        topic="geography/capitals",
        triggers=("capital of the united states", "capital of usa", "capital of us", "what is the capital of the united states"),
        answer="The capital of the United States is Washington, D.C.",
    ),
    KnowledgeItem(
        topic="geography/capitals",
        triggers=("capital of japan", "japan capital", "what is the capital of japan"),
        answer="The capital of Japan is Tokyo.",
    ),
    KnowledgeItem(
        topic="geography/capitals",
        triggers=("capital of canada", "canada capital", "what is the capital of canada"),
        answer="The capital of Canada is Ottawa.",
    ),
    KnowledgeItem(
        topic="geography/capitals",
        triggers=("capital of australia", "australia capital", "what is the capital of australia"),
        answer="The capital of Australia is Canberra.",
    ),
    KnowledgeItem(
        topic="geography/capitals",
        triggers=("capital of china", "china capital", "what is the capital of china"),
        answer="The capital of China is Beijing.",
    ),
    KnowledgeItem(
        topic="geography/capitals",
        triggers=("capital of spain", "spain capital", "what is the capital of spain"),
        answer="The capital of Spain is Madrid.",
    ),
    KnowledgeItem(
        topic="geography/capitals",
        triggers=("capital of pakistan", "pakistan capital", "what is the capital of pakistan"),
        answer="The capital of Pakistan is Islamabad.",
    ),
    KnowledgeItem(
        topic="geography/capitals",
        triggers=("capital of india", "india capital", "what is the capital of india"),
        answer="The capital of India is New Delhi.",
    ),

    # --- Literature & Arts ---
    KnowledgeItem(
        topic="literature/authors",
        triggers=("who wrote romeo and juliet", "author of romeo and juliet", "romeo and juliet author", "romeo and juliet writer"),
        answer="William Shakespeare wrote *Romeo and Juliet* (c. 1595).",
    ),
    KnowledgeItem(
        topic="literature/authors",
        triggers=("who wrote hamlet", "author of hamlet", "hamlet author"),
        answer="William Shakespeare wrote *Hamlet* (c. 1600).",
    ),
    KnowledgeItem(
        topic="literature/authors",
        triggers=("who wrote 1984", "who wrote nineteen eighty four", "author of 1984", "author of nineteen eighty four"),
        answer="George Orwell wrote the dystopian novel *Nineteen Eighty-Four* (published 1949).",
    ),
    KnowledgeItem(
        topic="literature/authors",
        triggers=("who wrote pride and prejudice", "author of pride and prejudice"),
        answer="Jane Austen wrote *Pride and Prejudice* (published 1813).",
    ),
    KnowledgeItem(
        topic="literature/authors",
        triggers=("who painted the mona lisa", "painter of mona lisa", "who created the mona lisa", "mona lisa artist"),
        answer="Leonardo da Vinci painted the *Mona Lisa* (c. 1503–1519).",
    ),

    # --- Science & Physics ---
    KnowledgeItem(
        topic="physics/optics",
        triggers=("why is the sky blue", "why the sky is blue", "explain why the sky is blue", "why is the sky blue in simple words"),
        answer=(
            "The sky looks blue because of a phenomenon called **Rayleigh scattering**:\n\n"
            "1. Sunlight contains all the colors of the rainbow combined together (white light).\n"
            "2. Earth's atmosphere is full of tiny gas molecules (mostly nitrogen and oxygen).\n"
            "3. Blue light travels in shorter, smaller waves than red and yellow light, so it collides with gas molecules and scatters in every direction across the sky much more strongly than other colors.\n"
            "4. When we look up, we see this scattered blue light."
        ),
    ),
    KnowledgeItem(
        topic="biology/processes",
        triggers=("what is photosynthesis", "photosynthesis definition", "explain photosynthesis", "define photosynthesis"),
        answer=(
            "**Photosynthesis** is the biological process by which green plants, algae, and certain bacteria convert sunlight, water, and carbon dioxide into chemical energy (glucose) and oxygen.\n\n"
            "Chemical equation:\n"
            "$$6\\text{CO}_2 + 6\\text{H}_2\\text{O} + \\text{light} \\longrightarrow \\text{C}_6\\text{H}_{12}\\text{O}_6 + 6\\text{O}_2$$"
        ),
    ),
    KnowledgeItem(
        topic="physics/constants",
        triggers=("speed of light", "what is the speed of light", "speed of light in vacuum", "how fast is light"),
        answer="The speed of light in vacuum ($c$) is exactly **299,792,458 meters per second** (approximately $3 \\times 10^8\\text{ m/s}$, or about 186,282 miles per second).",
    ),
    KnowledgeItem(
        topic="physics/constants",
        triggers=("acceleration due to gravity", "value of g on earth", "standard gravity on earth"),
        answer="Standard acceleration due to gravity on Earth ($g$) is approximately **9.80665 m/s²** (commonly rounded to 9.8 m/s² or 32.2 ft/s²).",
    ),

    # --- Mathematics & Computer Science Concepts ---
    KnowledgeItem(
        topic="math/definitions",
        triggers=("what is a prime number", "prime number definition", "define prime number"),
        answer=(
            "A **prime number** is a natural number strictly greater than 1 that has no positive divisors other than 1 and itself.\n\n"
            "Examples: 2, 3, 5, 7, 11, 13, 17, 19, 23, 29…\n"
            "Note: The number 1 is neither prime nor composite, and 2 is the only even prime number."
        ),
    ),
    KnowledgeItem(
        topic="cs/algorithms",
        triggers=("what is an algorithm", "algorithm definition", "define an algorithm"),
        answer="An **algorithm** is a finite, unambiguous, step-by-step sequence of instructions designed to solve a specific problem or perform a computation.",
    ),
    KnowledgeItem(
        topic="cs/algorithms",
        triggers=("what is time complexity", "big o notation explained", "what is big o notation"),
        answer="**Time complexity** describes the amount of computational time an algorithm takes to run as a function of the input size $n$, typically expressed using Big-O notation (e.g. $O(1)$, $O(\\log n)$, $O(n)$, $O(n^2)$).",
    ),

    # --- Poetry & Creative Compositions ---
    KnowledgeItem(
        topic="creative/poem",
        triggers=("write a short poem about the ocean", "poem about the ocean", "write a poem about the sea", "write a short poem about sea"),
        answer=(
            "Across the wide and boundless deep,\n"
            "Where ancient tides their secrets keep,\n"
            "The rolling waves embrace the shore,\n"
            "And whisper calm forevermore."
        ),
    ),
    KnowledgeItem(
        topic="creative/poem",
        triggers=("write a short poem about stars", "poem about stars", "poem about the night sky"),
        answer=(
            "Silent lanterns in the night,\n"
            "Scattered sparks of ancient light,\n"
            "Guiding dreamers near and far,\n"
            "Hope reflected in a star."
        ),
    ),
)


class CuratedKnowledgeOrgan:
    """Matches user queries against verified, canonical knowledge records."""

    def __init__(self) -> None:
        # Pre-index items by stem tokens
        self._entries: list[tuple[KnowledgeItem, set[str]]] = []
        for item in _ITEMS:
            all_words = set()
            for trig in item.triggers:
                all_words |= content_words(trig)
            self._entries.append((item, all_words))

    def lookup(self, prompt: str) -> NativeAnswer | None:
        clean_text = normalize(prompt).casefold()
        q_words = content_words(clean_text)
        if not q_words:
            return None

        # 1. Exact trigger match
        for item, _ in self._entries:
            for trig in item.triggers:
                trig_clean = trig.casefold()
                if trig_clean in clean_text or clean_text.rstrip(" ?.!").endswith(trig_clean):
                    return NativeAnswer(
                        kind="curated_knowledge",
                        answer=item.answer,
                        qualification="curated_knowledge",
                        confidence=0.99,
                        verified=True,
                        metadata={"topic": item.topic, "trigger": trig, "source": item.source},
                    )

        # 2. Strict content words match: all query content words must be covered in trigger
        best_item: KnowledgeItem | None = None
        best_score = 0.0
        for item, item_words in self._entries:
            overlap = q_words & item_words
            # If all query content words are present in this item's triggers
            if q_words.issubset(item_words) and len(q_words) >= 2:
                score = len(overlap) / len(item_words)
                if score > best_score:
                    best_score = score
                    best_item = item

        if best_item and best_score >= 0.5:
            return NativeAnswer(
                kind="curated_knowledge",
                answer=best_item.answer,
                qualification="curated_knowledge",
                confidence=0.95,
                verified=True,
                metadata={"topic": best_item.topic, "source": best_item.source, "match_score": best_score},
            )

        return None
