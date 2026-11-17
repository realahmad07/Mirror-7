"""Small-talk / conversational organ.

Handles greetings, identity, capability questions, thanks, farewells, jokes and
"are you human?" questions with honest, deterministic replies. It never makes
factual claims about the world.
"""
from __future__ import annotations

import hashlib
import re
from typing import Callable

from .text import normalize
from .types import NativeAnswer

JOKES = (
    "Why do programmers prefer dark mode? Because light attracts bugs.",
    "Why did the math book look sad? It had too many problems.",
    "I told my computer I needed a break, and it said: \"No problem — I'll go to sleep.\"",
    "Why was the equal sign so humble? It knew it wasn't less than or greater than anyone else.",
    "There are 10 kinds of people in the world: those who understand binary and those who don't.",
    "Why don't scientists trust atoms? Because they make up everything.",
    "Parallel lines have so much in common. It's a shame they'll never meet.",
    "Why did the developer go broke? Because he used up all his cache.",
    "What do you call a fake noodle? An impasta.",
    "Why can't a bicycle stand up by itself? It's two tired.",
)

CAPABILITIES = (
    "I'm Mirror 7, a CPU-only reasoning engine. I can:\n"
    "• do exact arithmetic, fractions, percentages and unit conversions;\n"
    "• differentiate, integrate and solve linear equation systems;\n"
    "• solve common word problems (speed, cost, percentages, totals) and show the steps;\n"
    "• work through logic puzzles (\"all X are Y\", \"A is taller than B\");\n"
    "• answer questions from my curated knowledge packs (capitals, authors, elements, science basics…);\n"
    "• write and test small Python functions;\n"
    "• remember what you tell me about yourself during our conversation.\n"
    "When I'm not sure, I say so instead of guessing."
)

_Rule = tuple[re.Pattern[str], Callable[[str | None, int], str], str]


def _greet(name: str | None, _: int) -> str:
    who = f", {name}" if name else ""
    return f"Hello{who}! I'm Mirror 7. How can I help you today?"


def _how_are_you(name: str | None, _: int) -> str:
    who = f", {name}" if name else ""
    return f"I'm running well, thanks for asking{who}! What would you like to work on?"


def _joke(_: str | None, turn: int) -> str:
    return JOKES[turn % len(JOKES)]


_RULES: tuple[_Rule, ...] = (
    (re.compile(r"^(?:(?:hi|hello|hey|hiya|howdy|yo|greetings|salam|salaam|assalam(?:u|o)? ?alaikum|good (?:morning|afternoon|evening|day))\b[\s,!.]*(?:there|mirror|everyone|friend)?[\s,!.]*)+(?:how are you(?: doing)?(?: today)?|how's it going|what's up)?[\s?!.]*$", re.I),
     _greet, "greeting"),
    (re.compile(r"^(?:how are you(?: doing)?(?: today)?|how's it going|how do you do|what's up|wassup|sup)[\s?!.]*$", re.I), _how_are_you, "how_are_you"),
    (re.compile(r"\b(?:what(?:'s| is) your name|who are you|what are you|introduce yourself)\b", re.I),
     lambda n, t: "I'm Mirror 7 — an open-source, CPU-first reasoning engine created by Ahmad Hassan. I combine exact math, logic, curated knowledge and session memory, and I tell you when I'm unsure.", "identity"),
    (re.compile(r"\b(?:who (?:made|created|built|developed|designed) you|who is your (?:creator|developer|author|founder))\b", re.I),
     lambda n, t: "I was created by Ahmad Hassan as part of the open-source Mirror 7 project (Apache-2.0).", "creator"),
    (re.compile(r"\b(?:what can you do|what do you do|how can you help|what are your (?:skills|capabilities|features)|help me|^help$)\b", re.I),
     lambda n, t: CAPABILITIES, "capabilities"),
    (re.compile(r"\b(?:are you (?:a )?(?:human|real|alive|conscious|sentient|a person|a robot|an ai|a bot|chatgpt))\b", re.I),
     lambda n, t: "I'm not a human and I'm not conscious. I'm a software system: a rule-and-reasoning engine that runs on a normal CPU.", "nature"),
    (re.compile(r"\b(?:tell|give|share)\s+(?:me\s+)?(?:a|another|one more)?\s*(?:funny\s+)?joke\b|\bmake me laugh\b|\bsay something funny\b", re.I), _joke, "joke"),
    (re.compile(r"^(?:thanks|thank you|thank u|thx|ty|cheers|much appreciated|thanks a lot|thank you so much)\b[\s\w,!.]*$", re.I),
     lambda n, t: "You're welcome" + (f", {n}" if n else "") + "! Anything else?", "thanks"),
    (re.compile(r"^(?:bye|goodbye|good bye|see you|see ya|good night|later|take care|cya)\b[\s\w,!.]*$", re.I),
     lambda n, t: "Goodbye" + (f", {n}" if n else "") + "! Come back any time.", "farewell"),
    (re.compile(r"^(?:ok|okay|cool|nice|great|awesome|perfect|got it|alright|good|wow|interesting|i see)[\s!.]*$", re.I),
     lambda n, t: "Great! What would you like to do next?", "acknowledge"),
    (re.compile(r"^(?:sorry|my bad|oops)\b[\s\w,!.]*$", re.I),
     lambda n, t: "No problem at all! How can I help?", "apology"),
    (re.compile(r"\b(?:do you have feelings|do you feel|are you happy|are you sad)\b", re.I),
     lambda n, t: "I don't have feelings — but I'm always ready to help.", "feelings"),
    (re.compile(r"\b(?:how old are you|when were you (?:born|created|made))\b", re.I),
     lambda n, t: "I don't have an age the way people do — I'm software that keeps improving with each version.", "age"),
    (re.compile(r"\b(?:i(?:'m| am) (?:bored|sad|tired|stressed|lonely))\b", re.I),
     lambda n, t: "I'm sorry to hear that. Want a joke, a quick puzzle, or help with something you're working on?", "empathy"),
)


class ConversationOrgan:
    """Deterministic conversational replies with per-session variety for jokes."""

    def __init__(self) -> None:
        self._turns: dict[str, int] = {}

    def respond(self, prompt: str, *, session_id: str = "default", user_name: str | None = None) -> NativeAnswer | None:
        text = normalize(prompt)
        if not text or len(text) > 160:
            return None
        for pattern, build, intent in _RULES:
            if pattern.search(text):
                turn = self._turns.get(session_id)
                if turn is None:
                    turn = int(hashlib.sha256(session_id.encode()).hexdigest(), 16) % len(JOKES)
                self._turns[session_id] = turn + 1
                if len(self._turns) > 4096:
                    self._turns.clear()
                return NativeAnswer(
                    kind="chat",
                    answer=build(user_name, turn),
                    qualification="conversational",
                    confidence=0.9,
                    metadata={"intent": intent},
                )
        return None
