"""Per-session personal fact memory ("My name is Ahmad" -> "What is my name?").

Facts are what the *user* said about themselves. They are context, never
verified world knowledge, and they stay in process memory for the session.
"""
from __future__ import annotations

import re
import threading
from collections import OrderedDict

from .text import clean_entity, normalize
from .types import NativeAnswer

_MAX_SESSIONS = 1024
_MAX_FACTS = 64

_NAME_STOP = {"not", "a", "an", "the", "very", "so", "really", "feeling", "fine", "good", "ok", "okay",
              "happy", "sad", "tired", "here", "back", "sorry", "sure", "ready", "busy", "hungry"}

_STORE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("name", re.compile(r"^(?:hi|hello|hey)?[,!\s]*(?:my name is|my name's|call me|i am called|i'm called|you can call me)\s+(?P<v>[a-z][a-z .'-]{0,40}?)\s*(?:[.!,]|and\b|$)", re.I)),
    ("name", re.compile(r"^(?:hi|hello|hey)?[,!\s]*(?:i am|i'm)\s+(?P<v>[A-Z][a-zA-Z'-]{1,30}(?:\s+[A-Z][a-zA-Z'-]{1,30})?)\s*[.!]?$")),
    ("age", re.compile(r"\b(?:i am|i'm)\s+(?P<v>\d{1,3})\s*(?:years?|yrs?)(?:\s+old)?\b", re.I)),
    ("age", re.compile(r"\bmy age is\s+(?P<v>\d{1,3})\b", re.I)),
    ("location", re.compile(r"\bi (?:live|stay|reside) in\s+(?P<v>[a-z][a-z .'-]{1,40}?)\s*(?:[.!,]|and\b|$)", re.I)),
    ("origin", re.compile(r"\b(?:i am|i'm|i come) from\s+(?P<v>[a-z][a-z .'-]{1,40}?)\s*(?:[.!,]|and\b|$)", re.I)),
    ("job", re.compile(r"\bi (?:work as|am working as)\s+(?:an?\s+)?(?P<v>[a-z][a-z .'-]{1,40}?)\s*(?:[.!,]|and\b|$)", re.I)),
    ("job", re.compile(r"\bmy (?:job|profession|occupation) is\s+(?:an?\s+)?(?P<v>[a-z][a-z .'-]{1,40}?)\s*(?:[.!,]|and\b|$)", re.I)),
    ("likes", re.compile(r"\bi (?:really )?(?:like|love|enjoy)\s+(?P<v>[a-z][a-z .'-]{1,40}?)\s*(?:[.!,]|and\b|$)", re.I)),
)
_FAVORITE = re.compile(r"\bmy fav(?:ou?rite)?\s+(?P<k>[a-z ]{2,24}?)\s+is\s+(?P<v>[a-z0-9][a-z0-9 .'-]{0,40}?)\s*(?:[.!,]|$)", re.I)
_GENERIC = re.compile(r"^my\s+(?P<k>[a-z][a-z ]{1,24}?)\s+is\s+(?P<v>[a-z0-9][a-z0-9 .'@-]{0,40}?)\s*[.!]?$", re.I)

_RECALL_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("name", re.compile(r"\b(?:what(?:'s| is) my name|who am i|do you (?:know|remember) my name|what do you call me)\b", re.I)),
    ("age", re.compile(r"\b(?:how old am i|what(?:'s| is) my age)\b", re.I)),
    ("location", re.compile(r"\b(?:where do i (?:live|stay)|where am i living)\b", re.I)),
    ("origin", re.compile(r"\b(?:where am i from|where do i come from)\b", re.I)),
    ("job", re.compile(r"\b(?:what(?:'s| is) my (?:job|profession|occupation)|what do i do for (?:a )?(?:living|work))\b", re.I)),
    ("likes", re.compile(r"\b(?:what do i (?:like|love|enjoy))\b", re.I)),
)
_RECALL_FAVORITE = re.compile(r"\bwhat(?:'s| is) my fav(?:ou?rite)?\s+(?P<k>[a-z ]{2,24}?)\s*\??$", re.I)
_RECALL_GENERIC = re.compile(r"^what(?:'s| is) my\s+(?P<k>[a-z][a-z ]{1,24}?)\s*\??$", re.I)
_RECALL_ALL = re.compile(r"\b(?:what do you (?:know|remember) about me|tell me about myself)\b", re.I)

_LABELS = {"name": "name", "age": "age", "location": "home", "origin": "origin", "job": "job", "likes": "likes"}


class PersonalFactMemory:
    """Thread-safe, bounded, per-session store of facts the user states about themselves."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._sessions: "OrderedDict[str, OrderedDict[str, str]]" = OrderedDict()

    # -- storage -------------------------------------------------------
    def _facts(self, session_id: str) -> "OrderedDict[str, str]":
        facts = self._sessions.get(session_id)
        if facts is None:
            facts = OrderedDict()
            self._sessions[session_id] = facts
            while len(self._sessions) > _MAX_SESSIONS:
                self._sessions.popitem(last=False)
        else:
            self._sessions.move_to_end(session_id)
        return facts

    def remember(self, session_id: str, key: str, value: str) -> None:
        with self._lock:
            facts = self._facts(session_id)
            if key == "likes" and "likes" in facts and value.casefold() not in facts["likes"].casefold():
                value = facts["likes"] + ", " + value
            facts[key] = value
            facts.move_to_end(key)
            while len(facts) > _MAX_FACTS:
                facts.popitem(last=False)

    def get(self, session_id: str, key: str) -> str | None:
        with self._lock:
            return self._sessions.get(session_id, {}).get(key)

    def all(self, session_id: str) -> dict[str, str]:
        with self._lock:
            return dict(self._sessions.get(session_id, {}))

    # -- parsing -------------------------------------------------------
    @staticmethod
    def extract(prompt: str) -> list[tuple[str, str]]:
        text = normalize(prompt)
        if "?" in text:
            return []
        found: list[tuple[str, str]] = []
        for clause in re.split(r"(?<=[.!])\s+|\s+and\s+(?=(?:i|my)\b)", text, flags=re.I):
            clause = clause.strip()
            if not clause:
                continue
            matched = False
            fav = _FAVORITE.search(clause)
            if fav:
                found.append(("favorite " + clean_entity(fav.group("k")).casefold(), clean_entity(fav.group("v"))))
                continue
            for key, pattern in _STORE_PATTERNS:
                m = pattern.search(clause)
                if not m:
                    continue
                value = clean_entity(m.group("v"))
                if key == "name":
                    first = value.split()[0].casefold() if value else ""
                    if not value or first in _NAME_STOP:
                        continue
                    value = " ".join(part[:1].upper() + part[1:] for part in value.split())
                if value:
                    found.append((key, value))
                    matched = True
                    break
            if not matched:
                gen = _GENERIC.match(clause)
                if gen:
                    key = clean_entity(gen.group("k")).casefold()
                    if key not in {"question", "answer", "problem", "code", "name is"}:
                        found.append((key, clean_entity(gen.group("v"))))
        return found

    def try_store(self, session_id: str, prompt: str) -> NativeAnswer | None:
        facts = self.extract(prompt)
        if not facts:
            return None
        for key, value in facts:
            self.remember(session_id, key, value)
        parts = []
        for key, value in facts:
            if key == "name":
                parts.append(f"Nice to meet you, {value}! I'll remember your name.")
            elif key == "age":
                parts.append(f"Got it — you are {value} years old.")
            elif key == "location":
                parts.append(f"Noted — you live in {value}.")
            elif key == "origin":
                parts.append(f"Noted — you are from {value}.")
            elif key == "job":
                parts.append(f"Noted — you work as {value}.")
            elif key == "likes":
                parts.append(f"Noted — you like {value}.")
            else:
                parts.append(f"Noted — your {key} is {value}.")
        return NativeAnswer(
            kind="memory_store",
            answer=" ".join(parts),
            qualification="memory_stored",
            confidence=0.95,
            metadata={"stored": dict(facts)},
        )

    def try_recall(self, session_id: str, prompt: str) -> NativeAnswer | None:
        text = normalize(prompt)
        if _RECALL_ALL.search(text):
            facts = self.all(session_id)
            if not facts:
                answer = "You haven't told me anything about yourself yet in this session."
            else:
                answer = "Here is what you told me: " + "; ".join(
                    f"{_LABELS.get(k, k)}: {v}" for k, v in facts.items()
                ) + "."
            return self._recall_answer(answer, "all", bool(facts))

        key: str | None = None
        fav = _RECALL_FAVORITE.search(text)
        if fav:
            key = "favorite " + clean_entity(fav.group("k")).casefold()
        if key is None:
            for candidate, pattern in _RECALL_PATTERNS:
                if pattern.search(text):
                    key = candidate
                    break
        if key is None:
            gen = _RECALL_GENERIC.match(text)
            if gen:
                guess = clean_entity(gen.group("k")).casefold()
                if guess in self.all(session_id):
                    key = guess
        if key is None:
            return None

        value = self.get(session_id, key)
        if value is None:
            missing = {
                "name": "I don't know your name yet. You can tell me with \"My name is …\".",
                "age": "You haven't told me your age yet.",
                "location": "You haven't told me where you live yet.",
                "origin": "You haven't told me where you are from yet.",
                "job": "You haven't told me what you do yet.",
                "likes": "You haven't told me what you like yet.",
            }
            return self._recall_answer(missing.get(key, f"You haven't told me your {key} yet."), key, False)
        templates = {
            "name": "Your name is {v}.",
            "age": "You told me you are {v} years old.",
            "location": "You told me you live in {v}.",
            "origin": "You told me you are from {v}.",
            "job": "You told me you work as {v}.",
            "likes": "You told me you like {v}.",
        }
        answer = templates.get(key, "You told me your " + key + " is {v}.").format(v=value)
        return self._recall_answer(answer, key, True)

    @staticmethod
    def _recall_answer(answer: str, key: str, found: bool) -> NativeAnswer:
        return NativeAnswer(
            kind="memory_recall",
            answer=answer,
            qualification="session_memory",
            confidence=0.95 if found else 0.6,
            metadata={"key": key, "found": found},
        )
