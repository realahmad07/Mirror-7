"""Structured Knowledge Graph (entity_triples) for fast CPU factual retrieval.

Stores and queries facts as (subject, predicate, object) triples in SQLite with
sub-millisecond indexed lookups. Contains seed knowledge across geography, science,
literature, astronomy, and technology, and can dynamically ingest millions of triples.
"""
from __future__ import annotations

import os
import re
import sqlite3
import threading
from pathlib import Path
from typing import Any, NamedTuple

from .text import clean_entity, normalize, stem
from .types import NativeAnswer


class FactTriple(NamedTuple):
    subject: str
    predicate: str
    object: str
    category: str = "general"
    confidence: float = 1.0
    provenance: str = "curated_canonical"


# Seed knowledge across core world domains
_SEED_TRIPLES: tuple[FactTriple, ...] = (
    # --- Geography & Capitals ---
    FactTriple("france", "capital", "Paris", "geography"),
    FactTriple("germany", "capital", "Berlin", "geography"),
    FactTriple("italy", "capital", "Rome", "geography"),
    FactTriple("spain", "capital", "Madrid", "geography"),
    FactTriple("united kingdom", "capital", "London", "geography"),
    FactTriple("uk", "capital", "London", "geography"),
    FactTriple("england", "capital", "London", "geography"),
    FactTriple("united states", "capital", "Washington, D.C.", "geography"),
    FactTriple("usa", "capital", "Washington, D.C.", "geography"),
    FactTriple("canada", "capital", "Ottawa", "geography"),
    FactTriple("japan", "capital", "Tokyo", "geography"),
    FactTriple("china", "capital", "Beijing", "geography"),
    FactTriple("india", "capital", "New Delhi", "geography"),
    FactTriple("pakistan", "capital", "Islamabad", "geography"),
    FactTriple("australia", "capital", "Canberra", "geography"),
    FactTriple("brazil", "capital", "Brasília", "geography"),
    FactTriple("russia", "capital", "Moscow", "geography"),
    FactTriple("egypt", "capital", "Cairo", "geography"),
    FactTriple("turkey", "capital", "Ankara", "geography"),
    FactTriple("south korea", "capital", "Seoul", "geography"),
    FactTriple("argentina", "capital", "Buenos Aires", "geography"),
    FactTriple("mexico", "capital", "Mexico City", "geography"),
    FactTriple("switzerland", "capital", "Bern", "geography"),
    FactTriple("sweden", "capital", "Stockholm", "geography"),
    FactTriple("norway", "capital", "Oslo", "geography"),
    FactTriple("netherlands", "capital", "Amsterdam", "geography"),
    FactTriple("greece", "capital", "Athens", "geography"),
    FactTriple("saudi arabia", "capital", "Riyadh", "geography"),
    FactTriple("united arab emirates", "capital", "Abu Dhabi", "geography"),
    FactTriple("uae", "capital", "Abu Dhabi", "geography"),

    # --- Currencies ---
    FactTriple("france", "currency", "Euro (€)", "economics"),
    FactTriple("germany", "currency", "Euro (€)", "economics"),
    FactTriple("italy", "currency", "Euro (€)", "economics"),
    FactTriple("spain", "currency", "Euro (€)", "economics"),
    FactTriple("united kingdom", "currency", "Pound Sterling (£)", "economics"),
    FactTriple("uk", "currency", "Pound Sterling (£)", "economics"),
    FactTriple("united states", "currency", "United States Dollar ($)", "economics"),
    FactTriple("usa", "currency", "United States Dollar ($)", "economics"),
    FactTriple("japan", "currency", "Japanese Yen (¥)", "economics"),
    FactTriple("china", "currency", "Chinese Yuan (Renminbi, ¥)", "economics"),
    FactTriple("india", "currency", "Indian Rupee (₹)", "economics"),
    FactTriple("pakistan", "currency", "Pakistani Rupee (₨)", "economics"),
    FactTriple("australia", "currency", "Australian Dollar (A$)", "economics"),
    FactTriple("canada", "currency", "Canadian Dollar (C$)", "economics"),
    FactTriple("switzerland", "currency", "Swiss Franc (CHF)", "economics"),
    FactTriple("saudi arabia", "currency", "Saudi Riyal (SAR)", "economics"),
    FactTriple("uae", "currency", "UAE Dirham (AED)", "economics"),

    # --- Continents ---
    FactTriple("france", "continent", "Europe", "geography"),
    FactTriple("germany", "continent", "Europe", "geography"),
    FactTriple("italy", "continent", "Europe", "geography"),
    FactTriple("japan", "continent", "Asia", "geography"),
    FactTriple("china", "continent", "Asia", "geography"),
    FactTriple("india", "continent", "Asia", "geography"),
    FactTriple("pakistan", "continent", "Asia", "geography"),
    FactTriple("brazil", "continent", "South America", "geography"),
    FactTriple("egypt", "continent", "Africa", "geography"),
    FactTriple("canada", "continent", "North America", "geography"),
    FactTriple("united states", "continent", "North America", "geography"),
    FactTriple("australia", "continent", "Oceania (Australia)", "geography"),

    # --- Literature & Arts ---
    FactTriple("romeo and juliet", "author", "William Shakespeare", "literature"),
    FactTriple("hamlet", "author", "William Shakespeare", "literature"),
    FactTriple("macbeth", "author", "William Shakespeare", "literature"),
    FactTriple("othello", "author", "William Shakespeare", "literature"),
    FactTriple("1984", "author", "George Orwell", "literature"),
    FactTriple("animal farm", "author", "George Orwell", "literature"),
    FactTriple("pride and prejudice", "author", "Jane Austen", "literature"),
    FactTriple("to kill a mockingbird", "author", "Harper Lee", "literature"),
    FactTriple("the great gatsby", "author", "F. Scott Fitzgerald", "literature"),
    FactTriple("the odyssey", "author", "Homer", "literature"),
    FactTriple("the iliad", "author", "Homer", "literature"),
    FactTriple("crime and punishment", "author", "Fyodor Dostoevsky", "literature"),
    FactTriple("mona lisa", "artist", "Leonardo da Vinci", "art"),
    FactTriple("the starry night", "artist", "Vincent van Gogh", "art"),
    FactTriple("the last supper", "artist", "Leonardo da Vinci", "art"),
    FactTriple("the scream", "artist", "Edvard Munch", "art"),

    # --- Science & Chemistry ---
    FactTriple("water", "chemical_formula", "H₂O", "chemistry"),
    FactTriple("carbon dioxide", "chemical_formula", "CO₂", "chemistry"),
    FactTriple("methane", "chemical_formula", "CH₄", "chemistry"),
    FactTriple("sodium chloride", "chemical_formula", "NaCl (table salt)", "chemistry"),
    FactTriple("gold", "chemical_symbol", "Au", "chemistry"),
    FactTriple("gold", "atomic_number", "79", "chemistry"),
    FactTriple("silver", "chemical_symbol", "Ag", "chemistry"),
    FactTriple("silver", "atomic_number", "47", "chemistry"),
    FactTriple("iron", "chemical_symbol", "Fe", "chemistry"),
    FactTriple("iron", "atomic_number", "26", "chemistry"),
    FactTriple("oxygen", "chemical_symbol", "O", "chemistry"),
    FactTriple("oxygen", "atomic_number", "8", "chemistry"),
    FactTriple("hydrogen", "chemical_symbol", "H", "chemistry"),
    FactTriple("hydrogen", "atomic_number", "1", "chemistry"),
    FactTriple("helium", "chemical_symbol", "He", "chemistry"),
    FactTriple("helium", "atomic_number", "2", "chemistry"),
    FactTriple("carbon", "chemical_symbol", "C", "chemistry"),
    FactTriple("carbon", "atomic_number", "6", "chemistry"),

    # --- Astronomy & Physics ---
    FactTriple("earth", "order_from_sun", "3rd planet from the Sun", "astronomy"),
    FactTriple("jupiter", "classification", "largest planet in the Solar System", "astronomy"),
    FactTriple("mercury", "order_from_sun", "1st planet from the Sun (closest to the Sun)", "astronomy"),
    FactTriple("venus", "order_from_sun", "2nd planet from the Sun", "astronomy"),
    FactTriple("mars", "order_from_sun", "4th planet from the Sun (the Red Planet)", "astronomy"),
    FactTriple("sun", "classification", "G-type main-sequence star (yellow dwarf)", "astronomy"),
    FactTriple("moon", "classification", "natural satellite of Earth", "astronomy"),
    FactTriple("light", "speed_in_vacuum", "299,792,458 meters per second", "physics"),
    FactTriple("earth gravity", "standard_acceleration", "9.80665 m/s²", "physics"),

    # --- Inventions & Discoveries ---
    FactTriple("telephone", "inventor", "Alexander Graham Bell", "history"),
    FactTriple("light bulb", "inventor", "Thomas Edison", "history"),
    FactTriple("world wide web", "inventor", "Tim Berners-Lee", "technology"),
    FactTriple("penicillin", "discoverer", "Alexander Fleming", "medicine"),
    FactTriple("theory of general relativity", "discoverer", "Albert Einstein", "physics"),
    FactTriple("laws of motion", "discoverer", "Sir Isaac Newton", "physics"),
    FactTriple("python programming language", "creator", "Guido van Rossum", "technology"),
    FactTriple("linux kernel", "creator", "Linus Torvalds", "technology"),
    FactTriple("git", "creator", "Linus Torvalds", "technology"),
    FactTriple("mirror 7", "founder", "Ahmad Hassan", "technology"),
    FactTriple("mirror 7", "license", "Apache-2.0", "technology"),
)

# Predicate synonym maps
_PREDICATE_SYNONYMS = {
    "capital": ("capital", "capital city", "seat of government"),
    "currency": ("currency", "money", "legal tender"),
    "continent": ("continent", "part of the world", "region"),
    "author": ("author", "writer", "written by", "wrote", "book author", "playwright"),
    "artist": ("artist", "painter", "painted by", "painted"),
    "inventor": ("inventor", "invented by", "invented", "creator"),
    "discoverer": ("discoverer", "discovered by", "discovered"),
    "creator": ("creator", "created by", "designer", "developed by"),
    "founder": ("founder", "founded by", "architect", "lead"),
    "chemical_symbol": ("chemical symbol", "element symbol", "symbol"),
    "chemical_formula": ("chemical formula", "formula"),
    "atomic_number": ("atomic number", "number of protons"),
    "license": ("license", "open source license"),
}


class KnowledgeGraphEngine:
    """SQLite-backed Knowledge Triples Engine supporting sub-millisecond factual lookups."""

    def __init__(self, db_path: Path | str | None = None) -> None:
        self._lock = threading.Lock()
        if db_path is None:
            db_path = Path("artifacts/mirror_knowledge_graph.sqlite")
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._init_schema()
        self._seed_default_triples()

    def _init_schema(self) -> None:
        with self._lock, self._conn:
            self._conn.executescript("""
                CREATE TABLE IF NOT EXISTS entity_triples (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    subject TEXT NOT NULL,
                    predicate TEXT NOT NULL,
                    object TEXT NOT NULL,
                    category TEXT NOT NULL,
                    confidence REAL DEFAULT 1.0,
                    provenance TEXT DEFAULT 'curated'
                );
                CREATE INDEX IF NOT EXISTS idx_triples_sub_pred ON entity_triples(subject, predicate);
                CREATE INDEX IF NOT EXISTS idx_triples_obj ON entity_triples(object);
                CREATE INDEX IF NOT EXISTS idx_triples_sub ON entity_triples(subject);
                CREATE INDEX IF NOT EXISTS idx_triples_pred ON entity_triples(predicate);
                CREATE UNIQUE INDEX IF NOT EXISTS idx_triples_unique ON entity_triples(subject, predicate, object);
            """)

    def _seed_default_triples(self) -> None:
        with self._lock, self._conn:
            cur = self._conn.cursor()
            count = cur.execute("SELECT count(*) FROM entity_triples").fetchone()[0]
            if count == 0:
                cur.executemany(
                    """
                    INSERT OR IGNORE INTO entity_triples (subject, predicate, object, category, confidence, provenance)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    [
                        (t.subject.casefold(), t.predicate.casefold(), t.object, t.category, t.confidence, t.provenance)
                        for t in _SEED_TRIPLES
                    ],
                )

    def insert_triple(
        self,
        subject: str,
        predicate: str,
        object_: str,
        category: str = "general",
        confidence: float = 1.0,
        provenance: str = "user_ingested",
    ) -> bool:
        s = clean_entity(subject).casefold()
        p = clean_entity(predicate).casefold()
        o = clean_entity(object_)
        if not s or not p or not o:
            return False
        with self._lock, self._conn:
            cur = self._conn.cursor()
            cur.execute(
                """
                INSERT OR REPLACE INTO entity_triples (subject, predicate, object, category, confidence, provenance)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (s, p, o, category, confidence, provenance),
            )
            return cur.rowcount > 0

    def query(self, subject: str, predicate: str) -> list[tuple[str, float]]:
        s = clean_entity(subject).casefold()
        p = clean_entity(predicate).casefold()
        with self._lock:
            cur = self._conn.cursor()
            rows = cur.execute(
                "SELECT object, confidence FROM entity_triples WHERE subject = ? AND predicate = ?",
                (s, p),
            ).fetchall()
            return [(r[0], float(r[1])) for r in rows]

    def query_by_object(self, object_: str, predicate: str | None = None) -> list[tuple[str, str]]:
        """Inverse query: find subjects that have this object (e.g. which country has capital Paris)."""
        o = clean_entity(object_)
        with self._lock:
            cur = self._conn.cursor()
            if predicate:
                rows = cur.execute(
                    "SELECT subject, predicate FROM entity_triples WHERE LOWER(object) = LOWER(?) AND predicate = ?",
                    (o, predicate.casefold()),
                ).fetchall()
            else:
                rows = cur.execute(
                    "SELECT subject, predicate FROM entity_triples WHERE LOWER(object) = LOWER(?)",
                    (o,),
                ).fetchall()
            return [(r[0], r[1]) for r in rows]

    def total_triples(self) -> int:
        with self._lock:
            return self._conn.execute("SELECT count(*) FROM entity_triples").fetchone()[0]

    # --- Natural Language Query Parser ---
    def lookup_from_prompt(self, prompt: str) -> NativeAnswer | None:
        text = normalize(prompt).strip(" ?!.").casefold()
        if len(text) > 200:
            return None

        # 1. Identify Target Predicate
        target_pred: str | None = None
        for canonical_pred, syns in _PREDICATE_SYNONYMS.items():
            for syn in syns:
                if re.search(r"\b" + re.escape(syn) + r"\b", text):
                    target_pred = canonical_pred
                    break
            if target_pred:
                break

        if not target_pred:
            return None

        # 2. Extract Candidate Subject
        # Strip question words and predicate markers
        stripped = re.sub(
            r"^(?:what(?:'s| is| are)?|who(?:'s| is| wrote| invented| created)?|which(?: city| country)? is|tell me|name the)\s+",
            "",
            text,
            flags=re.I,
        )
        stripped = re.sub(r"\b(?:capital of|currency of|continent of|chemical symbol of|symbol for|author of|artist of|inventor of)\b", "", stripped, flags=re.I)
        candidate_subject = clean_entity(stripped)

        # 3. Direct Query
        hits = self.query(candidate_subject, target_pred)
        if not hits:
            # Try fuzzy contains on subject
            with self._lock:
                cur = self._conn.cursor()
                rows = cur.execute(
                    "SELECT object, confidence, subject FROM entity_triples WHERE predicate = ? AND (? LIKE '%' || subject || '%' OR subject LIKE '%' || ? || '%') LIMIT 1",
                    (target_pred, candidate_subject, candidate_subject),
                ).fetchall()
                if rows:
                    hits = [(rows[0][0], float(rows[0][1]))]
                    candidate_subject = rows[0][2]

        if hits:
            obj_val, conf = hits[0]
            pred_display = target_pred.replace("_", " ")
            subj_display = candidate_subject.title()
            answer = f"The {pred_display} of {subj_display} is **{obj_val}**."
            if target_pred == "author":
                answer = f"{obj_val} wrote *{subj_display}*."
            elif target_pred in {"inventor", "creator", "discoverer"}:
                answer = f"{obj_val} is the {pred_display} of {subj_display}."

            return NativeAnswer(
                kind="knowledge_triple",
                answer=answer,
                qualification="curated_knowledge",
                confidence=conf,
                verified=True,
                metadata={
                    "subject": candidate_subject,
                    "predicate": target_pred,
                    "object": obj_val,
                    "engine": "entity_triples_sqlite",
                },
            )

        return None
