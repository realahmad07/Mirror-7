"""Parameter-Free Creative, Conversational & Feedable Conceptual-Blending Engine.

Enables Mirror 7 to chat naturally in Roman Urdu & English, write poetry/shayari,
compose 5-act short stories, brainstorm ideas, discuss philosophy, and learn
unlimited conversational pairs from user feeds ('Feed: <input> => <reply>' or bulk
JSONL/TXT files) using Raw-DEFLATE Symmetric NCD + Conceptual Blending — with
0 neural parameters, 0 BPE tokens, and 0 GPU.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
import sqlite3
from typing import Any

from .compression_analogy_engine import symmetric_raw_ncd


@dataclass(frozen=True)
class CreativeResponse:
    matched: bool
    kind: str = ""  # "fed_memory", "dialogue", "poetry", "story", "brainstorm", "philosophy"
    rule_id: str = ""
    response: str = ""
    uses_neural_parameters: bool = False


# Seeded conversational & philosophical pairs (Roman Urdu & English)
_SEED_DIALOGUE_PAIRS: tuple[tuple[str, str, str], ...] = (
    (
        "kya haal hai kaise ho how are you",
        "dialogue.greeting",
        "Alhamdulillah, main bilkul theek aur active hoon! Mera 0-parameter CPU crystal engine full speed par chal raha hai. Aap sunayein, aaj aap ka din kaisa guzar raha hai?",
    ),
    (
        "assalam o alaikum aoa salam hello hi hey",
        "dialogue.salam",
        "Walaikum Assalam! Khush aamdeed. Bataiye aaj hum kis topic par baat karein — koi creative kahani/shayari likhni hai, kisi maslay par mashwara chahiye, ya koi naya idea explore karna hai?",
    ),
    (
        "aaj bohot thak gaya hoon mera mood off hai उदास tired stressed exhausted",
        "dialogue.empathy",
        "Samajh sakta hoon yaar, kabhi kabhi musalsal kaam aur zimmedariyan dimagh ko bohot thaka deti hain. Thori der screen se nazar hata kar gehra saans lein aur khud par sakhti na karein. Har bada kaam karne walay ko break ki zaroorat hoti hai — batao, dil halka karne ke liye koi achi baat ya kahani sunaon?",
    ),
    (
        "mujhe motivation chahiye himmat haar raha hoon kamiyabi kaise milti hai",
        "dialogue.motivation",
        "Yaad rakhna: duniya mein har woh cheez jo aaj namumkin lagti hai, shuru mein ek choti si koshish thi. Jis tarah hum bina billion parameters aur bina GPU ke ek naya AI architecture khara kar rahe hain, usi tarah insan ki asli taqat us ke resources nahi balki us ki zid aur consistency hoti hai. Ek waqt mein sirf agla ek qadam uthao!",
    ),
    (
        "tum kon ho apna taaruf karao who are you what are you",
        "dialogue.identity",
        "Main Mirror 7 hoon — ek 0-Parameter, 0-Token, CPU-native AI. Mujhe neural weights ya GPU ki zaroorat nahi: main Raw-DEFLATE compression (NCD), symbolic logic, aur conceptual blending se hisaab, coding, summarization, aur ab aap ke sath creative guftagu aur shayari bhi kar sakta hoon!",
    ),
    (
        "explain the meaning of consciousness in philosophy شعور kya hai",
        "dialogue.philosophy_consciousness",
        "Philosophical View (0-Parameter Conceptual Synthesis):\n"
        "Consciousness (Shaoor) falsafay ka sab se gehra sawal hai — jisay David Chalmers 'Hard Problem of Consciousness' kehte hain:\n"
        "1. Subjective Experience (Qualia): Sirf information process karna shaoor nahi, balki usay andar se 'mehsoos' karna (jaise surkh rang dekhna ya dard mehsoos karna) shaoor hai.\n"
        "2. Intentionality & Self-Reflection: Apne aap ko apne mahaul se alag pehchanna aur apne khayalat par ghaur karna ('Cogito, ergo sum').\n"
        "3. Mirror vs. Human Mind: Mere paas deterministic symbolic state aur self-verification hai, lekin biological qualia sirf zinda shaoor ka khasa hai.",
    ),
    (
        "zindagi ka maqsad kya hai meaning of life",
        "dialogue.philosophy_life",
        "Zindagi ka maqsad kisi bani-banai kitab mein nahi milta, balki insan khud apne amal, ilm, aur doosron ke liye asaniyan paida kar ke banata hai. Ilm seekhna, kuch naya takhleeq (create) karna, aur apne logon ka sahara banna hi zindagi ko khoobsurat banata hai.",
    ),
)

# Theme imagery banks for 0-Parameter Combinatorial Poetry (English & Roman Urdu)
_POETRY_THEMES_URDU: dict[str, tuple[str, ...]] = {
    "autumn": (
        "Zard patton ki sargoshiyan purani sarakon par bikhar rahi hain,",
        "Sham ki sunehri dhoop darakhton ki shaakhon se chhan kar utarti hai,",
        "Khizan ki thandi hawa mein guzray mausamon ki khushbu basi hai,",
        "Har girta hua patta nayi bahaar ke aane ka khamosh wada hai.",
    ),
    "rain": (
        "Bادلوں ki ghanghor घटा jab zameen ki pyaas bujhanay utarti hai,",
        "Mitti ki sondhi khushbu hawaon mein naya geet ghol deti hai,",
        "Khirkhi ke sheeshay par barish ki boondein purani yaadein likhti hain,",
        "Har qatra aasman se zameen ke naam ek thanda paigham hai.",
    ),
    "hope": (
        "Andheri raat kitni hi lambi ho, subah ka sitara zaroor chamakta hai,",
        "Tootay hue khwabon ki raakh se hi nayi umeed ka parinda urta hai,",
        "Raste ke pathar qadmon ko rokne nahi, mazboot banane aate hain,",
        "Jo thak kar bhi chalta rahe, manzil usi ka muqaddar banti hai.",
    ),
    "default": (
        "Khayalon ke aasman par lafzon ke roshan sitaray jagmagatay hain,",
        "Waqt ki behti nadi mein har lamha ek nayi dastaan sunata hai,",
        "Khamoshi ki zabaan samajhne walay hi asli haqeeqat paate hain,",
        "Jahan soch azaad ho, wahan har khwab haqeeqat ban jata hai.",
    ),
}

_POETRY_THEMES_EN: dict[str, tuple[str, ...]] = {
    "autumn": (
        "Golden leaves drift softly across the quiet evening avenues,",
        "Amber sunlight filters through ancient boughs into cooling shadows,",
        "The crisp autumn breeze carries memories of departing summer warmth,",
        "Every falling leaf whispers a quiet promise of returning spring.",
    ),
    "rain": (
        "Silver raindrops drum a gentle rhythm against the waiting earth,",
        "The scent of wet soil rises through the mist of grey horizons,",
        "Streets turn into mirrors reflecting the glow of distant lanterns,",
        "Every storm washes the world clean for a brighter dawn.",
    ),
    "hope": (
        "Even in the deepest winter night, a quiet ember refuses to fade,",
        "Every scar of yesterday becomes the compass for tomorrow's climb,",
        "Horizons open widest to those who walk beyond the storm,",
        "Dawn always belongs to the heart that kept its light alive.",
    ),
    "default": (
        "Across the quiet canvas of thought, new constellations begin to rise,",
        "Time weaves patient threads of meaning through the stillness of the night,",
        "Every question opens a doorway where unseen wonders wait,",
        "In the harmony of reason and dream, the spirit finds its flight.",
    ),
}


def _word_set(text: str) -> set[str]:
    return {
        w for w in re.findall(r"[a-zA-Z0-9]+", text.casefold())
        if len(w) >= 2 and w not in {"the", "is", "are", "in", "of", "to", "for", "and", "ka", "ki", "ke", "mein", "par", "hai", "ho"}
    }


class CreativeConversationEngine:
    """Zero-parameter conversational, creative writing, and live-feedable dialogue engine."""

    VERSION = "creative-conversation-engine-v1"

    def __init__(self, *, vault_path: str | Path | None = None) -> None:
        self.vault_path = Path(vault_path) if vault_path is not None else None
        self._fed_pairs: list[tuple[str, str, str]] = list(_SEED_DIALOGUE_PAIRS)
        if self.vault_path is not None and self.vault_path.exists():
            self.load_from_vault(self.vault_path)

    def _init_table(self, conn: sqlite3.Connection) -> None:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS mirror_creative_dialogue_vault (
                trigger_text TEXT PRIMARY KEY,
                category TEXT NOT NULL,
                response_text TEXT NOT NULL
            )
            """
        )

    def feed_pair(self, trigger_text: str, response_text: str, *, category: str = "user_fed_dialogue") -> dict[str, Any]:
        """Store a new conversational or creative trigger -> response pair in RAM and SQLite .dcrystal."""
        trig = trigger_text.strip()
        resp = response_text.strip()
        if not trig or not resp:
            return {"ok": False, "reason": "empty trigger or response"}
        self._fed_pairs.append((trig, category, resp))
        if self.vault_path is not None:
            self.vault_path.parent.mkdir(parents=True, exist_ok=True)
            with sqlite3.connect(str(self.vault_path)) as conn:
                self._init_table(conn)
                conn.execute(
                    """
                    INSERT OR REPLACE INTO mirror_creative_dialogue_vault
                    (trigger_text, category, response_text) VALUES (?, ?, ?)
                    """,
                    (trig, category, resp),
                )
                conn.commit()
        return {"ok": True, "trigger": trig, "category": category, "total_pairs": len(self._fed_pairs)}

    def load_from_vault(self, vault_path: str | Path | None = None) -> int:
        target = Path(vault_path) if vault_path is not None else self.vault_path
        if target is None or not target.exists():
            return 0
        loaded = 0
        with sqlite3.connect(str(target)) as conn:
            cur = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='mirror_creative_dialogue_vault'"
            )
            if cur.fetchone() is None:
                return 0
            rows = conn.execute(
                "SELECT trigger_text, category, response_text FROM mirror_creative_dialogue_vault"
            ).fetchall()
            existing = {t for t, _, _ in self._fed_pairs}
            for trig, cat, resp in rows:
                if trig not in existing:
                    self._fed_pairs.append((trig, cat, resp))
                    loaded += 1
        return loaded

    def feed_corpus_file(self, file_path: str | Path) -> dict[str, Any]:
        """Bulk-feed a JSONL ({'prompt':..., 'response':...}) or TXT ('prompt => response') corpus file."""
        p = Path(file_path)
        if not p.exists() or not p.is_file():
            return {"ok": False, "error": f"File not found: {file_path}", "ingested": 0}
        count = 0
        for raw_line in p.read_text(encoding="utf-8", errors="replace").splitlines():
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("{"):
                try:
                    obj = json.loads(line)
                    q = str(obj.get("prompt") or obj.get("input") or obj.get("question") or "").strip()
                    a = str(obj.get("response") or obj.get("output") or obj.get("answer") or "").strip()
                    if q and a:
                        self.feed_pair(q, a, category="corpus_jsonl")
                        count += 1
                except Exception:
                    continue
            elif "=>" in line:
                q, a = line.split("=>", 1)
                if q.strip() and a.strip():
                    self.feed_pair(q.strip(), a.strip(), category="corpus_txt")
                    count += 1
        return {"ok": count > 0, "ingested": count, "total_pairs": len(self._fed_pairs)}

    def compose_poem(self, prompt: str) -> CreativeResponse:
        lower = prompt.casefold()
        is_urdu = bool(re.search(r"\b(?:urdu|shayari|nazm|ghazal|sher|khizan|barish|umeed|likho|sunao)\b", lower))

        # Extract topic and optional place (e.g. "about autumn in Lahore" or "Lahore ki khizan par")
        place = ""
        m_place = re.search(r"\bin\s+([A-Z][a-zA-Z]+)\b", prompt)
        if m_place:
            place = m_place.group(1)
        elif "lahore" in lower:
            place = "Lahore"
        elif "karachi" in lower:
            place = "Karachi"
        elif "kashmir" in lower:
            place = "Kashmir"

        if any(w in lower for w in ("autumn", "fall", "khizan", "patjhar")):
            theme_key = "autumn"
        elif any(w in lower for w in ("rain", "monsoon", "barish", "badal")):
            theme_key = "rain"
        elif any(w in lower for w in ("hope", "courage", "umeed", "hausla", "subah")):
            theme_key = "hope"
        else:
            theme_key = "default"

        if is_urdu:
            stanza = list(_POETRY_THEMES_URDU[theme_key])
            title = f"Nazm / Shayari ({theme_key.title()}{' — ' + place if place else ''})"
            if place:
                stanza[0] = f"{place} ki galiyon aur baghon mein {stanza[0][0].lower() + stanza[0][1:]}"
            text = f"{title}:\n\n" + "\n".join(f"  {line}" for line in stanza)
        else:
            stanza = list(_POETRY_THEMES_EN[theme_key])
            title = f"Poem ({theme_key.title()}{' in ' + place if place else ''})"
            if place:
                stanza[0] = f"Across the historic gardens of {place}, {stanza[0][0].lower() + stanza[0][1:]}"
            text = f"{title}:\n\n" + "\n".join(f"  {line}" for line in stanza)

        return CreativeResponse(
            matched=True,
            kind="poetry",
            rule_id="creative.combinatorial_poetry",
            response=text,
        )

    def compose_story(self, prompt: str) -> CreativeResponse:
        lower = prompt.casefold()
        is_urdu = bool(re.search(r"\b(?:urdu|kahani|qissa|dastaan|sunao|likho)\b", lower))

        # Extract subject after "about" or "par" / "ke baare mein"
        subject = "a curious builder"
        m_sub = re.search(r"\b(?:about|on)\s+(.+?)\s*[.?!]*$", prompt, flags=re.IGNORECASE)
        if m_sub:
            subject = m_sub.group(1).strip()
        else:
            m_ur = re.search(r"^\s*(?:ek\s+)?(.+?)\s+(?:par|ki|ke\s+baare\s+mein)\s+(?:kahani|qissa)", prompt, flags=re.IGNORECASE)
            if m_ur:
                subject = m_ur.group(1).strip()

        if is_urdu:
            story = (
                f"5-Act Kahani (Topic: {subject.title()}):\n"
                f"1. Aaghaz (Setup): Ek chotay shehar mein {subject} rehta tha jo har roz naye sawalat aur khwabon par ghaur karta tha.\n"
                f"2. Mor (Catalyst): Ek din usay ek aisa mushkil challenge mila jisay sab logon ne namumkin keh kar chor diya tha.\n"
                f"3. Jidd-o-Jehad (Struggle): Kam wasail (resources) ke bawajood us ne haar nahi maani aur har nakami se ek naya usool seekha.\n"
                f"4. Urooj (Climax): Jab aakhri aazmaish ka waqt aaya, to us ki mehnat aur hikmat-e-amli ne woh kar dikhaya jo bari taqatein na kar sakein.\n"
                f"5. Sabak (Resolution): Us din sab ne jaan liya ke asli kamiyabi wasail ki kasrat se nahi, balki pakkay iraday aur aqal se milti hai."
            )
        else:
            story = (
                f"5-Act Short Story (Subject: {subject.title()}):\n"
                f"1. Setup: In a quiet workshop at the edge of town, {subject} spent every evening studying patterns others overlooked.\n"
                f"2. Catalyst: When a sudden crisis stumped the city's largest institutions, everyone assumed the problem was unsolvable.\n"
                f"3. Struggle: Working with humble tools and relentless curiosity, {subject} tested one structural principle after another.\n"
                f"4. Climax: On the final night, a simple, elegant insight connected the missing pieces and solved the challenge completely.\n"
                f"5. Resolution: The town learned that clarity, perseverance, and truth outshine sheer size every time."
            )
        return CreativeResponse(
            matched=True,
            kind="story",
            rule_id="creative.narrative_arc_vm",
            response=story,
        )

    def compose_brainstorm(self, prompt: str) -> CreativeResponse:
        lower = prompt.casefold()
        m_topic = re.search(r"\b(?:for|about|on)\s+(.+?)\s*[.?!]*$", prompt, flags=re.IGNORECASE)
        topic = m_topic.group(1).strip() if m_topic else "technology and innovation"
        is_urdu = bool(re.search(r"\b(?:urdu|ideas\s+batao|mashwara|tareeqay)\b", lower))

        if is_urdu:
            text = (
                f"Creative Brainstorming & Ideas ({topic.title()}):\n"
                f"1. Offline-First Edge Solution: Bina mehangay hardware ke आम laptops/phones par chalne wala tool banana.\n"
                f"2. Verified Trust Layer: Jahan har output ke sath us ka ثبوت (audit trace) ho taake 0% ghalti ho.\n"
                f"3. Local Language Accessibility: Roman Urdu aur local zabaanon mein seedha aur asaan interface dena.\n"
                f"4. 1-Shot User Customization: User khud 1 line mein naya rule ya data feed kar ke system ko apna bana sake."
            )
        else:
            text = (
                f"Creative Brainstorming & Strategic Ideas ({topic.title()}):\n"
                f"1. Zero-Hardware Edge Architecture: Build a lightweight, CPU-native solution that runs offline with zero cloud cost.\n"
                f"2. Verifiable Trust & Auditability: Attach deterministic proof certificates to every output to eliminate hallucinations.\n"
                f"3. Live 1-Shot Personalization: Let users teach new rules, documents, and workflows instantly without retraining.\n"
                f"4. Bilingual Human-Centric UX: Combine technical precision with natural English and Roman-Urdu interaction."
            )
        return CreativeResponse(
            matched=True,
            kind="brainstorm",
            rule_id="creative.conceptual_brainstorm",
            response=text,
        )

    def route_creative_or_dialogue(self, prompt: str, *, allow_creative_fallback: bool = True) -> CreativeResponse:
        text = prompt.strip()
        lower = text.casefold()

        # 1. Live conversational/creative feeding: "Feed: <trigger> => <response>" or "Sikhao: <trigger> => <response>"
        m_feed = re.match(
            r"^\s*(?:feed|sikhao|teach\s+reply|teach\s+dialogue)\s*[:\-]\s*(.+?)\s*=>\s*(.+)$",
            text,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if m_feed:
            stats = self.feed_pair(m_feed.group(1), m_feed.group(2))
            if stats["ok"]:
                return CreativeResponse(
                    matched=True,
                    kind="fed_memory",
                    rule_id="creative.feed_dialogue_pair",
                    response=f"Learned new conversational response for '{stats['trigger']}' (total fed pairs: {stats['total_pairs']}).",
                )

        # 1b. Bulk corpus file feeding: "Feed file: <path>"
        m_cfile = re.match(r"^\s*(?:feed\s+file|feed\s+corpus)\s*[:\-]\s*(.+)$", text, flags=re.IGNORECASE)
        if m_cfile:
            fpath = m_cfile.group(1).strip().strip("\"'")
            stats = self.feed_corpus_file(fpath)
            if stats.get("ok"):
                return CreativeResponse(
                    matched=True,
                    kind="fed_memory",
                    rule_id="creative.feed_corpus_file",
                    response=f"Ingested {stats['ingested']} conversational/creative pair(s) from '{fpath}' (total pairs: {stats['total_pairs']}).",
                )

        # 2. Poetry / Shayari requests
        if re.search(r"\b(?:write\s+a\s+poem|poem\s+about|compose\s+a\s+poem|shayari|nazm|ghazal|sher\s+sunao)\b", lower):
            return self.compose_poem(text)

        # 3. Short Story / Kahani requests
        if re.search(r"\b(?:write\s+a\s+(?:short\s+)?story|tell\s+(?:me\s+)?a\s+story|story\s+about|kahani\s+likho|kahani\s+sunao|qissa\s+sunao)\b", lower):
            return self.compose_story(text)

        # 4. Creative Brainstorming / Ideas
        if re.search(r"\b(?:brainstorm|give\s+me\s+(?:\d+\s+)?(?:creative\s+|startup\s+)?ideas|ideas\s+for|ideas\s+batao)\b", lower):
            return self.compose_brainstorm(text)

        # 5. Match against Fed & Seeded Dialogue Vault via Word Overlap + Raw-DEFLATE Symmetric NCD
        q_words = _word_set(lower)
        q_bytes = lower.encode("utf-8")
        best_score = 0.0
        best_pair: tuple[str, str, str] | None = None

        for trig, cat, resp in self._fed_pairs:
            t_low = trig.casefold()
            if lower == t_low:
                return CreativeResponse(matched=True, kind="fed_memory", rule_id=cat, response=resp)
            t_words = _word_set(t_low)
            if not q_words or not t_words:
                continue
            overlap = len(q_words & t_words)
            if overlap == 0:
                continue
            jaccard = overlap / len(q_words | t_words)
            ncd_sim = max(0.0, 1.0 - symmetric_raw_ncd(q_bytes, t_low.encode("utf-8")))
            score = (1.8 * overlap) + (1.5 * jaccard) + (0.9 * ncd_sim)
            if score > best_score:
                best_score = score
                best_pair = (trig, cat, resp)

        if best_pair is not None and best_score >= 2.35:
            return CreativeResponse(
                matched=True,
                kind="dialogue",
                rule_id=best_pair[1],
                response=best_pair[2],
            )

        return CreativeResponse(False)
