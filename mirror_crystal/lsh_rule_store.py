"""
LSHRuleIndex — Sub-millisecond Locality-Sensitive Hash + Inverted Salient Posting Store
with Surprise-Driven 1-Pass Learning and Multi-Hop Causal/Relational Rule Chaining.
"""
from __future__ import annotations

from array import array
from collections import defaultdict
from dataclasses import dataclass, field
import math
import re
from typing import Any

from .bit_signatures import ContextBitTable
from .phase2_cognitive_core import CoreferenceAndSenseTracker
from .slot_abstractor import SlotAbstractor


@dataclass
class CrystalRule:
    rule_id: int
    signature: int
    salient_tokens: tuple[str, ...]
    abstract_prompt: str
    abstract_response: str
    abstract_tool: str = ""
    num_slots: int = 0
    derived_expr: str = ""
    pass_count: int = 1
    fail_count: int = 0
    domain: str = "general"
    trust_tier: int = 2  # 1 = sandbox-verified axiom, 2 = curated dataset, 3 = local user / untrusted
    updated_seq: int = 1
    inv_sqrt_bits: float = field(init=False, repr=False)
    token_set: frozenset[str] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        bc = self.signature.bit_count()
        self.inv_sqrt_bits = (1.0 / math.sqrt(bc)) if bc > 0 else 0.0
        self.token_set = frozenset(self.salient_tokens)

    @property
    def active(self) -> bool:
        return self.fail_count <= self.pass_count

    @property
    def reliability(self) -> float:
        tier_boost = 1.04 if self.trust_tier == 1 else (1.0 if self.trust_tier == 2 else 0.94)
        return min(1.05, ((self.pass_count + 1.0) / (self.pass_count + 2.0 * self.fail_count + 1.0)) * tier_boost)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.rule_id,
            "sig": hex(self.signature),
            "toks": list(self.salient_tokens),
            "ap": self.abstract_prompt,
            "ar": self.abstract_response,
            "at": self.abstract_tool,
            "ns": self.num_slots,
            "de": self.derived_expr,
            "pc": self.pass_count,
            "fc": self.fail_count,
            "dom": self.domain,
            "tt": self.trust_tier,
            "seq": self.updated_seq,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> CrystalRule:
        return cls(
            rule_id=int(d["id"]),
            signature=int(d["sig"], 16),
            salient_tokens=tuple(d["toks"]),
            abstract_prompt=d["ap"],
            abstract_response=d["ar"],
            abstract_tool=d.get("at", ""),
            num_slots=int(d.get("ns", 0)),
            derived_expr=d.get("de", ""),
            pass_count=int(d.get("pc", 1)),
            fail_count=int(d.get("fc", 0)),
            domain=d.get("dom", "general"),
            trust_tier=int(d.get("tt", 2)),
            updated_seq=int(d.get("seq", 1)),
        )


_CAUSAL_FACT_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^\s*([a-z0-9_ -]{2,40}?)\s+(?:is a|are a|is an|belongs to|consists of)\s+([a-z0-9_ -]{2,45}?)[.!]?\s*$", re.I),
    re.compile(r"^\s*(?:if|when)\s+([a-z0-9_ -]{2,45}?)\s*,\s*(?:then\s+)?([a-z0-9_ -]{2,60}?)[.!]?\s*$", re.I),
    re.compile(r"^\s*([a-z0-9_ -]{2,40}?)\s+(?:causes|leads to|produces|implies|triggers)\s+([a-z0-9_ -]{2,60}?)[.!]?\s*$", re.I),
)


class LSHRuleIndex:
    """Hardware-friendly inverted bit & token-posting rule engine with multi-hop causal chaining."""

    def __init__(self, max_postings_scan: int = 64):
        self.max_postings_scan = int(max_postings_scan)
        self.rules: list[CrystalRule] = []
        self._seq_counter: int = 0
        # O(1) exact prompt lookup so streaming thousands of .jsonl rows never runs redundant searches
        self._exact_prompt_map: dict[str, int] = {}
        # Compact C unsigned-int arrays ('I' = 4 bytes per rule_id) for tiny RAM & CPU cache locality
        self._token_postings: dict[str, array] = defaultdict(lambda: array("I"))
        self._band_postings: dict[int, array] = defaultdict(lambda: array("I"))
        self._slot_postings: dict[int, array] = defaultdict(lambda: array("I"))
        # Directed concept graph for multi-hop LEGO rule chaining: head_norm -> list[(tail_norm, raw_statement)]
        self._causal_edges: dict[str, list[tuple[str, str]]] = defaultdict(list)
        self._causal_token_index: dict[str, list[str]] = defaultdict(list)

    @staticmethod
    def _lsh_bands(sig: int) -> list[int]:
        """Extract 16 compact band keys across the 2,048-bit signature for O(1) candidate lookup."""
        bands: list[int] = []
        for band_idx in range(16):
            chunk = (sig >> (band_idx * 128)) & 0xFFF
            if chunk != 0:
                bands.append((band_idx << 12) | chunk)
        return bands

    @staticmethod
    def _normalize_entity(text: str) -> str:
        t = re.sub(r"\b(a|an|the|any|every|all)\b", " ", text.lower())
        return re.sub(r"\s+", " ", t).strip(" .?!,")

    def observe_causal_fact(self, text: str) -> bool:
        """Extract chainable A -> B causal/taxonomic bridges from concise declarative facts."""
        s = text.strip()
        if len(s) > 140 or "\n" in s:
            return False
        for pat in _CAUSAL_FACT_PATTERNS:
            m = pat.match(s)
            if m:
                head = self._normalize_entity(m.group(1))
                tail = self._normalize_entity(m.group(2))
                if head and tail and head != tail:
                    is_new_head = head not in self._causal_edges
                    edge = (tail, s)
                    if edge not in self._causal_edges[head]:
                        self._causal_edges[head].append(edge)
                    if is_new_head:
                        for w in ContextBitTable.tokenize(head):
                            if len(w) > 1:
                                self._causal_token_index[w].append(head)
                    return True
        return False

    def chain_infer(self, query: str, bit_table: ContextBitTable, max_hops: int = 4) -> tuple[str, list[str]] | None:
        """O(1)-indexed multi-hop forward chaining across separate learned Locks."""
        if not self._causal_edges:
            return None
        q_norm = self._normalize_entity(query)
        q_expanded = bit_table.expand_content_tokens(bit_table.tokenize(query))

        candidate_heads: set[str] = set()
        for tok in q_expanded:
            for head in self._causal_token_index.get(tok, ()):
                candidate_heads.add(head)

        starts: list[str] = []
        for node in candidate_heads:
            node_toks = set(bit_table.tokenize(node))
            if node in q_norm or (node_toks and node_toks.issubset(q_expanded)):
                starts.append(node)

        for start_node in starts:
            visited = {start_node}
            frontier: list[tuple[str, list[str], list[str]]] = [(start_node, [start_node], [])]
            best_chain: tuple[list[str], list[str]] | None = None

            while frontier:
                curr, path_nodes, path_facts = frontier.pop(0)
                if len(path_facts) >= max_hops:
                    continue
                for nxt, raw_stmt in self._causal_edges.get(curr, ()):
                    if nxt in visited:
                        continue
                    visited.add(nxt)
                    new_nodes = path_nodes + [nxt]
                    new_facts = path_facts + [raw_stmt]
                    if len(new_facts) >= 2:
                        best_chain = (new_nodes, new_facts)
                    frontier.append((nxt, new_nodes, new_facts))

            if best_chain is not None:
                nodes, facts = best_chain
                conclusion = f"Therefore, {nodes[0]} -> {nodes[-1]}."
                return conclusion, facts
        return None

    def query_candidates(
        self,
        query_sig: int,
        query_tokens: list[str],
        bit_table: ContextBitTable,
        top_k: int = 5,
        required_slots: int | None = None,
    ) -> list[tuple[float, CrystalRule]]:
        if not self.rules:
            return []

        raw_q_set = set(query_tokens)
        expanded = bit_table.expand_content_tokens(query_tokens)

        if required_slots is not None and required_slots > 0:
            top_rids = list(self._slot_postings.get(required_slots, ())[:128])
        else:
            candidate_hits: dict[int, int] = defaultdict(int)
            max_df = max(4096, len(self.rules) // 5)
            # 1. Salient + synonym token inverted postings (weighted by IDF salience & literal vs synonym)
            for tok in expanded:
                plist = self._token_postings.get(tok)
                if plist is not None and len(plist) <= max_df:
                    w = max(1, int(4.0 * bit_table.token_salience(tok)))
                    if tok in raw_q_set:
                        w *= 2
                    for rid in plist:
                        candidate_hits[rid] += w

            # 2. LSH 2,048-bit band postings
            for bkey in self._lsh_bands(query_sig):
                plist = self._band_postings.get(bkey)
                if plist is not None and len(plist) <= 2048:
                    for rid in plist:
                        candidate_hits[rid] += 1

            if candidate_hits:
                top_rids = [
                    rid
                    for rid, _ in sorted(candidate_hits.items(), key=lambda kv: kv[1], reverse=True)[: self.max_postings_scan]
                ]
            else:
                top_rids = list(range(min(len(self.rules), self.max_postings_scan)))

        scored: list[tuple[float, CrystalRule]] = []
        q_set = expanded if expanded else raw_q_set
        q_bc = query_sig.bit_count()
        q_inv_sqrt = (1.0 / math.sqrt(q_bc)) if q_bc > 0 else 0.0
        sal_cache = {
            t: bit_table.token_salience(t) * (1.0 if t in raw_q_set else 0.68)
            for t in q_set
        }
        q_sal_total = sum(bit_table.token_salience(t) for t in raw_q_set) if raw_q_set else sum(sal_cache.values())

        for rid in top_rids:
            rule = self.rules[rid]
            if not rule.active:
                continue
            overlap_bits = (query_sig & rule.signature).bit_count()
            bit_sim = (overlap_bits * q_inv_sqrt * rule.inv_sqrt_bits) if overlap_bits else 0.0
            r_set = rule.token_set
            if q_set and r_set:
                inter = q_set & r_set
                if inter:
                    overlap_sal = sum(sal_cache[t] for t in inter)
                    extra_r = sum(bit_table.token_salience(t) for t in (r_set - q_set))
                    union_sal = max(overlap_sal, q_sal_total) + extra_r
                    lex_sim = overlap_sal / max(1e-6, union_sal)
                else:
                    lex_sim = 0.0
            else:
                lex_sim = 0.0
            # Penalize polarity mismatch (e.g. 'enable SSH' vs 'NEVER enable SSH')
            q_neg = (query_sig >> 2040) != 0
            r_neg = (rule.signature >> 2040) != 0
            polarity_mult = 0.45 if (q_neg != r_neg) else 1.0
            score = (0.50 * bit_sim + 0.50 * lex_sim) * rule.reliability * polarity_mult
            scored.append((score, rule))

        scored.sort(key=lambda item: (item[0], item[1].updated_seq), reverse=True)
        return scored[:top_k]

    def add_or_reinforce_rule(
        self,
        prompt: str,
        response: str,
        bit_table: ContextBitTable,
        tool: str = "",
        domain: str = "general",
        verified_bonus: int = 0,
        trust_tier: int = 2,
        surprise_threshold: float = 0.94,
    ) -> tuple[CrystalRule, bool]:
        """O(1) surprise check + trust-tier contradiction supersession + 1-pass crystallization."""
        if len(response) <= 120:
            self.observe_causal_fact(response)

        self._seq_counter += 1
        effective_tier = 1 if verified_bonus > 0 else int(trust_tier)
        abs_pair = SlotAbstractor.abstract_pair(prompt, response, tool=tool)
        sense_tags = CoreferenceAndSenseTracker.detect_sense_tags(abs_pair.abstract_prompt, context_hint=response)
        for st in sense_tags:
            bit_table.observe_text(st)
        sense_suffix = f" {' '.join(sorted(sense_tags))}" if sense_tags else ""
        norm_key = abs_pair.abstract_prompt.strip().lower() + sense_suffix
        existing_rid = self._exact_prompt_map.get(norm_key)
        if existing_rid is not None:
            existing_rule = self.rules[existing_rid]
            if existing_rule.abstract_response == abs_pair.abstract_response:
                existing_rule.pass_count += 1 + verified_bonus
                existing_rule.updated_seq = self._seq_counter
                if effective_tier < existing_rule.trust_tier:
                    existing_rule.trust_tier = effective_tier
                return existing_rule, False
            # Contradiction detected! Block low-trust poisoning; allow equal/higher trust supersession!
            if effective_tier > existing_rule.trust_tier:
                return existing_rule, False
            existing_rule.abstract_response = abs_pair.abstract_response
            existing_rule.abstract_tool = abs_pair.abstract_tool
            existing_rule.num_slots = abs_pair.num_slots
            existing_rule.derived_expr = abs_pair.derived_expr
            existing_rule.trust_tier = effective_tier
            existing_rule.updated_seq = self._seq_counter
            existing_rule.pass_count += 1 + verified_bonus
            if existing_rule.num_slots > 0 and existing_rid not in self._slot_postings[existing_rule.num_slots]:
                self._slot_postings[existing_rule.num_slots].append(existing_rid)
            return existing_rule, False

        sig_input = abs_pair.abstract_prompt + sense_suffix
        sig, chosen_toks = bit_table.encode_text_signature(sig_input)

        rid = len(self.rules)
        salient = tuple(sorted(set(chosen_toks), key=bit_table.token_salience, reverse=True)[:14])
        rule = CrystalRule(
            rule_id=rid,
            signature=sig,
            salient_tokens=salient,
            abstract_prompt=abs_pair.abstract_prompt,
            abstract_response=abs_pair.abstract_response,
            abstract_tool=abs_pair.abstract_tool,
            num_slots=abs_pair.num_slots,
            derived_expr=abs_pair.derived_expr,
            pass_count=1 + verified_bonus,
            fail_count=0,
            domain=domain,
            trust_tier=effective_tier,
            updated_seq=self._seq_counter,
        )
        self.rules.append(rule)
        self._exact_prompt_map[norm_key] = rid

        if rule.num_slots > 0:
            self._slot_postings[rule.num_slots].append(rid)
        for tok in salient:
            self._token_postings[tok].append(rid)
        for bkey in self._lsh_bands(sig):
            self._band_postings[bkey].append(rid)

        return rule, True

    def to_state_dict(self) -> dict[str, Any]:
        return {
            "rules": [r.to_dict() for r in self.rules],
            "causal_edges": {k: [list(e) for e in v] for k, v in self._causal_edges.items()},
        }

    @classmethod
    def from_state_dict(cls, data: dict[str, Any]) -> LSHRuleIndex:
        idx = cls()
        for rd in data.get("rules", []):
            rule = CrystalRule.from_dict(rd)
            rid = rule.rule_id
            idx.rules.append(rule)
            s_tags = sorted(t for t in rule.salient_tokens if t.startswith("sense_"))
            s_sfx = f" {' '.join(s_tags)}" if s_tags else ""
            idx._exact_prompt_map[rule.abstract_prompt.strip().lower() + s_sfx] = rid
            if rule.num_slots > 0:
                idx._slot_postings[rule.num_slots].append(rid)
            for tok in rule.salient_tokens:
                idx._token_postings[tok].append(rid)
            for bkey in idx._lsh_bands(rule.signature):
                idx._band_postings[bkey].append(rid)
        for k, edges in data.get("causal_edges", {}).items():
            idx._causal_edges[k] = [(e[0], e[1]) for e in edges]
            for w in ContextBitTable.tokenize(k):
                if len(w) > 1:
                    idx._causal_token_index[w].append(k)
        return idx
