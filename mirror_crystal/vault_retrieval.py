"""Relational and causal retrieval helpers for DiskCrystalVault."""
from __future__ import annotations

import json
import re

from .bit_signatures import ContextBitTable
from .lsh_rule_store import LSHRuleIndex


class VaultRetrievalMixin:

    def _try_multi_hop_relational(self, query: str, session_id: str = "default") -> tuple[str, list[str]] | None:
        q = query.strip()
        m_poss = self._PAT_POSSESSIVE.match(q)
        m_of = self._PAT_OF_THE.match(q)
        m_where = self._PAT_WHERE.match(q)

        hop1_q = None
        rel = None
        if m_poss:
            d = m_poss.groupdict()
            rel = d['rel'].strip()
            hop1_q = f"What is the {d['inner_rel'].strip()} of {d['subj'].strip()}?"
        elif m_of:
            d = m_of.groupdict()
            rel = d['rel'].strip()
            hop1_q = f"What is the {d['inner_rel'].strip()} of {d['subj'].strip()}?"
        elif m_where:
            d = m_where.groupdict()
            rel = d['rel'].strip()
            hop1_q = f"Where {d['verb'].strip()} {d['subj'].strip()}?"

        if not hop1_q:
            return None

        resp1 = self.respond(hop1_q, session_id=session_id)
        if resp1.confidence < 0.65 or not resp1.answer:
            return None

        ans1 = resp1.answer.strip()
        m_is = re.search(r"\b(?:is|was|are|were)\s+([A-Z][a-zA-Z0-9_ ]+?)(?:\.|\$|,|;)", ans1)
        if m_is:
            ent_cand = m_is.group(1).strip()
        elif len(ans1.split()) <= 4:
            ent_cand = ans1.rstrip(".").strip()
        else:
            return None

        hop2_q = f"What is the {rel} of {ent_cand}?"
        resp2 = self.respond(hop2_q, session_id=session_id)
        if resp2.confidence < 0.65 or not resp2.answer:
            return None

        ans2 = resp2.answer.strip()
        lead2 = ans2[0].lower() + ans2[1:] if ans2.startswith("The ") else ans2
        combined_ans = f"{ans1.rstrip('.')}, and {lead2}"
        return combined_ans, [f"Hop 1 ({hop1_q}) -> {ans1}", f"Hop 2 ({hop2_q}) -> {ans2}"]

    def _chain_infer_disk(self, query: str, max_hops: int = 4) -> tuple[str, list[str]] | None:
        q_norm = LSHRuleIndex._normalize_entity(query)
        q_expanded = self.bit_table.expand_content_tokens(self.bit_table.tokenize(query))
        if not q_expanded:
            return None

        cur = self.conn.cursor()
        toks = list(q_expanded)[:24]
        qmarks = ",".join("?" for _ in toks)
        candidate_heads: set[str] = set()
        for (heads_json,) in cur.execute(f"SELECT heads_json FROM causal_tok_idx WHERE tok IN ({qmarks})", toks):
            for h in json.loads(heads_json):
                candidate_heads.add(h)

        starts = [
            node
            for node in candidate_heads
            if node in q_norm or set(self.bit_table.tokenize(node)).issubset(q_expanded)
        ]
        for start_node in starts:
            visited = {start_node}
            frontier: list[tuple[str, list[str], list[str]]] = [(start_node, [start_node], [])]
            best_chain: tuple[list[str], list[str]] | None = None
            while frontier:
                curr, path_nodes, path_facts = frontier.pop(0)
                if len(path_facts) >= max_hops:
                    continue
                row_e = cur.execute("SELECT edges_json FROM causal_edges WHERE head = ?", (curr,)).fetchone()
                if not row_e:
                    continue
                for nxt, raw_stmt in json.loads(row_e[0]):
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
                return f"Therefore, {nodes[0]} -> {nodes[-1]}.", facts
        return None
