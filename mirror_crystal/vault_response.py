"""Response routing and generation policy for DiskCrystalVault."""
from __future__ import annotations

import re
import time

from .bit_signatures import ContextBitTable
from .phase2_cognitive_core import EpistemicConfidenceCalibrator
from .grammar_and_sandbox import ZeroTrustSandboxJudge
from .intent import split_compound_intents
from .phase5_bicameral_and_cas import SymbolicCASOrgan
from .response_types import CrystalResponse
from .slot_abstractor import SlotAbstractor


class VaultResponseMixin:

    def respond(self, prompt: str, session_id: str = "default") -> CrystalResponse:
        """Out-of-core inference with UniversalConceptHub, CAS, Curiosity, Coreference & <= 64 disk reads."""
        with self._lock:
            return self._respond_locked(prompt, session_id=session_id)

    def _respond_locked(self, prompt: str, session_id: str = "default") -> CrystalResponse:
        t0 = time.perf_counter_ns()
        q_raw = prompt.strip()
        if not q_raw:
            return CrystalResponse(
                answer="No verified Crystal Rule matched above threshold yet — feed one example to crystallize instantly.",
                confidence=0.0,
                mode="unmatched_honest_abstain",
                matched_rule_id=None,
                latency_ms=0.0,
                sandbox_verified=False,
                reasoning_trace=["Empty prompt abstained cleanly."],
                epistemic_tier="honest_abstain",
            )

        q_canon, detected_lang, mapped_concepts = self.concept_hub.normalize_to_canonical(q_raw)
        pre_trace: list[str] = []
        if mapped_concepts:
            pre_trace.append(f"UniversalConceptHub ({detected_lang}): {', '.join(mapped_concepts[:6])}")

        q_canon, clarified_this_turn = self.curiosity_planner.merge_followup_if_pending(session_id, q_canon)
        if clarified_this_turn:
            pre_trace.append(f"ActiveCuriosityPlanner fused follow-up choice -> '{q_canon}'")

        cas_hit = SymbolicCASOrgan.try_solve(q_canon)
        if cas_hit is not None:
            ms = (time.perf_counter_ns() - t0) / 1_000_000.0
            return CrystalResponse(
                answer=cas_hit.answer,
                confidence=1.0 if cas_hit.verified else 0.9,
                mode=cas_hit.mode,
                matched_rule_id=None,
                latency_ms=round(ms, 3),
                sandbox_verified=cas_hit.verified,
                reasoning_trace=pre_trace + [cas_hit.trace],
                epistemic_tier="verified_exact",
            )

        self.sessions.update_from_prompt(session_id, q_canon)
        constraint_hit = self.sessions.evaluate_constraint_check(session_id, q_canon)
        if constraint_hit is not None:
            ans_c, trace_c = constraint_hit
            ok_c, _ = ZeroTrustSandboxJudge.verify_code_and_test(ans_c)
            ms = (time.perf_counter_ns() - t0) / 1_000_000.0
            return CrystalResponse(
                answer=ans_c,
                confidence=1.0,
                mode="session_constraint_lock",
                matched_rule_id=None,
                latency_ms=round(ms, 3),
                sandbox_verified=ok_c,
                reasoning_trace=pre_trace + trace_c,
                epistemic_tier="verified_exact",
            )

        q, coref_trace = self.coref_tracker.resolve_turn(session_id, q_canon)
        pre_trace.extend(coref_trace)

        sub_intents = split_compound_intents(q)
        if len(sub_intents) >= 2:
            sub_res = [self._respond_atomic_disk(part, detected_lang=detected_lang) for part in sub_intents]
            valid_subs = [r for r in sub_res if r.mode != "unmatched_honest_abstain" and r.confidence >= 0.30]
            if len(valid_subs) >= 2:
                combined_ans = "\n\n".join(f"[{idx + 1}] {r.answer}" for idx, r in enumerate(valid_subs))
                avg_conf = round(sum(r.confidence for r in valid_subs) / len(valid_subs), 4)
                all_ver = all(r.sandbox_verified for r in valid_subs)
                ms = (time.perf_counter_ns() - t0) / 1_000_000.0
                return CrystalResponse(
                    answer=combined_ans,
                    confidence=avg_conf,
                    mode="compound_multi_intent",
                    matched_rule_id=valid_subs[0].matched_rule_id,
                    latency_ms=round(ms, 3),
                    sandbox_verified=all_ver,
                    reasoning_trace=pre_trace + [tr for r in valid_subs for tr in r.reasoning_trace],
                    epistemic_tier=EpistemicConfidenceCalibrator.classify_tier(avg_conf, "compound_multi_intent", all_ver),
                )

        windows = ContextBitTable.split_focus_windows(q)
        if len(windows) > 1:
            best_win: CrystalResponse | None = None
            for win in windows:
                cand_r = self._respond_atomic_disk(win, detected_lang=detected_lang)
                if best_win is None or cand_r.confidence > best_win.confidence:
                    best_win = cand_r
            if best_win is not None and best_win.mode != "unmatched_honest_abstain":
                ms = (time.perf_counter_ns() - t0) / 1_000_000.0
                return CrystalResponse(
                    answer=best_win.answer,
                    confidence=best_win.confidence,
                    mode=best_win.mode,
                    matched_rule_id=best_win.matched_rule_id,
                    latency_ms=round(ms, 3),
                    sandbox_verified=best_win.sandbox_verified,
                    reasoning_trace=pre_trace + ["Focused long-log diagnostic window"] + best_win.reasoning_trace,
                    epistemic_tier=best_win.epistemic_tier,
                )

        return self._respond_atomic_disk(
            q,
            pre_trace=pre_trace,
            session_id=session_id,
            clarified_this_turn=clarified_this_turn,
            original_prompt=q_raw,
            detected_lang=detected_lang,
        )

    def _respond_atomic_disk(
        self,
        q: str,
        pre_trace: list[str] | None = None,
        session_id: str = "default",
        clarified_this_turn: bool = False,
        original_prompt: str = "",
        detected_lang: str = "en",
    ) -> CrystalResponse:
        t0 = time.perf_counter_ns()
        trace: list[str] = list(pre_trace) if pre_trace else []
        orig_p = original_prompt or q

        q_core = self.fluid_composer.strip_style_framing(q)
        q_sense, sense_tags = self.coref_tracker.enrich_for_query(q_core, known_vocab=self.bit_table.token_df)
        if sense_tags:
            trace.append(f"Disambiguated homonym sense -> {', '.join(sense_tags)}")
        sense_sfx = f" {' '.join(sorted(sense_tags))}" if sense_tags else ""

        if not self.fluid_composer.is_explicit_creative_request(orig_p):
            abs_pair_exact = SlotAbstractor.abstract_pair(q_core, "")
            candidate_hashes = [self._prompt_hash(abs_pair_exact.abstract_prompt.strip().lower() + sense_sfx)]
            if sense_sfx:
                candidate_hashes.append(self._prompt_hash(abs_pair_exact.abstract_prompt.strip().lower()))

            for phash_exact in candidate_hashes:
                rule_exact = None
                cached_rid = self._prompt_hash_lru.get(phash_exact)
                if cached_rid is not None:
                    self._prompt_hash_lru.move_to_end(phash_exact)
                    rule_exact = self._rule_lru.get(cached_rid)

                if rule_exact is None:
                    row_exact = self.conn.execute(
                        "SELECT rule_id, sig_bytes, num_slots, pass_count, fail_count, payload FROM rules WHERE prompt_hash = ? ORDER BY pass_count DESC, rule_id DESC",
                        (phash_exact,),
                    ).fetchone()
                    if row_exact is not None:
                        rule_exact = self._row_to_rule(row_exact)
                        self._prompt_hash_lru[phash_exact] = rule_exact.rule_id
                        if len(self._prompt_hash_lru) > self._max_prompt_hash_cache:
                            self._prompt_hash_lru.popitem(last=False)
                    else:
                        row_alias = self.conn.execute(
                            "SELECT rule_id FROM prompt_aliases WHERE alias_hash = ?",
                            (phash_exact,),
                        ).fetchone()
                        if row_alias is not None:
                            rule_exact = self.get_rule(int(row_alias[0]))
                            if rule_exact is not None:
                                self._prompt_hash_lru[phash_exact] = rule_exact.rule_id
                if rule_exact is not None and rule_exact.active:
                        instantiated = SlotAbstractor.instantiate(
                            query=q_core,
                            num_slots=rule_exact.num_slots,
                            abstract_response=rule_exact.abstract_response,
                            derived_expr=rule_exact.derived_expr,
                        )
                        if instantiated is not None:
                            ok_ex, j_msg = ZeroTrustSandboxJudge.verify_code_and_test(instantiated)
                            if ok_ex:
                                ms = (time.perf_counter_ns() - t0) / 1_000_000.0
                                mode_str = ("slot_formula_lock" if rule_exact.num_slots > 0 and rule_exact.derived_expr else ("slot_template_lock" if rule_exact.num_slots > 0 else "direct_crystal_lock"))
                                trace.append(f"O(1) Exact Prompt Hash Hit #{rule_exact.rule_id} (domain={rule_exact.domain})")
                                return CrystalResponse(
                                    answer=instantiated,
                                    confidence=1.0,
                                    mode=mode_str,
                                    matched_rule_id=rule_exact.rule_id,
                                    latency_ms=round(ms, 3),
                                    sandbox_verified=j_msg != "non_code",
                                    reasoning_trace=trace,
                                    epistemic_tier=EpistemicConfidenceCalibrator.classify_tier(1.0, mode_str, j_msg != "non_code", query_coverage=1.0),
                                )

        raw_sig, raw_toks = self.bit_table.encode_text_signature(q_sense)
        cands_raw = self._query_disk_candidates(raw_sig, raw_toks, top_k=4)

        masked_q = SlotAbstractor.mask_query_numbers(q_core)
        if masked_q != q_core:
            q_unique_nums = list(dict.fromkeys(SlotAbstractor.extract_numbers(q_core)))
            slot_sig, slot_toks = self.bit_table.encode_text_signature(masked_q)
            cands_slot = self._query_disk_candidates(
                slot_sig, slot_toks, top_k=4, required_slots=len(q_unique_nums)
            )
            merged_by_id: dict[int, tuple[float, CrystalRule]] = {}
            for sc, rl in cands_raw + cands_slot:
                prev = merged_by_id.get(rl.rule_id)
                if prev is None or sc > prev[0]:
                    merged_by_id[rl.rule_id] = (sc, rl)
            cands = sorted(merged_by_id.values(), key=lambda kv: kv[0], reverse=True)[:4]
        else:
            cands = cands_raw

        # Multi-Hop Relational Knowledge Chaining
        relational_hop = self._try_multi_hop_relational(q_core, session_id=session_id)
        if relational_hop is not None:
            conclusion, hop_facts = relational_hop
            ms = (time.perf_counter_ns() - t0) / 1_000_000.0
            return CrystalResponse(
                answer=conclusion,
                confidence=0.92,
                mode="multi_hop_relational_chain",
                matched_rule_id=None,
                latency_ms=round(ms, 3),
                sandbox_verified=False,
                reasoning_trace=trace + hop_facts,
                epistemic_tier="moderate_confidence",
            )

        best_score = cands[0][0] if cands else 0.0
        if best_score < 0.72:
            chained = self._chain_infer_disk(q_core)
            if chained is not None:
                conclusion, hop_facts = chained
                ms = (time.perf_counter_ns() - t0) / 1_000_000.0
                return CrystalResponse(
                    answer=f"{conclusion} (Chained from: {' ; '.join(hop_facts)})",
                    confidence=0.92,
                    mode="multi_hop_rule_chain",
                    matched_rule_id=None,
                    latency_ms=round(ms, 3),
                    sandbox_verified=False,
                    reasoning_trace=trace + hop_facts,
                    epistemic_tier="moderate_confidence",
                )

        is_coding_req = (
            EpistemicConfidenceCalibrator.is_coding_intent(q_core, sense_tags=sense_tags)
            or EpistemicConfidenceCalibrator.is_coding_intent(orig_p, sense_tags=sense_tags)
            or any(c[1].domain.lower() in ("algorithms", "algorithm", "code", "programming", "python", "leetcode") for c in cands[:2])
            or any(bool(ZeroTrustSandboxJudge.extract_python_code(c[1].abstract_response)) for c in cands[:2])
        )

        is_creative_req = self.fluid_composer.is_explicit_creative_request(orig_p)
        oov_entities = EpistemicConfidenceCalibrator.extract_oov_proper_entities(orig_p, self.bit_table)
        if is_coding_req:
            from .phase2_cognitive_core import _WRAPPER_PROPER_ALLOWLIST
            oov_entities = {e for e in oov_entities if e.lower() not in _WRAPPER_PROPER_ALLOWLIST}

        if not is_creative_req and not oov_entities:
            curiosity_hit = self.curiosity_planner.check_diagnostic_ambiguity(
                session_id=session_id,
                query=q_core,
                candidates=cands,
                query_salient_tokens=set(raw_toks),
                already_clarified_this_turn=clarified_this_turn,
            )
            if curiosity_hit is not None:
                q_msg, cur_trace = curiosity_hit
                ms = (time.perf_counter_ns() - t0) / 1_000_000.0
                return CrystalResponse(
                    answer=q_msg,
                    confidence=round(best_score, 4),
                    mode="clarification_needed",
                    matched_rule_id=None,
                    latency_ms=round(ms, 3),
                    sandbox_verified=False,
                    reasoning_trace=trace + cur_trace,
                    epistemic_tier="diagnostic_clarification",
                )

        different_domains = len(cands) >= 2 and (cands[0][1].domain or "") != (cands[1][1].domain or "")
        if (different_domains or not is_coding_req) and len(cands) >= 2 and (cands[0][0] - cands[1][0] <= 0.035) and cands[0][0] >= 0.60:
            c1, c2 = cands[0][1], cands[1][1]
            if c1.domain != c2.domain or (not is_coding_req and c1.abstract_response != c2.abstract_response):
                ms = (time.perf_counter_ns() - t0) / 1_000_000.0
                p1_desc = c1.abstract_prompt.split('\n')[0][:80]
                p2_desc = c2.abstract_prompt.split('\n')[0][:80]
                return CrystalResponse(
                    answer=(
                        f"Did you mean:\n"
                        f"1. [{c1.domain}] {p1_desc}\n"
                        f"2. [{c2.domain}] {p2_desc}\n"
                        f"Please specify which one you'd like to explore."
                    ),
                    confidence=round((cands[0][0] + cands[1][0]) / 2.0, 4),
                    mode="clarification_needed",
                    matched_rule_id=None,
                    latency_ms=round(ms, 3),
                    sandbox_verified=False,
                    reasoning_trace=trace + [
                        f"Ambiguity detected: Rule #{c1.rule_id} ({c1.domain}) vs "
                        f"Rule #{c2.rule_id} ({c2.domain}) delta <= 0.035"
                    ],
                    epistemic_tier="diagnostic_clarification",
                )

        q_expanded = self.bit_table.expand_content_tokens(raw_toks)
        min_lock_score = 0.85 if is_creative_req else 0.22
        for score, rule in cands:
            if oov_entities and rule.num_slots == 0 and not oov_entities.issubset(rule.token_set):
                trace.append(f"Rejected partial match #{rule.rule_id}: missing OOV proper entity {sorted(oov_entities)}")
                continue
            ok_match, cal_conf, q_cov, match_reason = EpistemicConfidenceCalibrator.verify_candidate_match(
                query_text=q_core,
                original_prompt=orig_p,
                raw_toks=raw_toks,
                q_expanded=q_expanded,
                rule=rule,
                score=score,
                bit_table=self.bit_table,
                detected_lang=detected_lang,
                min_lock_score=min_lock_score,
                is_coding_intent=is_coding_req,
            )
            if not ok_match:
                trace.append(f"Rejected partial/false match on rule #{rule.rule_id} ({match_reason})")
                continue
            instantiated = SlotAbstractor.instantiate(
                query=q_core,
                num_slots=rule.num_slots,
                abstract_response=rule.abstract_response,
                derived_expr=rule.derived_expr,
            )
            if instantiated is None:
                continue
            ok, judge_msg = ZeroTrustSandboxJudge.verify_code_and_test(instantiated)
            if not ok:
                if not any(judge_msg.startswith(p) for p in ("Blocked import", "Blocked from-import")):
                    with self.conn:
                        self.conn.execute("UPDATE rules SET fail_count = fail_count + 1 WHERE rule_id = ?", (rule.rule_id,))
                trace.append(f"Quarantined rule #{rule.rule_id} ({judge_msg})")
                continue

            mode_str = ("slot_formula_lock" if rule.num_slots > 0 and rule.derived_expr else ("slot_template_lock" if rule.num_slots > 0 else "direct_crystal_lock"))
            conf_val = cal_conf
            sb_ver = judge_msg != "non_code"
            if is_coding_req and sb_ver:
                trace.append(
                    f"Code Fast-Path: verified by ZeroTrustSandboxJudge for Rule #{rule.rule_id} (domain={rule.domain}, bit_score={score:.3f}, cov={q_cov:.2f})"
                )
            else:
                trace.append(
                    f"Matched DiskVault Rule #{rule.rule_id} (domain={rule.domain}, bit_score={score:.3f}, cov={q_cov:.2f}, slots={rule.num_slots})"
                )
            if not sb_ver:
                styled_ans, was_styled = self.fluid_composer.realize_grounded_style(orig_p, instantiated, sig2048=raw_sig)
                if was_styled:
                    instantiated = styled_ans
                    mode_str = "bicameral_grounded_compose"
                    trace.append("TernaryFluidComposer (1.58-bit) styled verified fact under FactFidelityGuard")
            ms = (time.perf_counter_ns() - t0) / 1_000_000.0
            return CrystalResponse(
                answer=instantiated,
                confidence=conf_val,
                mode=mode_str,
                matched_rule_id=rule.rule_id,
                latency_ms=round(ms, 3),
                sandbox_verified=sb_ver,
                reasoning_trace=trace,
                epistemic_tier=EpistemicConfidenceCalibrator.classify_tier(
                    conf_val, mode_str, sb_ver, query_coverage=q_cov
                ),
            )

        creative_out = self.fluid_composer.synthesize_creative(orig_p, sig2048=raw_sig)
        if creative_out is not None:
            ms = (time.perf_counter_ns() - t0) / 1_000_000.0
            return CrystalResponse(
                answer=creative_out,
                confidence=0.75,
                mode="bicameral_creative_synthesis",
                matched_rule_id=None,
                latency_ms=round(ms, 3),
                sandbox_verified=False,
                reasoning_trace=trace
                + ["Synthesized via 1.58-bit TernaryFluidComposer (CreativeFactualBleedGuard: vault untouched)"],
                epistemic_tier="creative_synthesis",
            )

        ms = (time.perf_counter_ns() - t0) / 1_000_000.0
        try:
            inferred_dom = self._infer_topic(q_core)
            self.conn.execute(
                "INSERT INTO unanswered_queries(query_text, inferred_domain, created_at) VALUES (?, ?, ?)",
                (q_core[:256], inferred_dom, time.time()),
            )
            self.conn.commit()
        except Exception:
            pass
        return CrystalResponse(
            answer="No verified Crystal Rule matched above threshold yet — feed one example to crystallize instantly.",
            confidence=0.0,
            mode="unmatched_honest_abstain",
            matched_rule_id=None,
            latency_ms=round(ms, 3),
            sandbox_verified=False,
            reasoning_trace=trace + ["Abstained honestly instead of hallucinating."],
            epistemic_tier="honest_abstain",
        )

    @staticmethod
    def _infer_topic(query: str) -> str:
        q_low = query.lower()
        if any(w in q_low for w in ("integral", "derivative", "equation", "algebra", "matrix", "prime", "divisors", "polynomial", "calculus", "geometry", "triangle")):
            return "math"
        if any(w in q_low for w in ("python", "code", "function", "class", "algorithm", "async", "thread", "bug", "sql", "git", "docker", "rust", "c++")):
            return "programming"
        if any(w in q_low for w in ("quantum", "physics", "velocity", "energy", "electron", "gravity", "relativity", "force", "photon", "particle", "newton")):
            return "physics"
        if any(w in q_low for w in ("cell", "dna", "gene", "protein", "organism", "mitochondria", "biology", "bacteria", "enzyme")):
            return "biology"
        if any(w in q_low for w in ("history", "century", "war", "king", "empire", "president", "treaty", "revolution", "dynasty")):
            return "history"
        if any(w in q_low for w in ("capital", "country", "river", "mountain", "ocean", "geography", "continent", "border")):
            return "geography"
        if any(w in q_low for w in ("money", "stock", "inflation", "gdp", "market", "economy", "interest", "bank")):
            return "finance"
        words = [w for w in re.findall(r"[a-z]{4,}", q_low) if w not in ("what", "when", "where", "which", "could", "would", "should", "about", "there", "their")]
        return words[0] if words else "general"
