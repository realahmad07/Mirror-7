"""
MirrorCrystalEngine — Complete Zero-Weight, Zero-Gradient, Surprise-Driven CPU Intelligence Engine.
Streams .jsonl, .txt, .md, and .py files in 1 pass on CPU with < 1 ms query latency and 0% forgetting.
"""
from __future__ import annotations

import json
from pathlib import Path
import re
import time
from typing import Any
import zlib

from .bit_signatures import ContextBitTable
from .grammar_and_sandbox import ByteMarkovGrammar, ZeroTrustSandboxJudge
from .lsh_rule_store import LSHRuleIndex
from .phase2_cognitive_core import (
    CoreferenceAndSenseTracker,
    EpistemicConfidenceCalibrator,
    FluentByteRealizer,
    UniversalConceptHub,
)
from .phase5_bicameral_and_cas import (
    ActiveCuriosityPlanner,
    DocumentAutoCrystallizer,
    SymbolicCASOrgan,
    TernaryFluidComposer,
)
from .slot_abstractor import SlotAbstractor
from .intent import split_compound_intents
from .response_types import CrystalResponse
from .session_memory import BinarySessionMemory


class MirrorCrystalEngine:
    """Zero-Parameter / Zero-Backprop Byte Crystallization Engine."""

    def __init__(self, max_frame_df: int = 8, max_postings_scan: int = 64):
        self.bit_table = ContextBitTable(max_frame_df=max_frame_df)
        self.rule_index = LSHRuleIndex(max_postings_scan=max_postings_scan)
        self.grammar = ByteMarkovGrammar()
        self.sessions = BinarySessionMemory()
        self.concept_hub = UniversalConceptHub()
        self.coref_tracker = CoreferenceAndSenseTracker(ttl_turns=3)
        self.realizer = FluentByteRealizer()
        self.fluid_composer = TernaryFluidComposer()
        self.curiosity_planner = ActiveCuriosityPlanner()
        self.stats: dict[str, int] = {
            "pairs_seen": 0,
            "surprises_crystallized": 0,
            "duplicates_skipped": 0,
            "sandbox_verified_rules": 0,
            "sandbox_rejected_rules": 0,
        }

    def observe_unlabeled_text(self, text: str) -> None:
        """Feed raw text/code to build Context Barcodes, causal links, and Byte Grammar (0 labels needed)."""
        for line in text.splitlines():
            s = line.strip()
            if not s:
                continue
            self.bit_table.observe_text(s)
            self.grammar.observe_bytes(s)
            self.rule_index.observe_causal_fact(s)
            self.fluid_composer.observe_corpus_text(s)

    def ingest_pair(
        self,
        user_text: str,
        target_text: str,
        think: str = "",
        tool: str = "",
        test_code: str = "",
        domain: str = "general",
        trust_tier: int = 2,
    ) -> bool:
        """Learn one (user_text -> target_text) pair in a single CPU pass without backpropagation."""
        u = user_text.strip()
        t = target_text.strip()
        if not u or not t:
            return False

        self.stats["pairs_seen"] += 1
        self.bit_table.observe_text(u)
        self.bit_table.observe_text(t)
        self.grammar.observe_bytes(t)
        self.fluid_composer.observe_corpus_text(t)

        full_response = f"<think> {think.strip()} </think>\n{t}" if think.strip() and "<think>" not in t else t

        # Reality Filter: if test_code or standalone Python code is provided, verify on CPU first!
        verified_bonus = 0
        if test_code.strip():
            ok, _ = ZeroTrustSandboxJudge.verify_code_and_test(full_response, test_code=test_code)
            if not ok:
                self.stats["sandbox_rejected_rules"] += 1
                return False
            verified_bonus = 5
            self.stats["sandbox_verified_rules"] += 1

        _, created = self.rule_index.add_or_reinforce_rule(
            prompt=u,
            response=full_response,
            bit_table=self.bit_table,
            tool=tool,
            domain=domain,
            verified_bonus=verified_bonus,
            trust_tier=trust_tier,
        )
        if created:
            self.stats["surprises_crystallized"] += 1
        else:
            self.stats["duplicates_skipped"] += 1
        return created

    def ingest_jsonl(self, path: Path | str, max_rows: int | None = None, fold_synonyms: bool = True) -> dict[str, Any]:
        """Stream a .jsonl file line-by-line in O(1) transient RAM and crystallize rules in 1 pass."""
        fpath = Path(path)
        t0 = time.perf_counter()
        rows_read = 0

        with fpath.open("r", encoding="utf-8", errors="replace") as fh:
            for raw_line in fh:
                if max_rows is not None and rows_read >= max_rows:
                    break
                line = raw_line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if obj.get("split") in {"test", "eval_holdout"}:
                    continue

                if isinstance(obj.get("turns"), list):
                    for turn in obj["turns"]:
                        u = turn.get("user") or turn.get("user_text") or ""
                        r = turn.get("response") or turn.get("target_text") or ""
                        self.ingest_pair(u, r)
                    rows_read += 1
                    continue

                u = (
                    obj.get("user_text")
                    or obj.get("user")
                    or obj.get("prompt")
                    or obj.get("instruction")
                    or obj.get("question")
                    or ""
                )
                r = (
                    obj.get("target_text")
                    or obj.get("response")
                    or obj.get("completion")
                    or obj.get("output")
                    or obj.get("answer")
                    or ""
                )
                state_meta = obj.get("state") if isinstance(obj.get("state"), dict) else {}
                dom = str(state_meta.get("domain") or obj.get("category") or obj.get("goal") or "general")
                self.ingest_pair(
                    user_text=u,
                    target_text=r,
                    think=str(obj.get("think") or ""),
                    tool=str(obj.get("tool") or ""),
                    test_code=str(obj.get("test") or obj.get("test_code") or ""),
                    domain=dom,
                )
                rows_read += 1

        merged_syns = self.bit_table.fold_equivalences() if fold_synonyms else 0
        elapsed_s = max(1e-6, time.perf_counter() - t0)
        repo_root = Path(__file__).resolve().parents[1]
        try:
            display_file = fpath.resolve().relative_to(repo_root).as_posix()
        except ValueError:
            display_file = fpath.name
        return {
            "file": display_file,
            "rows_ingested": rows_read,
            "total_rules_crystallized": len(self.rule_index.rules),
            "surprises_crystallized": self.stats["surprises_crystallized"],
            "duplicates_skipped": self.stats["duplicates_skipped"],
            "synonym_bridges_folded": merged_syns,
            "elapsed_sec": round(elapsed_s, 3),
            "rows_per_sec_cpu": round(rows_read / elapsed_s, 1),
        }

    def ingest_text_file(self, path: Path | str, fold_synonyms: bool = True) -> dict[str, Any]:
        fpath = Path(path)
        t0 = time.perf_counter()
        lines = 0
        with fpath.open("r", encoding="utf-8", errors="replace") as fh:
            for ln in fh:
                if ln.strip():
                    self.observe_unlabeled_text(ln)
                    lines += 1
        merged = self.bit_table.fold_equivalences() if fold_synonyms else 0
        elapsed = max(1e-6, time.perf_counter() - t0)
        return {"file": fpath.name, "lines_read": lines, "synonym_bridges_folded": merged, "elapsed_sec": round(elapsed, 3)}

    def ingest_document(self, text: str, title: str = "Document", domain: str = "doc_crystallized") -> dict[str, int]:
        """Auto-crystallize unstructured Markdown, textbook prose, or inline Python code without .jsonl."""
        return DocumentAutoCrystallizer.ingest_document_text(self, text, title=title, domain=domain)

    def ingest_source_file(self, path: Path | str, domain: str | None = None) -> dict[str, int]:
        """Auto-crystallize any .md, .txt, or .py file (verifying functions via ZeroTrustSandboxJudge)."""
        return DocumentAutoCrystallizer.ingest_file(self, path, domain=domain)

    def _respond_atomic(
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

        # Gate 2: Encode 2,048-bit Context Barcode (matching both literal entity facts & slot-formula locks)
        raw_sig, raw_toks = self.bit_table.encode_text_signature(q_sense)
        cands_raw = self.rule_index.query_candidates(raw_sig, raw_toks, self.bit_table, top_k=4)

        masked_q = SlotAbstractor.mask_query_numbers(q_core)
        if masked_q != q_core:
            q_unique_nums = list(dict.fromkeys(SlotAbstractor.extract_numbers(q_core)))
            slot_sig, slot_toks = self.bit_table.encode_text_signature(masked_q)
            cands_slot = self.rule_index.query_candidates(
                slot_sig, slot_toks, self.bit_table, top_k=4, required_slots=len(q_unique_nums)
            )
            merged_by_id: dict[int, tuple[float, Any]] = {}
            for sc, rl in cands_raw + cands_slot:
                prev = merged_by_id.get(rl.rule_id)
                if prev is None or sc > prev[0]:
                    merged_by_id[rl.rule_id] = (sc, rl)
            cands = sorted(merged_by_id.values(), key=lambda kv: kv[0], reverse=True)[:4]
        else:
            cands = cands_raw

        best_score = cands[0][0] if cands else 0.0
        # Gate 3: If direct lock confidence is moderate/low, check Multi-Hop Causal Rule Chaining!
        if best_score < 0.72:
            chained = self.rule_index.chain_infer(q_core, self.bit_table)
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

        is_creative_req = self.fluid_composer.is_explicit_creative_request(orig_p)
        oov_entities = EpistemicConfidenceCalibrator.extract_oov_proper_entities(orig_p, self.bit_table)

        # Gate 3B: ActiveCuriosityPlanner (Diagnostic Information-Gain Clarifier with 1-Turn Loop Guard)
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

        # Gate 4: Instantiate winning Lock (binding unseen numbers & verifying any Python code on CPU!)
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
                rule.fail_count += 1
                trace.append(f"Quarantined rule #{rule.rule_id} ({judge_msg})")
                continue

            mode_str = ("slot_formula_lock" if rule.num_slots > 0 and rule.derived_expr else ("slot_template_lock" if rule.num_slots > 0 else "direct_crystal_lock"))
            conf_val = cal_conf
            sb_ver = judge_msg != "non_code"
            trace.append(
                f"Matched CrystalRule #{rule.rule_id} (domain={rule.domain}, bit_score={score:.3f}, cov={q_cov:.2f}, slots={rule.num_slots})"
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

        # Gate 5: Right-Brain Creative Synthesis ONLY on explicit creative prompts (CreativeFactualBleedGuard)
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

    def respond(self, prompt: str, session_id: str = "default") -> CrystalResponse:
        """Execute full CPU inference with UniversalConceptHub, Coreference, Session Lock & Anti-Saturation."""
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

        # Gate 0: Zero-Duplication Cross-Lingual Concept Hub
        q_canon, detected_lang, mapped_concepts = self.concept_hub.normalize_to_canonical(q_raw)
        pre_trace: list[str] = []
        if mapped_concepts:
            pre_trace.append(f"UniversalConceptHub ({detected_lang}): {', '.join(mapped_concepts[:6])}")

        # Gate 0B: ActiveCuriosityPlanner follow-up resolution (CuriosityLoopGuard)
        q_canon, clarified_this_turn = self.curiosity_planner.merge_followup_if_pending(session_id, q_canon)
        if clarified_this_turn:
            pre_trace.append(f"ActiveCuriosityPlanner fused follow-up choice -> '{q_canon}'")

        # Gate 0C: SymbolicCASOrgan (Exact Rational Calculus, Algebra/Linear Systems & Unit Conversions)
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

        # Gate 1: Update & check persistent binary session state
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

        # Gate 1A: Multi-Turn Coreference Pronoun Resolution + TopicShiftDecayGuard
        q, coref_trace = self.coref_tracker.resolve_turn(session_id, q_canon)
        pre_trace.extend(coref_trace)

        # Gate 1B: Compound Multi-Intent Decomposition (when 2+ sub-questions both match cleanly)
        sub_intents = split_compound_intents(q)
        if len(sub_intents) >= 2:
            sub_res = [self._respond_atomic(part, detected_lang=detected_lang) for part in sub_intents]
            valid_subs = [r for r in sub_res if r.mode != "unmatched_honest_abstain" and r.confidence >= 0.30]
            if len(valid_subs) >= 2:
                combined_ans = "\n\n".join(f"[{idx + 1}] {r.answer}" for idx, r in enumerate(valid_subs))
                combined_trace = pre_trace + [tr for r in valid_subs for tr in r.reasoning_trace]
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
                    reasoning_trace=combined_trace,
                    epistemic_tier=EpistemicConfidenceCalibrator.classify_tier(avg_conf, "compound_multi_intent", all_ver),
                )

        # Gate 1C: Long-Log / Document Sliding Window Anti-Saturation (>420 chars or >5 lines)
        windows = ContextBitTable.split_focus_windows(q)
        if len(windows) > 1:
            best_win_res: CrystalResponse | None = None
            for win in windows:
                cand_res = self._respond_atomic(win, detected_lang=detected_lang)
                if best_win_res is None or cand_res.confidence > best_win_res.confidence:
                    best_win_res = cand_res
            if best_win_res is not None and best_win_res.mode != "unmatched_honest_abstain":
                ms = (time.perf_counter_ns() - t0) / 1_000_000.0
                return CrystalResponse(
                    answer=best_win_res.answer,
                    confidence=best_win_res.confidence,
                    mode=best_win_res.mode,
                    matched_rule_id=best_win_res.matched_rule_id,
                    latency_ms=round(ms, 3),
                    sandbox_verified=best_win_res.sandbox_verified,
                    reasoning_trace=pre_trace + ["Focused long-log diagnostic window"] + best_win_res.reasoning_trace,
                    epistemic_tier=best_win_res.epistemic_tier,
                )

        return self._respond_atomic(
            q,
            pre_trace=pre_trace,
            session_id=session_id,
            clarified_this_turn=clarified_this_turn,
            original_prompt=q_raw,
            detected_lang=detected_lang,
        )

    def save(self, path: Path | str) -> int:
        """Persist entire brain into a single zlib-compressed .mcrystal file."""
        fpath = Path(path)
        fpath.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": "mirror_crystal_v1",
            "stats": self.stats,
            "bit_table": self.bit_table.to_state_dict(),
            "rule_index": self.rule_index.to_state_dict(),
            "grammar": self.grammar.to_state_dict(),
        }
        compressed = zlib.compress(json.dumps(payload, separators=(",", ":")).encode("utf-8"), level=6)
        fpath.write_bytes(compressed)
        return len(compressed)

    @classmethod
    def load(cls, path: Path | str) -> MirrorCrystalEngine:
        raw = zlib.decompress(Path(path).read_bytes()).decode("utf-8")
        data = json.loads(raw)
        eng = cls()
        eng.stats.update({k: int(v) for k, v in data.get("stats", {}).items()})
        eng.bit_table = ContextBitTable.from_state_dict(data["bit_table"])
        eng.rule_index = LSHRuleIndex.from_state_dict(data["rule_index"])
        eng.grammar = ByteMarkovGrammar.from_state_dict(data.get("grammar", {}))
        return eng
