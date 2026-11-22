#!/usr/bin/env python3
"""Held-out evaluation for Mirror-7's five CPU-native organs.

No task prompts, answer keys, or corpus examples from Mirror-7's internal test fixtures
are used as the external tasks. Public benchmark data is downloaded at runtime and is
not copied into the repository.

Sources:
- LongMemEval oracle: MIT, pinned Git revision + SHA-256.
- MBPP sanitized: Google Research source file pinned to a Git commit; IDs 11..510
  are the benchmark's published test range.
- ROCStories Cloze: public Hugging Face dataset-server rows; revision reported by API.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SEED = 20261122
LONGMEMEVAL_URL = (
    "https://huggingface.co/datasets/xiaowu0162/longmemeval/resolve/"
    "b2009b6cfb6bd22616b4732253eb65ebc998e1b7/longmemeval_oracle.json"
)
LONGMEMEVAL_SHA256 = "821a2034d219ab45846873dd14c14f12cfe7776e73527a483f9dac095d38620c"
MBPP_URL = (
    "https://raw.githubusercontent.com/google-research/google-research/"
    "f82046ba5aabbbb427dbfd38a254d26bff08b533/mbpp/sanitized-mbpp.json"
)
MBPP_REVISION = "f82046ba5aabbbb427dbfd38a254d26bff08b533"
ROCSTORIES_DATASET = "qikp/rocstories-cloze"


def http_get(url: str, timeout: int = 90) -> bytes:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "Mirror7-External-Eval/1.0", "Accept": "*/*"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def download_verified(url: str, destination: Path, sha256: str | None = None) -> tuple[bytes, str]:
    data = http_get(url)
    digest = hashlib.sha256(data).hexdigest()
    if sha256 and digest.lower() != sha256.lower():
        raise RuntimeError(f"SHA-256 mismatch for {url}: got {digest}, expected {sha256}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(data)
    return data, digest


def hf_rows(
    dataset: str,
    max_rows: int = 200,
    *,
    config_hint: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    base = "https://datasets-server.huggingface.co"
    q = urllib.parse.urlencode({"dataset": dataset})
    split_payload = json.loads(http_get(f"{base}/splits?{q}").decode("utf-8"))
    splits = split_payload.get("splits", [])
    candidates = [
        s for s in splits
        if ("test" in str(s.get("split", "")).casefold()
            or "cloze" in str(s.get("split", "")).casefold())
        and (config_hint is None or str(s.get("config", "")) == config_hint)
    ]
    if not candidates:
        raise RuntimeError(f"No test/cloze split exposed for {dataset}")
    candidates.sort(
        key=lambda s: (
            0 if "cloze_test" in str(s.get("split", "")).casefold() else 1,
            str(s.get("config", "")),
            str(s.get("split", "")),
        )
    )
    chosen = candidates[0]
    params = urllib.parse.urlencode(
        {
            "dataset": dataset,
            "config": chosen["config"],
            "split": chosen["split"],
            "offset": 0,
            "length": min(500, max_rows),
        }
    )
    payload = json.loads(http_get(f"{base}/rows?{params}").decode("utf-8"))
    rows = [entry["row"] for entry in payload.get("rows", []) if isinstance(entry, dict) and isinstance(entry.get("row"), dict)]
    if not rows:
        raise RuntimeError(f"No rows returned for {dataset}/{chosen['config']}/{chosen['split']}")
    meta = {
        "dataset": dataset,
        "config": chosen["config"],
        "split": chosen["split"],
        "num_rows_returned": len(rows),
    }
    return rows, meta


def deterministic_sample(rows: list[dict[str, Any]], n: int, key: str) -> list[dict[str, Any]]:
    indexed = list(rows)
    random.Random(f"{SEED}:{key}").shuffle(indexed)
    return indexed[: min(n, len(indexed))]


def flatten_session(session: Any) -> str:
    if isinstance(session, str):
        return session.strip()
    if not isinstance(session, list):
        return str(session)
    parts: list[str] = []
    for turn in session:
        if isinstance(turn, dict):
            text = turn.get("content", turn.get("text", ""))
            if text:
                parts.append(str(text))
        elif turn:
            parts.append(str(turn))
    return " ".join(parts).strip()


def evaluate_organ1_babi_parser(babi_rows: list[dict[str, Any]]) -> dict[str, Any]:
    from mirror_crystal.cortical_active_inference import BilingualCCGHDCParser

    parser = BilingualCCGHDCParser()
    context_facts = actor_correct = location_cases = location_correct = 0
    sampled_fact_examples: list[dict[str, Any]] = []
    for row in babi_rows:
        story = row.get("story", {})
        if not isinstance(story, dict):
            continue
        texts = story.get("text") or []
        types = story.get("type") or []
        for text, kind in zip(texts, types):
            if int(kind) != 0:
                continue
            parsed = parser.parse_clause(str(text))
            tokens = re.findall(r"[A-Za-z]+", str(text).casefold())
            if not tokens:
                continue
            context_facts += 1
            expected_actor = tokens[0]
            actor_ok = parsed.who == expected_actor
            actor_correct += int(actor_ok)
            match = re.match(
                r"^([a-z]+)\\s+(?:moved|went|journeyed|travelled|travelled back)\\s+(?:to|into)\\s+the?\\s*([a-z]+)[.!]?$",
                str(text).strip().casefold(),
            )
            if match:
                location_cases += 1
                destination = match.group(2)
                target_ok = destination in parsed.targ
                location_correct += int(target_ok)
                if len(sampled_fact_examples) < 8:
                    sampled_fact_examples.append({
                        "text": str(text),
                        "parsed_actor": parsed.who,
                        "parsed_action": parsed.act,
                        "parsed_target": parsed.targ,
                        "expected_actor": expected_actor,
                        "expected_destination": destination,
                        "passed": bool(actor_ok and target_ok),
                    })
    return {
        "babi_test_records": len(babi_rows),
        "context_facts_parsed": context_facts,
        "actor_exact_matches": actor_correct,
        "actor_accuracy": round(actor_correct / max(1, context_facts), 4),
        "location_facts": location_cases,
        "location_target_matches": location_correct,
        "location_target_accuracy": round(location_correct / max(1, location_cases), 4),
        "sampled_examples": sampled_fact_examples,
        "metric_note": "Gold actor and destination labels are checked on bAbI test facts; this is a narrow parser probe, not full QA accuracy.",
    }


def evaluate_organ1_story_parser(story_rows: list[dict[str, Any]]) -> dict[str, Any]:
    from mirror_crystal.cortical_active_inference import BilingualCCGHDCParser

    parser = BilingualCCGHDCParser()
    sentence_count = 0
    nonempty_structure = 0
    causal_edges: list[tuple[str, str, str]] = []
    detected_languages: dict[str, int] = {}
    for row in story_rows:
        sentences = []
        for i in range(1, 5):
            value = row.get(f"InputSentence{i}") or row.get(f"input_sentence_{i}") or row.get(f"sentence{i}")
            if value:
                sentences.append(str(value))
        for sentence in sentences:
            parsed = parser.parse_clause(sentence)
            sentence_count += 1
            detected_languages[parsed.language] = detected_languages.get(parsed.language, 0) + 1
            if any((parsed.who, parsed.act, parsed.targ, parsed.cond, parsed.eff)):
                nonempty_structure += 1
            if parsed.cond and parsed.eff and parsed.act in {"causes", "implies", "caused_by"}:
                causal_edges.append((parsed.cond, parsed.act, parsed.eff))
    return {
        "external_sentences": sentence_count,
        "nonempty_structures": nonempty_structure,
        "structure_coverage": round(nonempty_structure / max(1, sentence_count), 4),
        "language_counts": detected_languages,
        "explicit_causal_edges_extracted": len(causal_edges),
        "_causal_edges": causal_edges,
        "metric_note": "Coverage only; no gold semantic-role annotations are provided by ROCStories.",
    }


def evaluate_organ2_symbolic_graph(causal_edges: list[tuple[str, str, str]]) -> dict[str, Any]:
    from mirror_crystal.quotient_egraph_resonator import (
        DEFAULT_ALGEBRA_REWRITE_RULES,
        QuotientEGraph,
        TwinEGraphDoCalculus,
    )

    # Fresh nested combinations deliberately absent from the repository's unit-test fixtures.
    expression_pairs = [
        (("mul", ("add", "x", "0"), "2"), ("add", "x", "x")),
        (("add", ("mul", "y", "0"), "z"), "z"),
        (("add", ("add", "a", "0"), "b"), ("add", "b", "a")),
        (("mul", ("mul", "p", "0"), "2"), "0"),
        (("add", ("not", ("not", "q")), "0"), "q"),
        (("mul", "2", ("mul", "r", "0")), "0"),
        (("add", ("mul", "m", "2"), ("mul", "n", "0")), ("add", "m", "m")),
    ]
    pair_results = []
    for left, right in expression_pairs:
        graph = QuotientEGraph()
        left_id = graph.add_expr(left)
        right_id = graph.add_expr(right)
        graph.saturate(DEFAULT_ALGEBRA_REWRITE_RULES, max_iters=10, max_nodes=2000)
        pair_results.append(bool(graph.equivalent(left_id, right_id)))

    # Apply the intervention organ to up to 20 causal relations extracted from unseen stories.
    intervention_checks = 0
    intervention_failures = 0
    for idx, (cause, _relation, effect) in enumerate(causal_edges[:20]):
        if not cause.strip() or not effect.strip() or cause.casefold() == effect.casefold():
            continue
        twin = TwinEGraphDoCalculus()
        twin.add_causal_rule(f"external_story_{idx}", (cause,), effect)
        twin.assert_world_fact(cause, True)
        baseline = twin.truth_jtms.query(effect)
        if baseline is None or baseline.truth_value is not True:
            intervention_failures += 1
            continue
        report = twin.do_intervene({effect: False})
        intervention_checks += 1
        if not report.severed_rules:
            intervention_failures += 1
    return {
        "compositional_equivalence_probes": len(pair_results),
        "compositional_equivalence_passed": sum(pair_results),
        "story_derived_interventions": intervention_checks,
        "story_derived_intervention_failures": intervention_failures,
        "metric_note": (
            "Equivalence probes are held-out operator compositions, not a published algebra benchmark; "
            "interventions use causal clauses extracted from external ROCStories."
        ),
    }


def evaluate_organ3_longmemeval(records: list[dict[str, Any]], sample_size: int) -> dict[str, Any]:
    from mirror_crystal.hippocampal_sdm_memory import KanervaSDMInt8

    by_type: dict[str, list[dict[str, Any]]] = {}
    for item in records:
        by_type.setdefault(str(item.get("question_type", "unknown")), []).append(item)
    sample: list[dict[str, Any]] = []
    for qtype in sorted(by_type):
        bucket = sorted(by_type[qtype], key=lambda r: str(r.get("question_id", "")))
        if bucket:
            sample.append(bucket[0])
    remainder = [r for qtype in sorted(by_type) for r in by_type[qtype][1:]]
    sample.extend(deterministic_sample(remainder, max(0, sample_size - len(sample)), "longmemeval"))
    sample = sample[:sample_size]

    found = 0
    answer_session_hit = 0
    no_evidence = 0
    per_type: dict[str, dict[str, int]] = {}
    for index, item in enumerate(sample):
        sessions = item.get("haystack_sessions") or []
        session_ids = item.get("haystack_session_ids") or []
        expected_ids = {str(x) for x in (item.get("answer_session_ids") or [])}
        if not expected_ids:
            no_evidence += 1
            continue
        memory = KanervaSDMInt8(
            num_hard_locations=256,
            sdr_dim=256,
            top_k_sdr=16,
            value_dim=64,
            active_locations_m=8,
            seed=f"external-longmemeval-{index}",
        )
        for sid, session in zip(session_ids, sessions):
            text = flatten_session(session)
            if text:
                memory.write(
                    episode_id=str(sid),
                    key_text=text,
                    payload={"session_id": str(sid), "text": text},
                )
        recall = memory.read(str(item.get("question", "")))
        qtype = str(item.get("question_type", "unknown"))
        stats = per_type.setdefault(qtype, {"n": 0, "recall_found": 0, "evidence_session_hit": 0})
        stats["n"] += 1
        if recall.found and recall.episode is not None:
            found += 1
            stats["recall_found"] += 1
            if str(recall.episode.episode_id) in expected_ids:
                answer_session_hit += 1
                stats["evidence_session_hit"] += 1
    return {
        "questions_tested": len(sample) - no_evidence,
        "recall_found": found,
        "evidence_session_hits": answer_session_hit,
        "evidence_session_recall_accuracy": round(answer_session_hit / max(1, len(sample) - no_evidence), 4),
        "no_evidence_records": no_evidence,
        "by_question_type": per_type,
        "metric_note": "Oracle evidence-session retrieval only; this does not score free-form answer generation.",
    }


def evaluate_organ4_mbpp(mbpp_rows: list[dict[str, Any]], sample_size: int) -> dict[str, Any]:
    from mirror7_backend.native_code_generation import NativeCodeGenerationProvider
    from mirror_crystal.grammar_and_sandbox import ZeroTrustSandboxJudge

    # MBPP's published test partition is task IDs 11..510; do not use prompt-pool/train IDs.
    test_rows = [
        row for row in mbpp_rows
        if isinstance(row.get("task_id"), int) and 11 <= int(row["task_id"]) <= 510
        and row.get("prompt") and row.get("test_list")
    ]
    sample = deterministic_sample(test_rows, sample_size, "mbpp-test")
    provider = NativeCodeGenerationProvider()
    generated = passed = execution_failed = 0
    unsupported_ids: list[int] = []
    task_results: list[dict[str, Any]] = []
    for row in sample:
        task_id = int(row["task_id"])
        try:
            candidate = provider(str(row["prompt"]))
        except Exception as exc:
            execution_failed += 1
            task_results.append({"task_id": task_id, "status": "provider_error", "error": type(exc).__name__})
            continue
        if not isinstance(candidate, dict) or not candidate.get("code"):
            unsupported_ids.append(task_id)
            task_results.append({"task_id": task_id, "status": "unsupported"})
            continue
        generated += 1
        imports = row.get("test_imports") or []
        tests = row.get("test_list") or []
        external_tests = "\n".join([str(x) for x in imports + tests])
        ok, output = ZeroTrustSandboxJudge.verify_code_and_test(str(candidate["code"]), external_tests)
        if ok:
            passed += 1
            task_results.append({"task_id": task_id, "status": "passed", "kind": candidate.get("generation_kind")})
        else:
            execution_failed += 1
            task_results.append({"task_id": task_id, "status": "failed", "reason": str(output)[:400]})
    return {
        "published_test_tasks_available": len(test_rows),
        "tasks_sampled": len(sample),
        "candidates_generated": generated,
        "external_test_passed": passed,
        "candidate_execution_failed": execution_failed,
        "unsupported_tasks": len(unsupported_ids),
        "unsupported_task_ids": unsupported_ids,
        "pass_at_1_over_all_sampled": round(passed / max(1, len(sample)), 4),
        "coverage_given_supported": round(passed / max(1, generated), 4) if generated else 0.0,
        "task_results": task_results,
        "metric_note": "No MBPP reference solution is imported or executed; only the task prompt and official test_list are used.",
    }


def get_story_fields(row: dict[str, Any]) -> tuple[list[str], str, str, int] | None:
    context = []
    for i in range(1, 5):
        value = row.get(f"InputSentence{i}") or row.get(f"input_sentence_{i}") or row.get(f"sentence{i}")
        if value:
            context.append(str(value))
    end1 = row.get("RandomFifthSentenceQuiz1") or row.get("sentence_quiz1") or row.get("ending1")
    end2 = row.get("RandomFifthSentenceQuiz2") or row.get("sentence_quiz2") or row.get("ending2")
    label = row.get("AnswerRightEnding", row.get("answer_right_ending", row.get("label")))
    try:
        label_i = int(label)
    except (TypeError, ValueError):
        return None
    if len(context) != 4 or not end1 or not end2 or label_i not in (1, 2):
        return None
    return context, str(end1), str(end2), label_i


def evaluate_organ5_storycloze(story_rows: list[dict[str, Any]]) -> dict[str, Any]:
    from mirror_crystal.ternary_lut_realizer import LTAGKneserNeyRealizer

    realizer = LTAGKneserNeyRealizer()
    # Keep the organ's shipped eight-sentence prior; do not train on external evaluation stories.
    scorer = realizer.pdcst
    sample = deterministic_sample(story_rows, 80, "storycloze")
    total = correct = ties = 0
    generated = 0
    for row in sample:
        parsed = get_story_fields(row)
        if parsed is None:
            continue
        context, end1, end2, label = parsed
        s1 = scorer.score_sentence_q16(" ".join(context + [end1]))
        s2 = scorer.score_sentence_q16(" ".join(context + [end2]))
        predicted = 1 if s1 > s2 else (2 if s2 > s1 else 0)
        total += 1
        if predicted == 0:
            ties += 1
        elif predicted == label:
            correct += 1
        generated += 1
    return {
        "story_cloze_pairs_scored": total,
        "correct_right_ending_ranked": correct,
        "ties": ties,
        "strict_accuracy_excluding_ties": round(correct / max(1, total - ties), 4),
        "accuracy_with_ties_as_half": round((correct + 0.5 * ties) / max(1, total), 4),
        "metric_note": (
            "Zero-shot PDCST ending-ranking probe; the scorer is not updated on test stories. "
            "This is not a human-rated narrative-generation metric."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="artifacts/five_organ_external_eval.json")
    parser.add_argument("--sample-size", type=int, default=20)
    args = parser.parse_args()

    started = time.perf_counter()
    cache = ROOT / ".external_eval_cache"
    report: dict[str, Any] = {
        "benchmark": "Mirror-7 Five-Organ External Holdout",
        "seed": SEED,
        "sample_size_requested": args.sample_size,
        "no_mirror_reference_solutions_used": True,
        "organs": {},
        "datasets": {},
        "warnings": [],
    }

    # LongMemEval is pinned to the original public MIT oracle release and checksum.
    longmem_path = cache / "longmemeval_oracle.json"
    longmem_bytes, longmem_hash = download_verified(
        LONGMEMEVAL_URL, longmem_path, LONGMEMEVAL_SHA256
    )
    longmem = json.loads(longmem_bytes.decode("utf-8"))
    report["datasets"]["longmemeval"] = {
        "url": LONGMEMEVAL_URL,
        "sha256": longmem_hash,
        "license": "MIT",
        "records": len(longmem),
        "revision": "b2009b6cfb6bd22616b4732253eb65ebc998e1b7",
    }

    # MBPP sanitized source is pinned by Git commit. Use test IDs and test_list only.
    mbpp_path = cache / "sanitized-mbpp.json"
    mbpp_bytes, mbpp_hash = download_verified(MBPP_URL, mbpp_path)
    mbpp = json.loads(mbpp_bytes.decode("utf-8"))
    report["datasets"]["mbpp"] = {
        "url": MBPP_URL,
        "revision": MBPP_REVISION,
        "sha256_observed": mbpp_hash,
        "license": "CC-BY-4.0 (as stated by the dataset mirror/card; check source terms before redistribution)",
        "records": len(mbpp),
        "test_id_range": [11, 510],
    }

    babi_rows, babi_meta = hf_rows("facebook/babi_qa", max_rows=100, config_hint="en-qa1")
    report["datasets"]["babi_qa"] = {
        **babi_meta,
        "license": "CC-BY-3.0",
        "task": "qa1 single-supporting-fact test split",
    }
    story_rows, story_meta = hf_rows(ROCSTORIES_DATASET, max_rows=300)
    report["datasets"]["rocstories_cloze"] = {
        **story_meta,
        "license": "dataset card reports other/unspecified; used for evaluation only, not redistributed",
    }

    # Organ 1: gold-label parser probe on bAbI context facts plus coverage on StoryCloze.
    story_sample = deterministic_sample(story_rows, 80, "organ1-parser")
    organ1 = evaluate_organ1_story_parser(story_sample)
    causal_edges = organ1.pop("_causal_edges")
    organ1["babi_gold_role_probe"] = evaluate_organ1_babi_parser(deterministic_sample(babi_rows, 50, "babi-qa1"))
    report["organs"]["1_cortical_parser"] = organ1

    # Organ 2: equality-saturation compositions + story-derived causal intervention.
    report["organs"]["2_quotient_egraph_and_twin_do"] = evaluate_organ2_symbolic_graph(causal_edges)

    # Organ 3: actual long-term memory recall on an external 500-question memory benchmark.
    report["organs"]["3_hippocampal_sdm_memory"] = evaluate_organ3_longmemeval(longmem, max(30, args.sample_size))

    # Organ 4: official MBPP test tasks, plus independent multi-file architecture prompts.
    organ4 = evaluate_organ4_mbpp(mbpp, args.sample_size)
    from mirror_crystal.multifile_cegis_architect import MultiFileCEGISArchitect
    architect = MultiFileCEGISArchitect()
    architecture_prompts = [
        "Build an offline-first volunteer shift planner with recurring assignments, waitlists, skill tags, and conflict checks.",
        "Create a local-first household expense ledger with shared bills, reimbursements, budgets, and audit history.",
        "Design an offline research notebook for lab samples, experiment runs, citations, and immutable observations.",
        "Make a campus club task board with role assignments, event checklists, deadlines, and reminders.",
    ]
    architecture_results = []
    for prompt_text in architecture_prompts:
        try:
            build = architect.synthesize_architecture(prompt_text)
            architecture_results.append({
                "sandbox_verified": bool(build.sandbox_verified),
                "modules": int(build.module_count),
                "lines": int(build.total_lines),
                "domain": str(build.spec.domain_family),
                "output": str(build.sandbox_output)[:250],
            })
        except Exception as exc:
            architecture_results.append({"sandbox_verified": False, "error": type(exc).__name__})
    organ4["held_out_multifile_specs"] = architecture_results
    organ4["multifile_sandbox_passed"] = sum(1 for r in architecture_results if r.get("sandbox_verified"))
    report["organs"]["4_multifile_cegis_and_code_synthesis"] = organ4

    # Organ 5: zero-shot StoryCloze right-vs-wrong ending ranking with its frozen prior.
    report["organs"]["5_ternary_lut_and_narrative_realizer"] = evaluate_organ5_storycloze(story_rows)

    report["elapsed_seconds"] = round(time.perf_counter() - started, 3)
    report["overall_note"] = (
        "This is a bounded, reproducible holdout, not a complete AGI evaluation. "
        "Organ 1 coverage is not semantic-role accuracy; Organ 2 includes fresh symbolic "
        "composition probes; Organ 3 measures evidence-session retrieval, not answer writing; "
        "Organ 4 scores only generated candidates on official MBPP tests; Organ 5 reports "
        "zero-shot ending ranking, not human-rated prose quality."
    )

    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "benchmark": report["benchmark"],
        "elapsed_seconds": report["elapsed_seconds"],
        "organs": report["organs"],
        "output": str(output.relative_to(ROOT)),
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
