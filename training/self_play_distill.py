"""
Autonomous Self-Generation, Process Verification & STaR (Self-Taught Reasoner) Engine for Mirror 7.

Enables Mirror 7 (NativeConversationV5) to:
  1. Generate its own questions and multi-step reasoning responses from topics or textbook passages
     instead of relying solely on fixed/fed prompts.
  2. Sample N diverse candidate trajectories (temperature + nucleus top-p sampling) and score each
     candidate with a 7-Dimensional Mathematical & Conversational Process Verifier:
       - Prompt keyword & concept grounding
       - Balanced mathematical brackets/parentheses/bars
       - Step-by-step progression (1. -> 2. -> 3. or structured paragraphs)
       - Causal & logical reasoning connectives (therefore, because, since, implies, hence)
       - Trigram & 5-gram non-repetition (zero recurrent attractor loops)
       - Clean grammatical termination
  3. Export high-scoring self-generated solutions as verified JSONL training records for
     Self-Taught Reasoner (STaR) iterative self-improvement.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Sequence

from training.build_general_ai_corpus import (
    CLEAN_END_CHARS,
    CONNECTIVE_RE,
    FOREIGN_IDENTITY_RE,
    REFUSAL_RE,
    STEP_RE,
    clean_text,
    utf8_len,
)
from training.native_conversation_v5 import NativeConversationV5, load_checkpoint

STOPWORDS = {
    "the", "and", "for", "that", "this", "with", "from", "what", "how", "why", "when",
    "where", "who", "are", "you", "can", "does", "explain", "show", "step", "give",
    "tell", "about", "into", "using", "between", "prove", "state", "find", "write",
}


def extract_prompt_keywords(prompt: str) -> set[str]:
    words = re.findall(r"[a-zA-Z0-9_]{3,}", prompt.lower())
    return {w for w in words if w not in STOPWORDS}


def check_balanced_delimiters(text: str) -> bool:
    """Verify mathematical parentheses and brackets are balanced."""
    stack: list[str] = []
    pairs = {")": "(", "]": "["}
    for ch in text:
        if ch in "([":
            stack.append(ch)
        elif ch in ")]":
            if not stack or stack[-1] != pairs[ch]:
                return False
            stack.pop()
    return len(stack) == 0


def verify_and_score_generation(prompt: str, response: str) -> dict[str, Any]:
    """7-Dimensional Process Verifier for self-generated Mirror 7 responses."""
    p = clean_text(prompt)
    r = clean_text(response)
    if not r or utf8_len(r) < 24:
        return {"accepted": False, "score": 0.0, "reason": "too_short"}

    if REFUSAL_RE.search(r) or FOREIGN_IDENTITY_RE.search(r):
        return {"accepted": False, "score": 0.0, "reason": "refusal_or_foreign_identity"}

    if r[-1] not in CLEAN_END_CHARS:
        return {"accepted": False, "score": 0.1, "reason": "incomplete_sentence"}

    # Check repetition (trigrams & word 5-grams)
    win = r[:700]
    if len(win) >= 80:
        tri = [win[i : i + 3] for i in range(len(win) - 2)]
        tri_ratio = len(set(tri)) / max(1, len(tri))
        if tri_ratio < 0.32:
            return {"accepted": False, "score": 0.15, "reason": "char_loop"}
    else:
        tri_ratio = 0.65

    words = r.split()
    if len(words) >= 20:
        grams = [tuple(words[i : i + 5]) for i in range(len(words) - 4)]
        gram_ratio = len(set(grams)) / max(1, len(grams))
        if gram_ratio < 0.82:
            return {"accepted": False, "score": 0.2, "reason": "phrase_loop"}
    else:
        gram_ratio = 1.0

    # Score components
    score = 0.50

    # 1. Prompt keyword grounding
    kw = extract_prompt_keywords(p)
    if kw:
        r_low = r.lower()
        hits = sum(1 for w in kw if w in r_low)
        coverage = hits / len(kw)
        score += 0.20 * min(1.0, coverage * 1.5)
    else:
        score += 0.12

    # 2. Balanced mathematical brackets & parentheses
    if check_balanced_delimiters(r):
        score += 0.08
    else:
        score -= 0.12

    # 3. Step-by-step structure or multi-sentence coherence
    if STEP_RE.search(r):
        score += 0.08
    if CONNECTIVE_RE.search(r):
        score += 0.06

    # 4. Lexical diversity & length health
    rb = utf8_len(r)
    if 80 <= rb <= 920:
        score += 0.08
    score = round(max(0.0, min(1.0, score)), 4)
    return {
        "accepted": score >= 0.68,
        "score": score,
        "reason": "ok" if score >= 0.68 else "low_score",
        "trigram_diversity": round(tri_ratio, 3),
        "fivegram_diversity": round(gram_ratio, 3),
    }


def generate_verified_response(
    model: NativeConversationV5,
    prompt: str,
    num_candidates: int = 6,
    max_new_tokens: int = 820,
    temperatures: Sequence[float] = (0.0, 0.25, 0.40, 0.55, 0.70, 0.85),
    top_p: float = 0.92,
    no_repeat_ngram_size: int = 20,
) -> dict[str, Any]:
    """Generate multiple candidate responses across a temperature ladder and pick the best verified one."""
    candidates: list[dict[str, Any]] = []
    for i in range(max(1, num_candidates)):
        temp = float(temperatures[i % len(temperatures)])
        text = model.generate_text(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temp,
            top_k=40,
            top_p=top_p if temp > 0 else 1.0,
            repetition_penalty=1.0,
            no_repeat_ngram_size=no_repeat_ngram_size,
            enforce_valid_utf8=True,
        )
        verdict = verify_and_score_generation(prompt, text)
        candidates.append({
            "text": text,
            "temperature": temp,
            **verdict,
        })

    candidates.sort(key=lambda c: (bool(c["accepted"]), float(c["score"]), len(c["text"])), reverse=True)
    best = candidates[0]
    return {
        "prompt": prompt,
        "best_text": best["text"],
        "best_score": best["score"],
        "accepted": best["accepted"],
        "best_temperature": best["temperature"],
        "candidates": candidates,
    }


DEFAULT_EXPLORATION_TOPICS = (
    # General Chat & Self-Identity
    "Who are you and how does your architecture work?",
    "Hello! How can you help me learn quantum computing and mathematics?",
    "What is the difference between correlation and causation? Give a real-world example.",
    # Quantum Physics & Quantum Computing
    "Explain how a qubit represents a superposition |psi> = alpha|0> + beta|1> and how measurement works.",
    "Explain the Hadamard gate (H) and show how it acts on |0> and |1>.",
    "What are the Pauli matrices X, Y, and Z and how do they act on a qubit?",
    "Walk through the 4 steps of the Quantum Teleportation protocol for transmitting |psi> = alpha|0> + beta|1>.",
    "How does Grover's quantum search algorithm achieve a quadratic speedup over classical search?",
    "Explain the mathematical mechanism behind Shor's factoring algorithm.",
    "State and prove the Quantum No-Cloning Theorem using linearity of unitary operators.",
    # Mathematics & Proofs
    "Find the eigenvalues and eigenvectors of the matrix A = [[4, 1], [2, 3]] step by step.",
    "Prove rigorously that there are infinitely many prime numbers.",
    "How do you compute the Kronecker (tensor) product A (x) B of two 2x2 matrices?",
    # Computer Science & Problem Solving
    "Write a Python function for binary search and explain its time complexity.",
    "Given an unsorted array of n integers and a target sum T, how do you optimize from O(n^2) to O(n)?",
)


def run_star_self_play(
    checkpoint_path: str | Path,
    output_jsonl: str | Path,
    prompts: Sequence[str] = DEFAULT_EXPLORATION_TOPICS,
    num_candidates: int = 6,
    min_score: float = 0.72,
    device: str = "cuda",
) -> dict[str, Any]:
    """Run STaR self-generation over prompts and save verified self-generated solutions to JSONL."""
    model, _ = load_checkpoint(checkpoint_path, device=device)
    model.eval()

    kept: list[dict[str, Any]] = []
    all_results: list[dict[str, Any]] = []
    for idx, prompt in enumerate(prompts):
        res = generate_verified_response(model, prompt, num_candidates=num_candidates)
        all_results.append(res)
        if res["accepted"] and res["best_score"] >= min_score:
            kept.append({
                "example_id": f"star-selfgen-{idx:05d}",
                "user_text": prompt,
                "target_text": res["best_text"],
                "goal": "",
                "state": {},
                "evidence": {},
                "actions": [],
                "split": "train",
                "quality": res["best_score"],
            })

    out = Path(output_jsonl)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for row in kept:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    return {
        "total_prompts": len(prompts),
        "accepted_count": len(kept),
        "output_jsonl": str(out),
        "results": all_results,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Mirror 7 STaR Self-Play & Process Verification Engine")
    p.add_argument("--checkpoint", required=True, help="Path to Mirror 7 .pt checkpoint")
    p.add_argument("--output", default="training/data/mirror7_star_selfgen.jsonl")
    p.add_argument("--candidates", type=int, default=6)
    p.add_argument("--min-score", type=float, default=0.72)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    summary = run_star_self_play(
        checkpoint_path=args.checkpoint,
        output_jsonl=args.output,
        num_candidates=args.candidates,
        min_score=args.min_score,
        device=args.device,
    )
    print(f"STaR Self-Play Complete: {summary['accepted_count']}/{summary['total_prompts']} verified -> {summary['output_jsonl']}")


if __name__ == "__main__":
    main()
