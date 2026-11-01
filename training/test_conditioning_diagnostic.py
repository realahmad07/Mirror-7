from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from training.dataset import load_jsonl


def jaccard_similarity(text1: str, text2: str) -> float:
    w1 = set(text1.lower().split())
    w2 = set(text2.lower().split())
    if not w1 and not w2:
        return 1.0
    if not w1 or not w2:
        return 0.0
    return len(w1.intersection(w2)) / len(w1.union(w2))


def run_conditioning_diagnostic(
    checkpoint_path: str,
    dataset_path: str = "training/data/mirror_oasst2_curated_v1.jsonl",
    num_prompts: int = 20,
    model_version: str = "v3",
    max_new_bytes: int = 150,
    device: str = "cpu",
    output_path: str | None = None,
):
    """Diagnose how strongly the model's generation is conditioned on the prompt.

    Compares normal generation against:
      1. Blank prompt ("")
      2. Shuffled prompt (irrelevant prompt from another domain)

    If a model is unconditioned and drifting into its prior, it generates identical
    or near-identical text regardless of the prompt. A well-conditioned model produces
    distinct, prompt-responsive outputs.
    """
    if model_version.lower() == "v5":
        from training.native_conversation_v5 import load_checkpoint
    else:
        from training.native_conversation import load_checkpoint

    model, payload = load_checkpoint(checkpoint_path, device=device)
    model.eval()

    examples = load_jsonl(dataset_path)
    test_items = [x for x in examples if x.split == "test"]

    # Use first num_prompts development examples
    dev_items = test_items[:: max(1, len(test_items) // num_prompts)][:num_prompts]
    shuffled_prompts = [x.user_text for x in dev_items]
    random.seed(42)
    random.shuffle(shuffled_prompts)

    results = []
    normal_vs_blank_sims = []
    normal_vs_shuffled_sims = []
    normal_prompt_relevances = []

    for i, ex in enumerate(dev_items):
        prompt = ex.user_text
        shuffled_p = shuffled_prompts[i]

        # 1. Normal prompt generation
        gen_normal = model.generate_text(
            prompt,
            max_new_tokens=max_new_bytes,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.2,
            no_repeat_ngram_size=5,
        ).strip()

        # 2. Blank prompt generation
        gen_blank = model.generate_text(
            "",
            max_new_tokens=max_new_bytes,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.2,
            no_repeat_ngram_size=5,
        ).strip()

        # 3. Shuffled prompt generation
        gen_shuffled = model.generate_text(
            shuffled_p,
            max_new_tokens=max_new_bytes,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.2,
            no_repeat_ngram_size=5,
        ).strip()

        sim_blank = jaccard_similarity(gen_normal, gen_blank)
        sim_shuffled = jaccard_similarity(gen_normal, gen_shuffled)
        prompt_relevance = jaccard_similarity(gen_normal, prompt)

        normal_vs_blank_sims.append(sim_blank)
        normal_vs_shuffled_sims.append(sim_shuffled)
        normal_prompt_relevances.append(prompt_relevance)

        results.append(
            {
                "index": i + 1,
                "prompt": prompt[:100],
                "shuffled_prompt": shuffled_p[:100],
                "gen_normal": gen_normal,
                "gen_blank": gen_blank,
                "gen_shuffled": gen_shuffled,
                "similarity_to_blank": sim_blank,
                "similarity_to_shuffled": sim_shuffled,
                "prompt_relevance": prompt_relevance,
            }
        )

    summary = {
        "checkpoint": str(checkpoint_path),
        "model_version": model_version,
        "num_prompts": len(dev_items),
        "avg_similarity_to_blank": sum(normal_vs_blank_sims) / max(1, len(normal_vs_blank_sims)),
        "avg_similarity_to_shuffled": sum(normal_vs_shuffled_sims) / max(1, len(normal_vs_shuffled_sims)),
        "avg_prompt_relevance": sum(normal_prompt_relevances) / max(1, len(normal_prompt_relevances)),
        "prompt_distinctness_score": 1.0 - (sum(normal_vs_shuffled_sims) / max(1, len(normal_vs_shuffled_sims))),
    }

    report = {"summary": summary, "outputs": results}
    if output_path:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with out_p.open("w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

    return report


def main():
    p = argparse.ArgumentParser(description="Prompt-Conditioning Diagnostic")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--model-version", default="v3", choices=["v3", "v5"])
    p.add_argument("--dataset", default="training/data/mirror_oasst2_curated_v1.jsonl")
    p.add_argument("--num-prompts", type=int, default=20)
    p.add_argument("--max-new-bytes", type=int, default=150)
    p.add_argument("--output", default=None)
    args = p.parse_args()

    report = run_conditioning_diagnostic(
        checkpoint_path=args.checkpoint,
        dataset_path=args.dataset,
        num_prompts=args.num_prompts,
        model_version=args.model_version,
        max_new_bytes=args.max_new_bytes,
        output_path=args.output,
    )

    print("=" * 60)
    print(f"PROMPT-CONDITIONING DIAGNOSTIC SUMMARY ({args.model_version.upper()})")
    print("=" * 60)
    for k, v in report["summary"].items():
        if isinstance(v, float):
            print(f"  {k:35s}: {v:.4f}")
        else:
            print(f"  {k:35s}: {v}")
    print("=" * 60)


if __name__ == "__main__":
    main()
