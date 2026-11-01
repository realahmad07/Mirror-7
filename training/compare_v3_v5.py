"""
Side-by-side comparison of V3 baseline vs V5 after 5 epochs.
Compares metrics on both dev and untouched splits, plus qualitative examples.
"""
import json
import sys
import os
import argparse


def load_benchmark(path):
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def compare_metrics(v3_data, v5_data, split_name):
    v3 = v3_data['summary']
    v5 = v5_data['summary']
    
    metrics = [
        'benchmark_loss', 'proper_eos_rate', 'avg_repetition_ratio',
        'avg_vocabulary_diversity', 'avg_char_4gram_f1', 'avg_prompt_jaccard',
        'mean_bytes'
    ]
    
    print(f"\n{'='*70}")
    print(f" METRICS COMPARISON — {split_name} split (100 examples)")
    print(f"{'='*70}")
    print(f"{'Metric':<30} {'V3 Baseline':>14} {'V5 (5ep)':>14} {'Delta':>14}")
    print(f"{'-'*30} {'-'*14} {'-'*14} {'-'*14}")
    
    for m in metrics:
        v3_val = v3.get(m, 0)
        v5_val = v5.get(m, 0)
        delta = v5_val - v3_val
        sign = '+' if delta > 0 else ''
        # For loss, lower is better; for others, higher is generally better
        if m == 'benchmark_loss':
            indicator = ' ✓' if delta < 0 else ' ✗' if delta > 0 else ''
        elif m in ('avg_repetition_ratio',):
            indicator = ' ✓' if delta <= 0 else ' ✗'
        else:
            indicator = ' ✓' if delta > 0 else ' ✗' if delta < 0 else ''
        
        print(f"{m:<30} {v3_val:>14.4f} {v5_val:>14.4f} {sign}{delta:>13.4f}{indicator}")


def compare_examples(v3_data, v5_data, split_name, n=10):
    v3_outputs = v3_data.get('outputs', [])
    v5_outputs = v5_data.get('outputs', [])
    
    if not v3_outputs or not v5_outputs:
        print(f"\nNo per-example outputs available for {split_name} comparison")
        return
    
    print(f"\n{'='*70}")
    print(f" QUALITATIVE COMPARISON — {split_name} split (first {n} examples)")
    print(f"{'='*70}")
    
    # Find examples with largest F1 improvement
    pairs = []
    for v3e, v5e in zip(v3_outputs, v5_outputs):
        delta_f1 = v5e.get('char_4gram_f1', 0) - v3e.get('char_4gram_f1', 0)
        delta_jac = v5e.get('prompt_jaccard', 0) - v3e.get('prompt_jaccard', 0)
        pairs.append((v3e, v5e, delta_f1, delta_jac))
    
    # Sort by F1 improvement
    pairs.sort(key=lambda x: x[2], reverse=True)
    
    print(f"\n--- Top {n//2} IMPROVED examples (by 4-gram F1) ---")
    for v3e, v5e, df1, djac in pairs[:n//2]:
        print(f"\nQ: {v3e['prompt'][:100]}")
        print(f"Target: {v3e['target'][:120]}")
        print(f"V3: {v3e['prediction'][:150]}")
        print(f"V5: {v5e['prediction'][:150]}")
        print(f"  F1: V3={v3e.get('char_4gram_f1',0):.3f} → V5={v5e.get('char_4gram_f1',0):.3f} (Δ={df1:+.3f})")
        print(f"  Jaccard: V3={v3e.get('prompt_jaccard',0):.3f} → V5={v5e.get('prompt_jaccard',0):.3f} (Δ={djac:+.3f})")
    
    print(f"\n--- Bottom {n//2} DEGRADED examples (by 4-gram F1) ---")
    for v3e, v5e, df1, djac in pairs[-(n//2):]:
        print(f"\nQ: {v3e['prompt'][:100]}")
        print(f"Target: {v3e['target'][:120]}")
        print(f"V3: {v3e['prediction'][:150]}")
        print(f"V5: {v5e['prediction'][:150]}")
        print(f"  F1: V3={v3e.get('char_4gram_f1',0):.3f} → V5={v5e.get('char_4gram_f1',0):.3f} (Δ={df1:+.3f})")
        print(f"  Jaccard: V3={v3e.get('prompt_jaccard',0):.3f} → V5={v5e.get('prompt_jaccard',0):.3f} (Δ={djac:+.3f})")


def compare_diagnostics(v3_diag_path, v5_diag_path):
    if not os.path.exists(v3_diag_path) or not os.path.exists(v5_diag_path):
        print("\nDiagnostic files not both available, skipping conditioning comparison")
        return
    
    v3d = json.load(open(v3_diag_path, encoding='utf-8'))
    v5d = json.load(open(v5_diag_path, encoding='utf-8'))
    
    print(f"\n{'='*70}")
    print(f" CONDITIONING DIAGNOSTIC COMPARISON")
    print(f"{'='*70}")
    
    metrics = ['avg_similarity_to_blank', 'avg_similarity_to_shuffled', 
               'avg_prompt_relevance', 'prompt_distinctness_score']
    
    print(f"{'Metric':<35} {'V3':>12} {'V5':>12} {'Delta':>12}")
    print(f"{'-'*35} {'-'*12} {'-'*12} {'-'*12}")
    
    for m in metrics:
        v3_val = v3d.get('summary', v3d).get(m, 0)
        v5_val = v5d.get('summary', v5d).get(m, 0)
        delta = v5_val - v3_val
        sign = '+' if delta > 0 else ''
        print(f"{m:<35} {v3_val:>12.4f} {v5_val:>12.4f} {sign}{delta:>11.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--v3-benchmark', required=True, help='V3 baseline benchmark JSON')
    parser.add_argument('--v5-benchmark', required=True, help='V5 benchmark JSON')
    parser.add_argument('--split-name', default='unknown', help='Split name for display')
    parser.add_argument('--v3-diagnostic', default='', help='V3 diagnostic JSON')
    parser.add_argument('--v5-diagnostic', default='', help='V5 diagnostic JSON')
    parser.add_argument('--examples', type=int, default=10, help='Number of examples to show')
    parser.add_argument('--output', default='', help='Output JSON with comparison')
    args = parser.parse_args()
    
    v3_data = load_benchmark(args.v3_benchmark)
    v5_data = load_benchmark(args.v5_benchmark)
    
    compare_metrics(v3_data, v5_data, args.split_name)
    compare_examples(v3_data, v5_data, args.split_name, args.examples)
    
    if args.v3_diagnostic and args.v5_diagnostic:
        compare_diagnostics(args.v3_diagnostic, args.v5_diagnostic)
    
    # Save comparison summary
    if args.output:
        v3s = v3_data['summary']
        v5s = v5_data['summary']
        comparison = {
            'split': args.split_name,
            'v3_checkpoint': v3s.get('checkpoint', ''),
            'v5_checkpoint': v5s.get('checkpoint', ''),
            'metrics': {}
        }
        for m in ['benchmark_loss', 'proper_eos_rate', 'avg_repetition_ratio',
                   'avg_vocabulary_diversity', 'avg_char_4gram_f1', 'avg_prompt_jaccard', 'mean_bytes']:
            comparison['metrics'][m] = {
                'v3': v3s.get(m, 0),
                'v5': v5s.get(m, 0),
                'delta': v5s.get(m, 0) - v3s.get(m, 0)
            }
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(comparison, f, indent=2)
        print(f"\nComparison saved to {args.output}")


if __name__ == '__main__':
    main()
