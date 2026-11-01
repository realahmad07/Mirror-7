import json

with open('artifacts/model_comparison_100.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

print('=' * 85)
print('100 HELD-OUT QUESTIONS — 4-WAY COMPARATIVE BENCHMARK')
print('=' * 85)

header = f"{'Model':<20} | {'Decoding Mode':<25} | {'RepRatio':<8} | {'VocabDiv':<8} | {'4g-F1':<7} | {'PromptJac':<9} | {'EOS-Rate':<8}"
print(header)
print('-' * len(header))

for model_name, modes in data.items():
    for mode_name, details in modes.items():
        s = details['summary']
        print(f"{model_name:<20} | {mode_name:<25} | {s['avg_repetition_ratio']:8.4f} | {s['avg_vocabulary_diversity']:8.4f} | {s['avg_char_4gram_f1']:7.4f} | {s['avg_prompt_jaccard']:9.4f} | {s['proper_eos_rate']:8.4f}")

print('\n' + '=' * 85)
print('DETAILED QUALITATIVE COMPARISON ACROSS REPRESENTATIVE QUESTIONS')
print('=' * 85)

# Select 4 representative prompts:
# 1. Open-ended conversational
# 2. Technical / coding
# 3. Direct inquiry / philosophy
# 4. Short direct instruction
sample_indices = [1, 2, 4, 15]

for idx in sample_indices:
    p = data['Baseline (Epoch 5)']['Unconstrained (Raw Sampling)']['outputs'][idx - 1]
    print(f"\n[QUESTION {idx}] {p['prompt'][:120]}")
    print(f"TARGET REF: {p['target'][:120]}...\n")
    for model_name in ['Baseline (Epoch 5)', 'V4 (Epoch 6)']:
        for mode_name in ['Unconstrained (Raw Sampling)', 'Constrained (Penalized Decoding)']:
            out = data[model_name][mode_name]['outputs'][idx - 1]
            print(f"  --> {model_name} [{mode_name}]:")
            print(f"      {out['prediction']}")
