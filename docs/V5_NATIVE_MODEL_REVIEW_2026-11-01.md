# V5 Native Model Review — 2026-11-01

## Scope and evidence

This review covers the uploaded V5-enhanced source tree and its automated regression suite. The local fixes described below were made in the extracted review working copy. This GitHub documentation commit does **not** include those Python source changes; integrating the source tree into `main` remains pending. It does **not** claim that a full Google Colab GPU run has been completed. The controlled V5 comparison included in the supplied project notes was a small 30-step CPU experiment (128 training examples and 32 validation examples); it is useful for finding implementation regressions, not for proving useful conversational capability.

## Architecture changes retained and completed

The enhanced V5 model uses a bidirectional GRU encoder, multi-head cross-attention, output context gating, and scheduled sampling. This review found that two declared capabilities were not actually applied in the initial implementation: configured dropout was not used, and context gating affected output fusion but did not gate the context entering the recurrent decoder state.

The implementation now:

- applies configured dropout to encoder and decoder embeddings;
- adds an optional recurrent-context gate before the decoder GRU cell, while retaining the existing output-fusion gate;
- enables that recurrent gate in the enhanced-model factory and GPU-training default;
- validates model dimensions, bidirectional hidden-size parity, attention-head divisibility, dropout, sequence length, and scheduled-sampling configuration;
- keeps `recurrent_context_gating=False` as the config default for old checkpoints whose serialized config predates the new gate.

### Parameter counts and checkpoint compatibility

| Variant | Parameters | Evidence |
|---|---:|---|
| Previous enhanced V5 (without recurrent-context gate) | 924,739 | Instantiated from the prior config; old checkpoint strict-load passed |
| Enhanced V5 with recurrent-context gate | 974,019 | Instantiated from the updated enhanced config |
| Earlier V5 checkpoint | 957,251 | Existing checkpoint strict-load passed |
| V3/V4 checkpoint | 990,915 | Existing checkpoint strict-load passed |

The new recurrent gate adds parameters; it does not overwrite or convert the legacy checkpoints. The `.pt` files used for compatibility checks remain local artifacts and are not part of this source commit.

## Training-pipeline correctness improvements

The portable GPU trainer now:

- buckets examples using the longer of source and target byte lengths;
- scales incomplete final gradient-accumulation groups by their true group size;
- seeds Python and PyTorch RNGs for repeatable setup;
- rejects empty train/validation splits;
- stores the dataset fingerprint and rejects a resume when a recorded fingerprint does not match;
- preserves optimizer and AMP scaler state in checkpoint training metadata.

A resumed run starts a new scheduler stage using the requested learning-rate configuration; it is not an exact continuation of the prior scheduler's internal timeline. GPU/Colab training and throughput have **not** been verified by this local test run.

## Teacher-data and evaluation safeguards

The knowledge-transfer collector now assigns splits from a normalized prompt hash, so duplicate prompts with different task IDs or categories do not silently land in different data splits. Split ratios are validated.

The response validator is a **heuristic structural filter**. It checks length, repetition, limited Python syntax and simple derivation markers. It does not fact-check statements, prove mathematical correctness, or establish semantic correctness. Its quality score is not a truth score; the metrics explicitly record factual verification as not performed and semantic correctness as not verified.

The local Hugging Face teacher interface validates generation arguments and omits `temperature` in greedy mode rather than passing `None` to Transformers.

The teacher-disconnection verifier now requires a real checkpoint, loads a supported native checkpoint, performs local generation with teacher providers disconnected, and invokes the held-out evaluator. A successful disconnection test demonstrates that this tested code path can execute without the teacher; it does not establish high answer quality.

## Reasoning-discovery sandbox boundary

The candidate-code evaluator now restricts the accepted AST subset and applies execution/time budgets to ordinary test-case execution. It remains an **in-process restricted evaluator**, not a process, container, or operating-system security boundary. Do not execute hostile third-party code in it. Untrusted code requiring adversarial isolation must run in a separately isolated environment.

## Test results

The uploaded/local review working-copy regression suite reported **95 passed** after the changes in this review. This is not a CI/test result for the GitHub `main` tree. The targeted V5, GPU-pipeline, knowledge-transfer, evaluation, and reasoning-discovery tests also passed during development. The test suite exercises implementation behavior and compatibility; it does not replace GPU training, human review, factual evaluation, or broad generalization benchmarks.

## Controlled comparison interpretation

The supplied small comparison records weaker train/validation loss for the enhanced variant than the baseline at the end of its short 30-step CPU experiment, and generated text remained repetitive or incoherent. This is not evidence of a quality improvement. The architecture corrections in this review fix identified implementation defects; they still require a controlled retraining and held-out comparison before a performance claim is justified.

## Training readiness

**Ready for:** a short Colab GPU smoke run, then a controlled training/evaluation experiment with fixed data fingerprints and preserved baselines.

**Not yet established:** successful full GPU training, improved free-running conversational quality, strong factuality, broad reasoning generalization, or AGI.

## Architecture invariants

The native language model remains a learned realization component within Mirror 7. This work does not replace Mirror's structured state, prediction, reasoning, computational-discovery, memory, persistence, or bounded self-change/self-rewrite mechanisms with a pretrained teacher. Teacher models are data-generation/training aids, not a required production inference dependency.
