# Research Workstream A — Native Intelligence and Semantic Reasoning

## Scope

This workstream combines the two unresolved model-research tracks in
`docs/09-roadmap.md`: native language realization and native semantic-state
generalization. Product UI, API, authentication, and deployment surfaces are
outside this workstream.

The objective is measurable capability improvement, not more phase numbers or
larger checkpoints without a falsifiable hypothesis.

## Starting evidence

- The 10.57M-class byte language-model route showed prompt-conditioned behavior
  during teacher-forced evaluation, but free-running continuation was unstable.
- `training/free_running_eval.py` reports transparent lexical, repetition,
  termination, and byte-integrity diagnostics. These are not factual or
  semantic correctness scores.
- Native semantic-state Experiments 1–7 contain bounded synthetic results.
  Experiments 6–7 did not establish transfer to natural semantic relations.
- The old Experiment 6 batching path padded sequences with operation ID zero
  and executed every padded column. This meant shorter examples received extra
  OP_A transformations during training. The fix is implemented in this
  workstream; old checkpoints are not evidence for the corrected path.

## Implemented increment: variable-length semantics and evaluation integrity

The corrected Experiment 6 implementation:

1. Uses `-1` as an explicit right-padding sentinel.
2. Masks padded columns during composition and rejects non-suffix padding.
3. Seeds model initialization, data generation, and row order.
4. Deduplicates repeated `(concept, operation-chain)` training tasks and ranks
   predictions against distinct in-batch target alternatives. Target-equivalent
   vectors are masked so they cannot become false negatives.
5. Stores each training example's intermediate state trace and adds
   teacher-forced per-transition supervision, while keeping held-out traces out
   of optimization. The contrastive and transition-supervision weights are
   explicit benchmark parameters.
6. Fingerprints the generated semantic world, including final and intermediate
   target vectors plus train/held-out assignment.
7. Evaluates held-out predictions against held-out candidate targets, not a
   small arbitrary slice of the full dataset.
8. Measures swapped-order controls and representation-collapse diagnostics.
9. Provides a three-seed screening runner with per-seed checks.

Run the focused tests:

```bash
python -m pytest training/test_experiment6_learned_operations.py training/test_evaluate_experiment6.py -q
```

Run the three-seed screening experiment:

```bash
python -m training.evaluate_experiment6 --device cuda --epochs 240 --seeds 17,29,43 --output artifacts/exp6_multiseed_screening.json
```

## Structural transition-model ablation

A second Experiment 6 architecture is available for a controlled structural
comparison. The `linear_operator` variant gives each operation a learned matrix
and applies `normalize(tanh(state @ matrix.T))`, matching the *form* (not the
values) of the synthetic target generator. It is evaluated alongside the
original MLP with the same generated worlds, seeds, training budget and held-out
gate. This tests whether a more suitable transition inductive bias recovers
synthetic latent dynamics; even a pass would not establish natural semantic
understanding. The first operator run is invalid: the world and model shared a
seeded random stream, and matching initialization shapes copied hidden target
matrices into the model. The implementation now derives a separate model seed
and has a regression test asserting that learned initialization differs from
the benchmark's hidden matrices. The leaked run is excluded from evidence.

## Three-seed results

At 60 epochs with intermediate-transition supervision, both architectures passed
order-control and no-collapse checks across the three seeds, but all six
seed-runs failed the positive held-out-margin check:

| Architecture | Mean held-out top-1 | Mean positive similarity | Mean candidate margin | Mean order margin | Margin gate |
|---|---:|---:|---:|---:|---|
| MLP | 2.61% | 0.0262 | -0.3131 | 0.0128 | **FAIL (0/3)** |
| Linear operator | 2.06% | 0.0151 | -0.3037 | 0.0169 | **FAIL (0/3)** |

At 240 epochs, the models made a substantial improvement:

| Architecture | Held-out top-1 | Mean positive similarity | Mean candidate margin | Mean order margin | Margin gate |
|---|---:|---:|---:|---:|---|
| MLP | 34.96% | 0.2782 | -0.0852 | 0.2732 | **FAIL (0/3)** |
| Linear operator | 40.72% | 0.3012 | -0.0649 | 0.3005 | **FAIL (0/3)** |

Chance top-1 is about 0.75–0.81%, depending on seed candidate count. At
240 epochs, both architectures exceed chance on all seeds, produce positive
order margins, and avoid representation collapse. The linear operator is
stronger than the MLP on these aggregate synthetic metrics. However, all six
seed-runs still fail the unchanged requirement that the correct held-out target
rank above its hardest distractor with a positive margin. This is a substantial
synthetic generalization improvement, not a gate pass or evidence of
natural-language reasoning.

The next predeclared ablation increases the linear operator's contrastive
weight from `0.10` to `0.25` at the same 240-epoch budget, retaining all
three seeds and the exact same acceptance criteria. The 0.10 results remain
frozen for comparison.

## Natural semantic benchmark intake

CLUTRR was examined as a candidate natural-language relational reasoning
benchmark because it tests multi-hop kinship inference, held-out rule
combinations, and robustness to irrelevant/disconnected facts. It is **not
approved for inclusion in Mirror's active training/evaluation bundle at this
time**: the upstream repository declares CC-BY-NC 4.0, while the Hugging Face
mirror marks its license as unknown. Resolve the applicable rights and intended
commercial-use permissions before downloading, caching, transforming, or
redistributing the dataset. Do not treat the existing external API reference
in the repository as license approval.

## Acceptance boundaries

A screening report is only an implementation-level research gate. It is not a
claim that the learned operation model is useful or that reasoning has
generalized.

Promotion additionally requires:

- comparison against a frozen Experiment 5/7 baseline on the same data,
  hardware class, parameter budget, training steps, and evaluator;
- at least three seeds and retained per-seed outputs;
- a naturally occurring semantic-relation benchmark with disjoint train,
  validation, and test partitions;
- hard negative controls (wrong order, wrong relation, distractor concept,
  and unsupported relation);
- independent verification of every exact-answer task;
- a blinded evaluator or external reproduction before broad public claims.

## Remaining native-language work

The native free-running evaluator now emits versioned reports with checkpoint
SHA-256/size, raw prompt-file SHA-256, and the normalized selected-case
fingerprint; repeated seed IDs are rejected. These additions make runs
reproducible but do not add semantic correctness labels.

The natural-language generation track is still open. A new model is not
accepted on teacher-forced loss or lexical diversity alone. The next experiment
must freeze prompt and target fingerprints, use held-out prompts not used for
training or teacher generation, measure free-running outputs under fixed
decoding budgets, and include human-adjudicated or independently verifiable
quality labels. Recurrent, state-space, and compact-attention alternatives must
share the same data, parameter/compute budget, and evaluator.

## Native free-running evaluation hardening
The existing repository held-out split can be exported for a reproducible
**smoke run** (10 prompts across the ten evaluation dimensions):

```bash
python -m training.export_native_free_running_prompts \
  --partition held_out \
  --output artifacts/native_free_running_legacy_heldout_v1.jsonl

python -m training.free_running_eval \
  --checkpoint path/to/native_conversation_v5.pt \
  --prompts artifacts/native_free_running_legacy_heldout_v1.jsonl \
  --seeds 17,29,43 --temperature 0.7 \
  --max-new-tokens 384 --device cuda \
  --output artifacts/native_free_running_report.json
```

The exporter writes a manifest with the prompt-file hash and explicitly sets
`promotion_eligible=false`. The repository's held-out label is not proof that
these prompts are disjoint from the separate Google Drive training corpus.
Use this bundle for a smoke diagnosis only; do not use it to claim model
improvement or natural semantic understanding.


The V5 free-running evaluator records checkpoint, source-prompt-file and
selected-case fingerprints. It now also labels decoding mode correctly:
repeating different seeds under greedy decoding is deterministic replay, not
seed-variance evidence. Sampling runs report seed variation as sampling-only;
the model weights remain fixed.

A paired report comparator requires identical prompt/case fingerprints, seed
lists and decoding settings before calculating per-case/per-seed deltas for
format and lexical diagnostics. Those deltas are not semantic correctness,
factuality or user-preference scores. The evaluator still needs a frozen
source-disjoint prompt set and independent human/oracle quality labels before
it can support capability claims.

Compare two saved reports without loading a model:

```bash
python -m training.compare_free_running_reports \\
  --baseline artifacts/native_free_running_baseline.json \\
  --candidate artifacts/native_free_running_candidate.json \\
  --output artifacts/native_free_running_paired_comparison.json
```


## Status

**Implementation and evaluation hardening:** completed for this increment.

**Synthetic Experiment 6:** the 240-epoch, three-seed MLP and linear-operator
runs show substantial improvement over the 60-epoch results, but all seeds
still fail the positive held-out candidate-margin gate. The 0.25 operator
contrastive-weight ablation is predeclared and under evaluation; no gate has
been relaxed.

**Track A completion:** not complete. It still requires a passing synthetic
screen, frozen Experiment 5/7 baseline comparison, a license-approved natural
semantic benchmark, and human/independent external evaluation. Native-language
free-running quality also remains unmeasured on the current checkpoint by this
workstream.
