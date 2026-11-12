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
   vectors are masked so they cannot become false negatives; the contrastive
   weight is configurable and recorded in the benchmark config.
5. Fingerprints the generated semantic world, including target vectors and
   train/held-out assignment.
6. Evaluates held-out predictions against held-out candidate targets, not a
   small arbitrary slice of the full dataset.
7. Measures swapped-order controls and representation-collapse diagnostics.
8. Provides a three-seed screening runner with per-seed checks.

Run the focused tests:

```bash
python -m pytest training/test_experiment6_learned_operations.py training/test_evaluate_experiment6.py -q
```

Run the three-seed screening experiment:

```bash
python -m training.evaluate_experiment6 --device cuda --epochs 60 --seeds 17,29,43 --output artifacts/exp6_multiseed_screening.json
```

## Structural transition-model ablation

A second Experiment 6 architecture is available for a controlled structural
comparison. The `linear_operator` variant gives each operation a learned matrix
and applies `normalize(tanh(state @ matrix.T))`, matching the *form* (not the
values) of the synthetic target generator. It is evaluated alongside the
original MLP with the same generated worlds, seeds, training budget and held-out
gate. This tests whether a more suitable transition inductive bias recovers
synthetic latent dynamics; even a pass would not establish natural semantic
understanding.

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

The natural-language generation track is still open. A new model is not
accepted on teacher-forced loss or lexical diversity alone. The next experiment
must freeze prompt and target fingerprints, use held-out prompts not used for
training or teacher generation, measure free-running outputs under fixed
decoding budgets, and include human-adjudicated or independently verifiable
quality labels. Recurrent, state-space, and compact-attention alternatives must
share the same data, parameter/compute budget, and evaluator.

## Status

**Implementation hardening:** completed in this increment.

**Native intelligence / semantic-reasoning research outcome:** pending the
three-seed run, frozen-baseline comparison, and natural-semantic evaluation.
