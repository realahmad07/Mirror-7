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

The matched contrastive-weight `0.25` ablation did not improve the
linear-operator result:

| Operator configuration | Held-out top-1 | Mean positive similarity | Mean candidate margin | Mean order margin | Margin gate |
|---|---:|---:|---:|---:|---|
| Contrastive weight 0.10 | 40.72% | 0.3012 | -0.0649 | 0.3005 | **FAIL (0/3)** |
| Contrastive weight 0.25 | 34.10% | 0.2602 | -0.1024 | 0.2583 | **FAIL (0/3)** |

Both configurations pass candidate-chance, positive-order and no-collapse checks
on all seeds, but all six runs fail the positive candidate-margin gate. The
higher contrastive weight worsened all aggregate held-out measures and is not
selected as the default.

## Next mechanism hypothesis: free-running rollout supervision

The existing intermediate-state objective teacher-forces the gold previous
state at each transition. That constrains local transitions but does not train
the error accumulation that occurs when a predicted state becomes the next
input. A new, zero-default `rollout_supervision_weight` adds a distinct loss:
the model predicts each transition and feeds its own prediction into the next
operation, comparing every predicted rollout state with the corresponding
training trace. Held-out traces remain excluded from optimization. The next
matched experiment uses the linear-operator architecture, contrastive weight
`0.10`, teacher-forced transition weight `1.0`, rollout weight `1.0`, 240
epochs and the same three seeds. The run completed but was worse than the frozen operator baseline: mean held-out top-1 fell to 31.98% and mean candidate margin to -0.1112. The rollout objective is not selected.

## Evaluation correction: equivalent target classes

A post-run audit found that the contrastive training loss masks target-equivalent
vectors as alternate positives, while the held-out evaluator previously counted
those same vectors as false negatives. The evaluator now applies the identical
cosine-equivalence threshold to held-out candidates, reports both unique target
classes and mean effective candidate count, excludes equivalent pairs from
candidate margins and collapse diagnostics, and retains the original four
promotion checks. New regression tests construct distinct tasks with identical
target vectors and verify that they cannot become false negatives.

All prior Experiment 6 scores are retained as historical results under the
older evaluator and must not be compared numerically with the corrected
evaluator. Re-run the matched configurations with the corrected evaluator
before making any selection or promotion decision.

## Corrected-evaluator rerun and failure localization

The three-seed rerun was executed with the equivalent-target correction active.
The per-seed reports were retrieved from the CI artifact and inspected:

| Configuration | Seed 17 margin / accuracy | Seed 29 margin / accuracy | Seed 43 margin / accuracy |
|---|---:|---:|---:|
| MLP, contrastive 0.10 | -0.1182 / 31.34% | -0.0770 / 35.34% | -0.0602 / 38.21% |
| Linear operator, contrastive 0.10 | -0.0863 / 35.82% | -0.0301 / 48.12% | -0.0783 / 38.21% |
| Linear operator, contrastive 0.25 | -0.1215 / 28.36% | -0.0690 / 40.60% | -0.1167 / 33.33% |
| Linear operator, rollout 1.0 | -0.1410 / 26.12% | -0.0733 / 41.35% | -0.1194 / 28.46% |

All configurations remain above candidate chance and have positive order margins
without representation collapse, but every seed fails the positive candidate
margin requirement. No configuration passes the screen. The 0.10 linear
operator remains the best of these tested configurations, not an accepted model.

To localize this failure rather than change thresholds, the evaluator now emits
ranking metrics stratified by seen/unseen concept, operation-chain length, and
the two held-out composition patterns (prefix 0→1 and suffix 4→5). Each
stratum ranks its examples against the same complete held-out candidate bank
and reports count, accuracy, mean candidate margin, and positive-margin rate.
This separates concept-transfer failures from composition-depth or specific
operator-combination failures without changing the main acceptance metric.

**Execution status:** the stratified evaluator and regression test have been
added to the research branch. The existing CI artifact predates this diagnostic
addition, so the new strata are not yet measured. Re-run the three-seed screen
and inspect all per-seed strata before choosing another architecture or loss.

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
still fail the positive held-out candidate-margin gate. The operator
contrastive-weight 0.25 ablation also failed all three seeds (34.10% held-out
top-1, mean candidate margin -0.1024), so the 0.10 configuration remains the
stronger frozen operator baseline. The free-running rollout-supervision ablation (weight 1.0) completed below the operator baseline (31.98% top-1, mean margin -0.1112) and was rejected. A subsequent evaluator audit fixed inconsistent treatment of equivalent target states; matched experiments must be rerun under the corrected evaluator before any model selection.

**Track A completion:** not complete. It still requires a passing synthetic
screen, frozen Experiment 5/7 baseline comparison, a license-approved natural
semantic benchmark, and human/independent external evaluation. Native-language
free-running quality also remains unmeasured on the current checkpoint by this
workstream.
