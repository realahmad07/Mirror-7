# Experiment 6 — Learned Operation Semantics

Experiment 6 tests whether Mirror's native state-transition mechanism can learn
opaque operations and compose variable-length transformations.

## Correctness fix

The previous batching path padded short sequences with operation ID `0`. Because
the model executed every column, padding was incorrectly treated as a real
operation. The current implementation uses `-1` as a right-padding sentinel,
masks it during composition, and rejects malformed padding. This changes the
training semantics; checkpoints trained by the earlier implementation must not
be treated as directly comparable to this corrected run.

World generation and model initialization use independent, reproducible random
streams. Training deduplicates repeated `(concept, operation-chain)` tasks and
combines duplicate-aware in-batch hard-negative ranking (default contrastive
weight `0.10`, configurable with `--contrastive-weight`) with teacher-forced
supervision for every intermediate transition (default weight `1.0`, configurable
with `--transition-supervision-weight`). An optional free-running rollout loss
(default `0.0`, configurable with `--rollout-supervision-weight`) feeds each
predicted intermediate state into the next operation and supervises accumulated
rollout error using training traces only. Target-equivalent vectors are masked
rather than treated as false negatives. Intermediate traces from held-out rows
are never used for optimization and are included in the benchmark fingerprint.
The generated world, including target vectors and train/held-out assignment, has
a SHA-256 fingerprint recorded in checkpoints and evaluation reports.

## Pilot

From the repository root:

```bash
python -m training.experiment6_learned_operations \
  --output artifacts/mirror7_exp6_corrected_seed42.pt \
  --epochs 60 --seed 42
```

## Three-seed screening run

After the implementation tests pass, run:

```bash
python -m training.evaluate_experiment6 \
  --device cuda --epochs 240 --seeds 17,29,43 \
  --contrastive-weight 0.10 --transition-supervision-weight 1.0 \
  --output artifacts/exp6_multiseed_screening.json

python -m training.evaluate_experiment6 \
  --device cuda --epochs 240 --seeds 17,29,43 \
  --contrastive-weight 0.10 --transition-supervision-weight 1.0 \
  --transition-architecture linear_operator \
  --output artifacts/exp6_operator_screening.json

# Predeclared contrastive-weight ablation (same model, data and compute budget)
python -m training.evaluate_experiment6 \
  --device cuda --epochs 240 --seeds 17,29,43 \
  --contrastive-weight 0.25 --transition-supervision-weight 1.0 \
  --transition-architecture linear_operator \
  --output artifacts/exp6_operator_contrastive025_screening.json

# Free-running rollout supervision: own predictions become later operation inputs
python -m training.evaluate_experiment6 \
  --device cuda --epochs 240 --seeds 17,29,43 \
  --contrastive-weight 0.10 --transition-supervision-weight 1.0 \
  --rollout-supervision-weight 1.0 \
  --transition-architecture linear_operator \
  --output artifacts/exp6_operator_rollout1_screening.json
```

The default `mlp` transition is the shared MLP baseline. A structural
ablation matching the synthetic generator's per-operation transition form is
available as `--transition-architecture linear_operator`. It learns a separate
matrix per operation and applies `normalize(tanh(state @ matrix.T))`. The target
matrices are not copied into the model; the model-initialization RNG is derived
separately from the world-generation seed. An initial ablation run accidentally
reused the same stream and is invalidated; it is excluded from all reported
results. This is a declared inductive-bias comparison, not evidence of
natural-language semantics.

Two additional matched rollout-dose ablations are now predeclared in CI.
They retain the same three seeds, 240 epochs, architecture, contrastive weight,
teacher-forced transition weight, and evaluator, changing only rollout
supervision weight to 0.10 or 0.25:

```bash
python -m training.evaluate_experiment6 \
  --device cuda --epochs 240 --seeds 17,29,43 \
  --contrastive-weight 0.10 --transition-supervision-weight 1.0 \
  --rollout-supervision-weight 0.10 \
  --transition-architecture linear_operator \
  --output artifacts/exp6_operator_rollout010_screening.json

python -m training.evaluate_experiment6 \
  --device cuda --epochs 240 --seeds 17,29,43 \
  --contrastive-weight 0.10 --transition-supervision-weight 1.0 \
  --rollout-supervision-weight 0.25 \
  --transition-architecture linear_operator \
  --output artifacts/exp6_operator_rollout025_screening.json
```

These are dose-response experiments motivated by the observed degradation on
long (five-operation) held-out chains and the `4→5` suffix composition. They
are not assumed to improve results. All six structured-operator configurations
are included in the unchanged all-seed acceptance gate; no threshold is relaxed.

Use the same device, PyTorch version, data fingerprint, training budget, and
evaluator when comparing the two architectures. The evaluator reports held-out
candidate ranking, contrastive margin, swapped-order negative controls,
representation-collapse diagnostics, and per-seed benchmark fingerprints.

## Promotion gate

The screening report checks, independently for each seed:

1. Positive held-out candidate margin.
2. Accuracy above the candidate-ranking chance rate.
3. Positive score margin against swapped final operations.
4. No near-total representation collapse.

A screening pass is **not** promotion. Promotion remains blocked until the
corrected model is compared with a frozen Experiment 5/7 baseline under the same
benchmark and compute budget, and until natural semantic relations are tested
with disjoint train/validation/test cases plus adversarial controls.

There is no identity operation in the current OP_A–OP_F task grammar; therefore
an OP_A/OP_A sequence is not an identity test and is not reported as one.

The experiment is bounded synthetic evidence, not proof of open-domain
reasoning, general intelligence, or AGI.
