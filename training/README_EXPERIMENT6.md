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

Model initialization and row shuffling are seeded. Training deduplicates repeated
`(concept, operation-chain)` tasks and uses a duplicate-aware in-batch hard-negative
margin objective (default contrastive weight `0.10`, configurable with
`--contrastive-weight`). Target-equivalent vectors are masked rather than treated
as false negatives.
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
  --device cuda --epochs 60 --seeds 17,29,43 \
  --output artifacts/exp6_multiseed_screening.json
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
