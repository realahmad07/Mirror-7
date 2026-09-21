# Experiment 6 — Learned Operation Semantics

Experiment 6 tests whether Mirror's native state-transition mechanism can learn
the meaning of operations from experience instead of relying on semantically
named operations such as INVERT/RESET.

## Design

- Opaque operation IDs: OP_A through OP_F.
- No operation-name semantics are encoded in the model.
- A learned operation embedding conditions the state transition.
- Chains vary from length 1 through 5.
- Held-out concept identities and structured operation combinations are never
  used for training.
- Evaluation measures target similarity, contrastive margin, order sensitivity,
  and identity behavior.

The pilot is deliberately small and isolated from the Exp-5 checkpoint format.

## Colab

From the repository root:

```bash
python -m training.experiment6_learned_operations \
  --output /content/drive/MyDrive/Mirror7/mirror7_exp6_pilot.pt \
  --epochs 60 --seed 42
```

Run the pilot only. Do not start 3-seed validation until the frozen benchmark
shows a predefined improvement over the Exp-5 mechanism.

## Gate

Proceed only if the pilot demonstrates:

1. positive held-out margin;
2. materially above-chance held-out accuracy;
3. positive order margin against swapped operations;
4. stable identity behavior where an identity composition is included;
5. no collapse where all operation sequences map to the same state.

A passing pilot is evidence for learned operation-conditioned state
transformation on this benchmark, not evidence of general reasoning or AGI.
