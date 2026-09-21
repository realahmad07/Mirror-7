# Mirror 7 Roadmap

The first 75+ hour engineering/research phase is complete. Work from this point should be a new research program with frozen baselines rather than arbitrary numbered phases.

## Current engineering target

Keep the backend and UI integration stable while the research layer evolves independently.

## Research priority 1 — Native language realization

The unresolved high-value problem is strong native free-running language realization.

Required next milestone:

- stronger held-out benchmark;
- frozen evaluator and dataset fingerprint;
- a new mechanism targeting the teacher-forcing/free-running gap;
- multi-seed confirmation;
- direct comparison with the frozen 10.57M baseline;
- no production promotion from loss alone.

## Research priority 2 — Native semantic-state generalization

Experiments 6–7 indicate a depth/generalization bottleneck.

The next experiment should ask whether learned transformations can transfer to richer naturally occurring semantic relations without sacrificing compositional performance.

A new benchmark is more valuable than blind scaling.

## Research priority 3 — Independent evaluation

Move important claims toward:

- sealed benchmark;
- evaluator hash;
- training/evaluation separation;
- external reproduction.

## Compute rule

Use A100 time only after a falsifiable hypothesis is written down.

Preferred loop:

1. small pilot;
2. negative controls;
3. frozen baseline;
4. multi-seed replication;
5. scale only if the evidence warrants it.

## Product path

The current backend provides a stable interface for the UI. Product/UI development does not need to wait for native language research to finish.

## Avoid

- repeating the same failed scratch ByteGRU scaling path without a new mechanism;
- treating synthetic benchmark scores as general intelligence scores;
- deleting failed experiments;
- changing benchmark/evaluator definitions after seeing results;
