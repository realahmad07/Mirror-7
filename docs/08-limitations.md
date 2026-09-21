# Limitations and Research Boundary

## Not established

The current repository does not establish:

- AGI;
- human-level intelligence;
- consciousness;
- unrestricted real-world generalization;
- a ChatGPT-class standalone language model;
- autonomous scientific discovery in the open world;
- safe unrestricted self-modification;
- broad multimodal competence outside bounded phase environments.

## Language-model limitation

The direct scratch language-model route remains the largest practical gap.

The 10M-class ByteGRU learned the supervised target distribution under teacher forcing, but free-running generation remained weak. The problem therefore cannot be reduced to “more epochs” or “more parameters”.

The current engineering compromise is an optional pretrained response realization layer while native-brain research continues.

## Native-brain limitation

Experiments 1–7 demonstrate controlled semantic state and transformation behavior, but their environments are synthetic and tasks are bounded.

The evidence supports statements such as “the model learned the defined transformation relation”, not “the model learned general reasoning”.

## Evaluation limitation

Many historical benchmarks are internally authored. Some use synthetic environments and project-specific acceptance criteria.

They are useful engineering and research evidence, but weaker than independent external reproduction.

## Compute limitation

The training program used limited compute and selective A100 access in Colab.

This encouraged small falsifiable experiments instead of massive blind sweeps.

An untested architecture is not disproven merely because it did not receive large-scale compute. Conversely, a small positive result is not a scalable result without further evidence.

## Data limitation

The curated corpus contains 14,627 examples.

That is suitable for controlled first experiments, but it is far too small to support a broad world-knowledge claim for a scratch model.

## Backend limitation

The backend can host and orchestrate Mirror research modules. It does not create cognition by itself.

## Security limitation

The repository is public. Training notebooks, logs, and debugging cells must be reviewed for credentials before publication.

The September 2026 training notebook used in this work contained a personal access credential in one Colab cell. The published repository copy is sanitized.

Any credential previously exposed in a local notebook should be revoked and replaced.
