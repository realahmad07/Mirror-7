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

## Universal intelligence pipeline boundaries

The Phase 1–9 core now has a passing bounded integration test, but important research claims remain open:

- Task understanding is a deterministic parser with lexical intent/domain cues and explicit abstention for unresolved cases; it is not unrestricted natural-language understanding.
- Capability matching relies on declared domains, intents and typed ports. It does not independently invent or discover an arbitrary algorithm from a task description.
- Execution requires a host-registered native handler. The child-process runner has time, CPU, address-space and payload bounds but is not a hardened boundary for hostile code; production untrusted execution needs the configured container or a stronger VM/microVM.
- Independent verifier identities are configured by the host. The framework ensures required checks are called and bound to the execution fingerprint, but organizational/implementation independence must be established by the operator.
- Composite learning is admitted only after repeated matching plans on distinct tasks and multiple successful verifiers. This proves a reusable structural composition on the tested cases, not transfer to arbitrary unseen domains.
- Cross-domain transfer and continuous open-world capability expansion still need dedicated held-out, adversarial and independently evaluated benchmarks.


## Universal intelligence pipeline boundaries

The Phase 1–9 implementation has a passing bounded integration test, but important research claims remain open:

- Task understanding is deterministic parsing with lexical intent/domain cues and abstention for unresolved cases; it is not unrestricted natural-language understanding.
- Capability matching uses declared domain, intent and typed ports. It does not independently invent arbitrary algorithms or native code from a task description.
- Execution requires a host-registered native handler. The bounded child-process runner is not a hardened boundary for hostile code; use the configured container/VM or stronger isolation for untrusted execution.
- Verifier independence is an explicit host/operator trust assignment. The framework enforces separate checker registration and required checks but cannot prove organizational independence itself.
- Composite learning was demonstrated in a bounded mathematics example across three distinct tasks. General cross-domain transfer and continuous open-world capability growth remain unproven.
- The engineering acceptance does not establish AGI, consciousness, human-level intelligence, unrestricted autonomous scientific discovery or a ChatGPT-class standalone language model.
