# Mirror 7 — Current Status

## Repository state

The 75+ hour initial build and research cycle is complete.

| Area | Current state |
|---|---|
| Core research phases | Preserved in the phase tree with their tests and acceptance records. |
| Backend | Implemented and packaged under mirror7_backend/. |
| HTTP service | Implemented and UI-independent. |
| Response-generation boundary | Implemented through phases 355–366. |
| Pretrained response realization | Available as an optional language layer. |
| Native semantic-state research | Experiments 1–7 completed and documented. |
| Scratch language model | 10.57M ByteGRU trained and evaluated; not accepted as a strong free-running model. |
| Active CI | Reduced to five intentional workflows. |
| Historical workflows | Preserved under docs/archive/legacy-workflows/. |
| Training notebook | Sanitized full Colab notebook published under training/colab/. |

## Native-brain evidence

| Experiment | Result |
|---|---|
| 1 | Top-1 14.27%, Top-5 25.47%, semantic margin 0.2209, transition margin 0.1150 across three seeds. |
| 2 | Transition margin 0.1605 mean, with Top-1 decreasing to 12.93%. |
| 3 | Explicit state transformations became learnable on the defined benchmark. |
| 4B | Held-out Top-1 87.18% and compositional margin 0.2967 on the defined benchmark. |
| 5 | 60.71% mean accuracy on bounded variable-length chains. |
| 6 | Opaque operation learning exposed a depth/generalization bottleneck. |
| 7 | 63.33% length-5 and 65.56% unseen-concept mean; matched the frozen primary baseline. |

## Research boundary

These are bounded engineering and synthetic research results. They do not establish AGI, human-level intelligence, unrestricted open-world generalization, or a ChatGPT-class standalone language model.

## Authoritative documentation

- README.md
- docs/04-training.md
- docs/05-evaluation.md
- docs/research/
- docs/06-testing-ci.md
- docs/08-limitations.md
- docs/09-roadmap.md
