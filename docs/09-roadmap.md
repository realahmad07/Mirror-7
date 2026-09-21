# Roadmap After the 75+ Hour Build

The first 75+ hour construction phase is complete. The next work should be treated as a new research program rather than a continuation of arbitrary numbered phases.

## Track A — Public engineering surface

| Goal | State |
|---|:---:|
| Clean active checks | ✅ |
| Documentation hierarchy | ✅ |
| Backend smoke gate | ✅ |
| Release workflow | ✅ |
| Security guard | ✅ |
| UI/backend contract | ✅ |

## Track B — Native language realization

This is the highest-value unresolved model problem.

The next work should:

1. define a stronger held-out response benchmark;
2. freeze evaluator and data fingerprints;
3. test latent/state-prediction objectives rather than only raw byte prediction;
4. compare recurrent, state-space, and transformer alternatives under the same compute budget;
5. measure free-running generation directly;
6. retain a frozen baseline after every successful experiment.

## Track C — Native semantic-state scaling

The next question after Experiments 6–7 is not simply “make it bigger”.

The next question is whether learned state and transformation behavior can move from controlled synthetic operations to richer naturally occurring semantic relations without losing compositional generalization.

That requires new benchmarks, not just more epochs.

## Track D — Independent evaluation

Before making broad public intelligence claims:

- freeze the benchmark;
- publish the evaluator hash;
- separate training and evaluation artifacts;
- invite an external evaluator to reproduce or challenge the result.

## Track E — Product integration

The UI can be built against the current backend without waiting for the native model to finish.

## Track F — Compute strategy

A100 time should be spent only after an experiment has a falsifiable hypothesis.

Preferred sequence:

small pilot → negative controls → fixed baseline → multi-seed confirmation → scale.

Avoid another broad sweep of the failed scratch ByteGRU route without a new mechanism that directly addresses free-running instability.

## Next milestone

The next major milestone should be a measurable improvement in native response generation or a latent-state alternative that reduces dependence on raw autoregressive byte prediction while improving held-out free-running behavior.
