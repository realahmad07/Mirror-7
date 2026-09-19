# Mirror 7 — Targeted Continuous Upgrade Loop

## Phases 299–306

299: capability frontier and weighted gap tracking.
300: deterministic train/held-out/regression curriculum generation.
301: bounded capability upgrade campaigns.
302: persistent improvement history and stagnation detection.
303: cross-capability regression protection.
304: automatic frontier scheduling with rotating seeds.
305: continuous frontier upgrade controller.
306: connection to the existing source-redesign engine for an actual sequence-capability upgrade path.

## Targeted loop

measure capability -> rank frontier gap -> choose fresh evaluation seed -> launch bounded improvement campaign -> validate candidate -> protect other capabilities -> promote only verified gains -> update frontier -> choose next gap -> repeat

## Current boundary

The loop is designed to keep improving toward a configurable capability target. Phase 306 demonstrates the loop connected to one real source-redesign path. It is still bounded by available capability adapters, evaluation families, resource budgets, and verified promotion gates. It does not imply automatic attainment of frontier-model or AGI capability.
