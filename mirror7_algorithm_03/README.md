# Algorithm 3 - Causal Experiment Designer

## Purpose
Maintains competing causal explanations for observed phenomena and chooses experiments/actions that most efficiently distinguish between them.

## API
- `add_hypothesis(model)`
- `observe(data)`
- `suggest_experiment(interventions)`
- `get_posteriors()`
